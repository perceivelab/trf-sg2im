import sys

sys.path.append("/home/rsortino/trf-sg2im")

import json
import math
from pathlib import Path

import torch
import wandb
import yaml
from einops import rearrange
import torch.nn.functional as F
from logzero import logger
from omegaconf import OmegaConf
from utils.data import get_hw
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import top_k_top_p_filtering
from modules.vqvae.vqgan import VQModel
from utils.arg_parser import get_args_parser, setup
from utils.data import construct_dgl_graph
from utils.logging import Logger
from utils.misc import count_parameters, instantiate_from_config
from utils.vis import draw_scene_graph
from utils.layout import encode_layout


def encode_scene_graphs(scene_graphs, vocab):
    """
    Encode one or more scene graphs using this model's vocabulary. Inputs to
    this method are scene graphs represented as dictionaries like the following:

    {
      "objects": ["cat", "dog", "sky"],
      "relationships": [
        [0, "next to", 1],
        [0, "beneath", 2],
        [2, "above", 1],
      ]
    }

    This scene graph has three relationshps: cat next to dog, cat beneath sky,
    and sky above dog.

    Inputs:
    - scene_graphs: A dictionary giving a single scene graph, or a list of
      dictionaries giving a sequence of scene graphs.

    Returns a tuple of LongTensors (objs, triples, obj_to_img) that have the
    same semantics as self.forward. The returned LongTensors will be on the
    same device as the model parameters.
    """
    if isinstance(scene_graphs, dict):
      # We just got a single scene graph, so promote it to a list
      scene_graphs = [scene_graphs]

    objs, triples, obj_to_img = [], [], []
    obj_offset = 0
    for i, sg in enumerate(scene_graphs):
      # Insert dummy __image__ object and __in_image__ relationships
      sg['objects'].append('__image__')
      image_idx = len(sg['objects']) - 1
      for j in range(image_idx):
        sg['relationships'].append([j, '__in_image__', image_idx])

      for obj in sg['objects']:
        obj_idx = vocab['object_name_to_idx'].get(obj, None)
        if obj_idx is None:
          raise ValueError('Object "%s" not in vocab' % obj)
        objs.append(obj_idx)
        obj_to_img.append(i)
      for s, p, o in sg['relationships']:
        pred_idx = vocab['pred_name_to_idx'].get(p, None)
        if pred_idx is None:
          raise ValueError('Relationship "%s" not in vocab' % p)
        triples.append([s + obj_offset, pred_idx, o + obj_offset])
      obj_offset += len(sg['objects'])
    objs = torch.tensor(objs, dtype=torch.int64)
    triples = torch.tensor(triples, dtype=torch.int64)
    obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64)
    return objs, triples, obj_to_img

def decode_to_x(vqvae, latents):
        with torch.no_grad():
            if isinstance(vqvae, VQModel):
                latents = vqvae.quantize.get_codebook_entry(
                    latents, shape=None)
                if len(latents.shape) == 3:
                    latents = rearrange(
                        latents, 'b (h w) c -> b h w c', h=int(latents.shape[1]**(1/2)))
                latents = rearrange(latents, 'b h w c -> b c h w')
            return vqvae.decode(latents)

def sample(img_trf, cond, steps, top_k, cond_mask=None, temperature=1.0, top_p=1.0):

        x = torch.tensor([], dtype=torch.int64, device=cond.device)
        cond_len = cond.shape[1]
        out_idxs = []
        out_logits = []
        block_size = img_trf.get_block_size()
        sampled = torch.cat((cond, x), dim=1)

        past = None
        for n in tqdm(range(steps), desc='Autoregressive sampling'):

            x_cond = sampled if sampled.size(
                1) <= block_size else sampled[:, -block_size:]  # crop context if needed
            logits = img_trf(x_cond)
            logits = logits[:, -1, :] / temperature
            out_logits.append(logits)
            if top_k is not None:
                logits = top_k_top_p_filtering(
                    logits, top_k=top_k, top_p=top_p)

            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            # append to the sequence and continue
            out_idxs.append(idx)
            sampled = torch.cat((sampled, idx), dim=1)
        del past
        sampled = sampled[:, cond_len:]  # cut conditioning off
        return torch.stack(out_idxs), torch.stack(out_logits)

def main(args):

    # make_reproducible(args.seed)

    # Data Loading
    config = OmegaConf.load(args.config)
    data_module = instantiate_from_config(config.datamodule)

    args.data['n_objs'] = data_module.n_objs
    args.data['n_rels'] = data_module.n_rels
    args.vocab = data_module.vocab

    writer = Logger(args.run_name, args.vocab)
    writer.log_hparams(args)

    # Setup SGTransformer model and checkpoint
    sgtrf_config = OmegaConf.load(config.trainer.params.sgtransformer_config)
    sgtrf_config.params.update(
        {'n_objs': data_module.n_objs, 'n_rels': data_module.n_rels, 'pos_enc_dim': data_module.pos_enc_dim})
    sgtrf = instantiate_from_config(sgtrf_config).to(args.device)

    checkpoint = torch.load(args.sgtrf_ckpt)
    start_epoch = checkpoint['epoch']
    logger.info(f'Loading SGTransformer at epoch {start_epoch} from {args.sgtrf_ckpt}')
    m, u = sgtrf.load_state_dict(
        checkpoint['sgtrf'], strict=False)
    logger.warning(f'Missing keys: {m}')
    logger.warning(f'Unexpected keys: {u}')

    # Setup VQVAE model and checkpoint
    vqvae_config = OmegaConf.load(config.trainer.params.vqvae_config)
    vqvae = instantiate_from_config(vqvae_config).to(args.device)
    codebook_size = vqvae_config.params.emb_size

    # Setup Image Transformer model and checkpoint

    img_trf_config = OmegaConf.load(config.trainer.params.img_transformer_config)
    assert img_trf_config.params.emb_size % img_trf_config.params.n_head == 0, "Embedding size not divisible by number of heads"
    img_trf = instantiate_from_config(
        img_trf_config).to(args.device)
    assert img_trf_config.params['vocab_size'] == vqvae_config.params['emb_size'],  \
        f"Vocab size must be equal for VQVAE ({vqvae_config.params['emb_size']}) and Transformer {img_trf_config.params['vocab_size']} must have the same embedding size"

    checkpoint = torch.load(args.img_trf_ckpt)
    start_epoch = checkpoint['epoch']
    logger.info(f'Loading Image Transformer at epoch {start_epoch} from {args.img_trf_ckpt}')
    m, u = img_trf.load_state_dict(
        checkpoint['img_trf'], strict=False)
    logger.warning(f'Missing keys: {m}')
    logger.warning(f'Unexpected keys: {u}')   
    
    print('*'*70)
    logger.info(f'Debugging mode: {args.debug}')
    logger.info(f'Logging model information')
    logger.info(
        f'SGTransformer type: {sgtrf.__class__.__name__}')
    logger.info(
        f'SGTransformer total parameters {(count_parameters(sgtrf) / 1e6):.2f}M')
    logger.info(
        f'VQVAE type: {vqvae.__class__.__name__}')
    logger.info(
        f'VQVAE total parameters {(count_parameters(vqvae) / 1e6):.2f}M')
    logger.info(
        f'Image Transformer type: {img_trf.__class__.__name__}')
    logger.info(
        f'Image Transformer total parameters {(count_parameters(img_trf) / 1e6):.2f}M')
    print('*'*70)

    sgtrf.eval()
    vqvae.eval()
    img_trf.eval()

    out_dir = Path("inference_output")
    out_dir.mkdir(exist_ok=True, parents=True) 

    scene_graphs_json = args.scene_graph
    print(f"Generating from scene graph: {scene_graphs_json}")

    latent_size = get_hw(
            config.datamodule.params.image_size, vqvae_config.params.ddconfig.ch_mult)

    # Load the scene graphs
    with open(scene_graphs_json, 'r') as f:
        scene_graphs = json.load(f)
    objs, triples, obj_to_img = encode_scene_graphs(scene_graphs, args.vocab)
    graph = construct_dgl_graph(objs, triples, data_module.pos_enc_dim).to(args.device)
    sgim = draw_scene_graph(objs, triples, args.vocab)
    Image.fromarray(sgim).save(out_dir / 'scene_graph.png')

    for i in range(10):
        with torch.no_grad():
            pred_boxes, pred_label_logits = sgtrf(graph)
        pred_label_logits = rearrange(
            pred_label_logits, "b o c -> b c o")
        pred_labels = torch.argmax(pred_label_logits, dim=1)
        layout = {'boxes': pred_boxes, 'labels': pred_labels}

        cond = encode_layout(layout['boxes'], layout['labels'], no_sections=int(
            math.sqrt(codebook_size)))

        with torch.no_grad():
            idxs, _ = sample(img_trf,
                cond, steps=latent_size**2, top_k=config.trainer.params.top_k_logits)

        idxs = rearrange(idxs, '(h w) b 1 -> b h w', h=latent_size)
        gen_images = decode_to_x(vqvae, idxs)

        # idxs, _, _, _ = model.autoregressive_generate(graph, steps=256, top_k=32, return_logits=True)

        # # idx_logits = rearrange(idx_logits, 's b e -> b e s') # s = h * w
        # # idxs = idxs.unsqueeze(1)
        # idxs = rearrange(idxs, '(h w) b 1 -> b h w', h=int(math.sqrt(idxs.shape[0])))
        # # pred_boxes = rearrange(pred_boxes, 'b s e -> (b s) e')
        # # pred_label_logits = rearrange(pred_label_logits, 'b s e -> b e s')
        
        # # labels = rearrange(labels, '(b o) -> b o', b=b)
        # gen_images = model.decode_latents(idxs)
        save_image(gen_images, out_dir / f'image_{i}.png', normalize=True)

    
if __name__ == '__main__':

    parser = get_args_parser()
    parser.add_argument('--scene-graph', default='scene_graphs/airplane.json',
                        help='Path to scene graph JSON file')
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        conf = yaml.safe_load(stream)

    args.data = conf['datamodule']['params']

    setup(args)

    wandb.init(
        config=conf,
        name=args.run_name,
        mode='disabled'
    )

    main(args)
