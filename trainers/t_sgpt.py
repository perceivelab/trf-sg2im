import math

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.ops import complete_box_iou_loss
from tqdm import tqdm
from transformers import top_k_top_p_filtering

from modules.vqvae.vqgan import VQModel
from trainers.t_base import TrainerBase
from utils.data import get_hw
from utils.graph import get_labels_from_graph
from utils.layout import encode_layout
from utils.misc import instantiate_from_config
from utils.transformer import configure_optimizers


class TrainerSGPT(TrainerBase):

    def __init__(self, base_lr, resume, scheduler=None, **kwargs):
        sgtrf_config = OmegaConf.load(kwargs.pop('sgtransformer_config'))
        sgtrf_config.params.update(
            {'n_objs': kwargs['n_objs'], 'n_rels': kwargs['n_rels'], 'pos_enc_dim': kwargs['pos_enc_dim']})
        self.sgtrf = instantiate_from_config(sgtrf_config).to(kwargs['device'])

        vqvae_config = OmegaConf.load(kwargs.pop('vqvae_config'))
        self.vqvae = instantiate_from_config(vqvae_config).to(kwargs['device'])
        self.codebook_size = vqvae_config.params.n_embed

        img_trf_config = OmegaConf.load(kwargs.pop('img_transformer_config'))

        assert img_trf_config.params.n_embd % img_trf_config.params.n_head == 0, "Embedding size not divisible by number of heads"

        self.img_trf = instantiate_from_config(
            img_trf_config).to(kwargs['device'])

        # TODO Rename n_emb to emb size
        assert img_trf_config.params['vocab_size'] == vqvae_config.params['n_embed'],  \
            f"Vocab size must be equal for VQVAE ({vqvae_config.params['n_embed']}) and Transformer {img_trf_config.params['vocab_size']} must have the same embedding size"

        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.eval()

        self.latent_size = get_hw(
            kwargs['image_size'], vqvae_config.params.ddconfig.ch_mult)

        self.losses = ["box", "iou", "idx", "label", "total"]

        self.sgtrf_opt = torch.optim.Adam(self.sgtrf.parameters(), lr=base_lr)
        self.img_trf_opt = configure_optimizers(self.img_trf, base_lr)
        super().__init__(resume, scheduler, **kwargs)
        self.box_beta = kwargs.pop('box_beta', 10)
        self.top_k_logits = kwargs.pop('top_k_logits', 50)
        self.gt_layout = kwargs.pop('use_gt_layout', False)
        self.top_k = kwargs.pop('top_k', 100)
        self.logger.info(
            f'Using {"GT" if self.gt_layout else "predicted"} layout')

        sched_params = kwargs.pop('lr_scheduler', {})
        self.get_scheduler(scheduler, self.img_trf_opt, **sched_params)

    def encode_to_z(self, images):
        with torch.no_grad():
            if isinstance(self.vqvae, VQModel):
                z, _, info = self.vqvae.encode(images)
                indices = info[-1]
                b = z.shape[0]
                h, w = z.shape[-2:]
                indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
                return indices
            return self.vqvae.encode(images)

    def decode_to_x(self, latents):
        with torch.no_grad():
            if isinstance(self.vqvae, VQModel):
                latents = self.vqvae.quantize.get_codebook_entry(
                    latents, shape=None)
                if len(latents.shape) == 3:
                    latents = rearrange(
                        latents, 'b (h w) c -> b h w c', h=int(latents.shape[1]**(1/2)))
                latents = rearrange(latents, 'b h w c -> b c h w')
            return self.vqvae.decode(latents)

    def _make_ckpt(self, epoch):
        ckpt = {
            'epoch': epoch,
            'sgtrf': self.sgtrf.state_dict(),
            'img_trf': self.img_trf.state_dict(),
            'sgtrf_opt': self.sgtrf_opt.state_dict(),
            'img_trf_opt': self.img_trf_opt.state_dict(),
        }

        return ckpt

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.logger.info(f'Training [{epoch} / {self.epochs}]')
            train_losses = self.train_one_epoch()

            {self.logger.log_metric(f'train/{k}_loss', v, epoch)
             for k, v in train_losses.items()}
            self.logger.info(f'Epoch {epoch}: {train_losses["total"]}')

            ckpt = self._make_ckpt(epoch)

            if (epoch + 1) % self.log_every == 0:
                self.logger.info(f'Validation [{epoch} / {self.epochs}]')
                val_losses = self.evaluate(epoch)

                if val_losses['total'] < self.best_loss:
                    self.best_loss = val_losses['total']
                    self.logger.save_ckpt(ckpt, 'best.pt')
                self.logger.save_ckpt(ckpt)

    def training_step(self, batch):

        images, graph, boxes = batch

        B = images.shape[0]
        labels = get_labels_from_graph(graph)
        boxes = rearrange(boxes, '(b o) c -> b o c', b=B)
        layout = {'boxes': boxes, 'labels': labels}

        layout_loss = 0
        if not self.gt_layout:
            pred_boxes, pred_label_logits = self.sgtrf(graph)
            pred_label_logits = rearrange(pred_label_logits, "b o c -> b c o")

            pred_labels = torch.argmax(pred_label_logits, dim=1)
            layout = {'boxes': pred_boxes, 'labels': pred_labels}
            box_loss = self.box_beta * F.l1_loss(pred_boxes, boxes)

            label_loss = F.cross_entropy(pred_label_logits, labels)

            iou_loss = complete_box_iou_loss(
                pred_boxes, boxes, reduction='mean')
            layout_loss = box_loss + label_loss + iou_loss

            self.sgtrf_opt.zero_grad()
            layout_loss.backward()
            self.sgtrf_opt.step()

        latents, idx_logits = self.forward_img_trf(images, layout)

        idx_loss = F.cross_entropy(idx_logits, latents)
        loss = idx_loss + layout_loss

        if self.gt_layout:
            losses = {'box': 0, 'iou': 0, 'idx': idx_loss.item(),
                      "label": 0, "total": loss.item()}
        else:
            losses = {'box': box_loss.item(), 'iou': iou_loss.item(), 'idx': idx_loss.item(),
                      "label": label_loss.item(), "total": loss.item()}

        self.img_trf_opt.zero_grad()
        idx_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.img_trf.parameters(), max_norm=1.0)
        self.img_trf_opt.step()
        self.scheduler_step()

        return losses

    @torch.no_grad()
    def evaluate(self, epoch=0):
        # Run one epoch of evaluation
        losses_epoch, layout, images = self.eval_one_epoch()

        # Log evaluation metrics
        for key in losses_epoch.keys():
            self.logger.log_metric(f'eval/{key}', losses_epoch[key], epoch)

        # Log sample images

        self.logger.log_images_w_boxes(
            images['gt'], layout['gt_boxes'], layout['gt_labels'], "eval", "gt")
        if self.gt_layout:
            self.logger.log_images_w_boxes(
                images['pred'], layout['gt_boxes'], layout['gt_labels'], "eval", "pred")
        else:
            self.logger.log_images_w_boxes(
                images['pred'], layout['pred_boxes'], layout['pred_labels'], "eval", "pred")
        return losses_epoch

    @torch.no_grad()
    def eval_one_epoch(self):
        epoch_losses = {k: 0 for k in self.losses}
        self.sgtrf.eval()
        self.img_trf.eval()
        layout = {'gt_boxes': [], 'pred_boxes': [],
                  'gt_labels': [], 'pred_labels': []}
        for batch in tqdm(self.val_loader):
            images, graph, boxes = batch

            B = images.shape[0]
            labels = get_labels_from_graph(graph)

            boxes = rearrange(boxes, '(b o) c -> b o c', b=B)
            layout = {'boxes': boxes, 'labels': labels}

            if not self.gt_layout:
                pred_boxes, pred_label_logits = self.sgtrf(graph)
                pred_label_logits = rearrange(
                    pred_label_logits, "b o c -> b c o")
                pred_labels = torch.argmax(pred_label_logits, dim=1)
                layout = {'boxes': pred_boxes, 'labels': pred_labels}

            cond = encode_layout(layout['boxes'], layout['labels'], no_sections=int(
                math.sqrt(self.codebook_size)))
            latents = self.encode_to_z(images)
            latents = rearrange(latents, 'b h w -> b (h w)')
            labels = get_labels_from_graph(graph)
            with torch.no_grad():
                idxs, idx_logits = self.sample(
                    cond, steps=self.latent_size**2, top_k=self.top_k)

            box_loss = label_loss = iou_loss = 0
            if not self.gt_layout:
                box_loss = F.mse_loss(pred_boxes, boxes)
                iou_loss = complete_box_iou_loss(
                    pred_boxes, boxes, reduction='mean')
                label_loss = F.cross_entropy(pred_label_logits, labels)

            idx_logits = rearrange(idx_logits, 's b e -> b e s')
            idx_loss = F.cross_entropy(idx_logits, latents)
            loss = box_loss + iou_loss + label_loss + idx_loss

            if not self.gt_layout:
                losses = {'box': box_loss.item(), 'label': label_loss.item(),
                          "iou": iou_loss.item(), "idx": idx_loss.item(), "total": loss.item()}
            else:
                losses = {'box': 0, 'label': 0, "iou": 0,
                          "idx": idx_loss.item(), "total": loss.item()}

            epoch_losses = {k: v + losses[k] for k, v in epoch_losses.items()}

        layout['gt_boxes'] = boxes
        layout['gt_labels'] = labels
        if not self.gt_layout:
            layout['pred_boxes'] = pred_boxes
            layout['pred_labels'] = pred_labels
        epoch_losses = {k: v / len(self.val_loader)
                        for k, v in epoch_losses.items()}

        idxs = rearrange(idxs, '(h w) b 1 -> b h w', h=self.latent_size)
        gen_images = self.decode_to_x(idxs)

        images = {'gt': images, 'pred': gen_images}
        return epoch_losses, layout, images

    def forward_img_trf(self, images, layout, cond_mask=None):
        latents = self.encode_to_z(images)
        if len(latents.shape) == 3:
            latents = rearrange(latents, 'b h w -> b (h w)')

        cond = encode_layout(layout['boxes'], layout['labels'], no_sections=int(
            math.sqrt(self.codebook_size)))
        xc = torch.cat((cond, latents), dim=1)
        idx_logits = self.img_trf(xc[:, :-1])
        idx_logits = idx_logits[:, cond.shape[1]-1:]

        # reshape output
        idx_logits = rearrange(idx_logits, 'b s e -> b e s')  # s = h * w
        return latents, idx_logits

    def sample(self, cond, steps, top_k, cond_mask=None, temperature=1.0, top_p=1.0):

        x = torch.tensor([], dtype=torch.int64, device=cond.device)
        cond_len = cond.shape[1]
        out_idxs = []
        out_logits = []
        block_size = self.img_trf.get_block_size()
        sampled = torch.cat((cond, x), dim=1)

        past = None
        for n in tqdm(range(steps), desc='Autoregressive sampling'):

            x_cond = sampled if sampled.size(
                1) <= block_size else sampled[:, -block_size:]  # crop context if needed
            logits = self.img_trf(x_cond)
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

    def train_one_epoch(self):
        epoch_losses = {k: 0 for k in self.losses}
        self.img_trf.train()
        for batch in tqdm(self.train_loader):
            losses = self.training_step(batch)
            epoch_losses = {k: v + losses[k]
                            for k, v in epoch_losses.items()}
        epoch_losses = {k: v / len(self.train_loader)
                        for k, v in epoch_losses.items()}
        return epoch_losses
