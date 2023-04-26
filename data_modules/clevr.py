# Implementation from https://github.com/roeiherz/CanonicalSg2Im
import collections
import json
import os
import pickle
from functools import partial

import PIL
from einops import rearrange
from torch.utils.data import Dataset
from utils.data import *

from data_modules.base import BaseDataModule
from data_modules.loader import *


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)


def encode_image():
    return T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def decode_image(rescale_image=False):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=[2.0, 2.0, 2.0]),
        T.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


class CLEVRDataModule(BaseDataModule):
    def __init__(self, collate_fn, config={}, data_dir="data", debug=False):
        super(CLEVRDataModule, self).__init__(data_dir)
        dataset_config = {
            "common": {
                "base_path": '{}/CLEVR'.format(data_dir),
                "max_objects": 1000,
                # "min_objects": 0,
                "debug": debug,
            },
            "train": {
                "h5_path": 'clevr_dialog_train_raw_minimal.json',
                "mode": "train",
            },
            "val": {
                "h5_path": 'clevr_dialog_val_raw_minimal.json',
                "mode": "val",
            },
            "class": CLEVRDialogDataset
        }

        config.update(dataset_config['common'])
        config.update(dataset_config['train'])

        self.train_ds = CLEVRDialogDataset(**config)
        config.pop('h5_path')
        config.pop('mode')
        config.update(dataset_config['val'])
        self.val_ds = CLEVRDialogDataset(**config)
        self.test_ds = self.val_ds

        if collate_fn == 'clevr_collate_fn':
            self.collate_fn = partial(clevr_collate_fn, vocab=self.vocab)
        elif collate_fn == 'dgl_clevr_collate_fn':
            self.collate_fn = partial(dgl_clevr_collate_fn, vocab=self.vocab)

    @property
    def vocab(self):
        return self.train_ds.vocab

    @property
    def n_objs(self):
        return len(self.vocab['object_name_to_idx'])

    @property
    def n_rels(self):
        return len(self.vocab['pred_idx_to_name'])


class CLEVRDialogDataset(Dataset):
    def __init__(self, h5_path, base_path, mode, image_size=(64, 64), mask_size=0,
                 normalize_images=True, max_objects=10, max_samples=None, include_dummies=False,
                 include_relationships=True, use_orphaned_objects=True, debug=False, learned_converse=False,
                 use_transitivity=False, use_converse=False, learned_transitivity=False, learned_symmetry=False,
                 dense_scenes=False, sort_ids=None, eval_func=None):
        super(CLEVRDialogDataset, self).__init__()

        self.image_dir = os.path.join(base_path, 'images')
        self.image_size = image_size
        self.use_transitivity = use_transitivity
        self.mode = mode

        # objects
        self.vocab = {}
        self.vocab["use_object_embedding"] = False
        self.vocab['pred_name_to_idx'] = {'__in_image__': 0, 'right': 1, "behind": 2, "front": 3, "left": 4,
                                          '__padding__': 5}
        self.vocab['pred_idx_to_name'] = {
            v: k for k, v in self.vocab['pred_name_to_idx'].items()}

        # attributes, currently ignored.
        self.vocab["attributes"] = {}
        self.vocab["attributes"]['shape'] = {
            '__image__': 0, 'cube': 1, 'sphere': 2, 'cylinder': 3}
        self.vocab["attributes"]["color"] = {'__image__': 0, 'gray': 1, 'red': 2, 'blue': 3, 'green': 4, 'brown': 5,
                                             'purple': 6, 'cyan': 7, 'yellow': 8}
        self.vocab["attributes"]["material"] = {
            '__image__': 0, 'rubber': 1, 'metal': 2}
        self.vocab["attributes"]["size"] = {
            '__image__': 0, 'small': 1, 'large': 2}

        self.vocab["reverse_attributes"] = {}
        for attr in self.vocab["attributes"].keys():
            self.vocab["reverse_attributes"][attr] = {
                v: k for k, v in self.vocab["attributes"][attr].items()}

        self.vocab['object_name_to_idx'] = {}
        obj_data = json.load(
            open(os.path.join(base_path, 'merged_objects.json')))

        self.vocab['object_name_to_idx'] = obj_data['obj_name_to_idx']
        self.vocab['object_idx_to_name'] = {
            v: k for k, v in self.vocab['object_name_to_idx'].items()}
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.max_samples = max_samples
        self.include_relationships = include_relationships
        self.image_paths = []
        transform = [T.Resize(image_size), T.ToTensor()]
        if normalize_images:
            # transform.append(imagenet_preprocess())
            transform.append(encode_image())
        self.transform = T.Compose(transform)

        if debug:
            self.clevr_data = pickle.load(open("clevr_data_sample.pkl", 'rb'))
            self.dialog_data = pickle.load(
                open("dialog_data_sample.pkl", 'rb'))
        else:
            self.clevr_data = json.load(
                open(os.path.join(base_path, 'scenes/CLEVR_{mode}_scenes.json'.format(mode=mode)), 'rb'))
            self.dialog_data = json.load(
                open(os.path.join(base_path, h5_path), 'rb'))

        if dense_scenes:
            self.keep_dense_scenes()

        if sort_ids:
            # Replace scenes
            self.keep_scenes_per_id(sort_ids, eval_func)

    def __len__(self):
        return len(self.dialog_data)

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - shapes: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j] means that (shapes[i], p, shapes[j]) is a triple.
        """

        # Get image
        entry = self.dialog_data[index]
        img_path = os.path.join(
            self.image_dir, entry['split'], entry['image_filename'])
        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                image = self.transform(image.convert('RGB'))

        sg = self.clevr_data['scenes'][index]
        vocab = self.vocab

        triplets = extract_triplets(
            sg, vocab, use_transitivity=self.use_transitivity)
        objs = extract_objs(sg, vocab)

        # Get boxes
        x, y, w, h = extract_bounding_boxes(sg)
        boxes = list(zip(x, y, w, h))
        boxes.append([0., 0., 1., 1.])
        boxes = torch.FloatTensor(boxes)

        new_objs = []

        for i in range(len(objs['shape']) - 1):
            shape = self.vocab['reverse_attributes']['shape'][objs['shape'][i].item()]
            color = self.vocab['reverse_attributes']['color'][objs['color'][i].item()]
            material = self.vocab['reverse_attributes']['material'][objs['material'][i].item(
            )]
            size = self.vocab['reverse_attributes']['size'][objs['size'][i].item()]
            new_obj_name = f'{color}_{material}_{size}_{shape}'
            new_obj = self.vocab['object_name_to_idx'][new_obj_name]
            new_objs.append(new_obj)

        new_objs.append(0)
        objs = torch.tensor(new_objs)
        return image, objs, boxes, triplets,  # sg

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def keep_dense_scenes(self):
        new_clv_data = []
        new_clv_dial_data = []
        for ind in range(len(self.clevr_data['scenes'])):
            num_obj = len(self.clevr_data['scenes'][ind]['objects'])
            if num_obj > self.max_objects:
                new_clv_data.append(self.clevr_data['scenes'][ind])
                new_clv_dial_data.append(self.dialog_data[ind])

        if len(new_clv_data) == 0 or len(new_clv_dial_data) == 0:
            print("No data has been selected in dense scenes mode")

        # Replace to dense scenes
        self.clevr_data['scenes'] = new_clv_data
        self.dialog_data = new_clv_dial_data

    def keep_scenes_per_id(self, ids, eval_func):
        new_clv_data = []
        new_clv_dial_data = []
        for ind in range(len(self.clevr_data['scenes'])):
            if ind not in ids:
                continue

            self.clevr_data['scenes'][ind] = eval_func(
                self.clevr_data['scenes'][ind], self.vocab)
            new_clv_data.append(self.clevr_data['scenes'][ind])
            new_clv_dial_data.append(self.dialog_data[ind])

        if len(new_clv_data) == 0 or len(new_clv_dial_data) == 0:
            print("No data has been selected in dense scenes mode")

        # Replace to dense scenes
        self.clevr_data['scenes'] = new_clv_data
        self.dialog_data = new_clv_dial_data


def extract_objs(sg, vocab):
    objs = {}
    for attr in vocab["attributes"].keys():
        attr_list = [vocab["attributes"][attr][obj[attr]]
                     for obj in sg['objects']]
        attr_list.append(vocab["attributes"][attr]['__image__'])
        objs[attr] = torch.LongTensor(attr_list)
    return objs


def clevr_collate_fn(batch):
    """
    Collate function to be used when wrapping a CLEVRDialogDataset in a
    DataLoader. Returns a tuple of the following:
    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
    - triples: FloatTensor of shape (T, 3) giving all triples, where
    triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triples to images;
    triple_to_img[t] = n means that triples[t] belongs to imgs[n].
    """
    # batch is a list, and each element is (image, objs, boxes, triplets)
    all_imgs, all_boxes, all_triplets = [], [], []
    all_objs = collections.defaultdict(list)
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0

    for i, (img, objs, boxes, triplets) in enumerate(batch):
        all_imgs.append(img[None])
        # to determine number objects, use an arbitrary choice - first attribute
        O, T = objs[list(objs.keys())[0]].size(0), triplets.size(0)

        for k, v in objs.items():
            all_objs[k].append(v)

        all_boxes.append(boxes)
        triplets = triplets.clone()
        triplets[:, 0] += obj_offset
        triplets[:, 2] += obj_offset
        all_triplets.append(triplets)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_imgs = torch.cat(all_imgs)

    for k, v in all_objs.items():
        all_objs[k] = torch.cat(v)

    all_boxes = torch.cat(all_boxes)
    all_triplets = torch.cat(all_triplets)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_imgs, all_objs, all_boxes, all_triplets,
           all_obj_to_img, all_triple_to_img)
    return out


def extract_triplets(sg, vocab, use_transitivity=0.):
    O = len(sg['objects']) + 1
    triplets = []

    for rel in sg['relationships'].keys():
        rel_triplets = []
        for o1 in range(O - 1):
            for o2 in sg['relationships'][rel][o1]:
                rel_triplets.append([o2, vocab['pred_name_to_idx'][rel], o1])
        # rel_triplets = reduce_transitive_edges(rel_triplets, p_keep=use_transitivity)
        triplets.extend(rel_triplets)

    # Add dummy __in_image__ relationships for all objects
    in_image = vocab['pred_name_to_idx']['__in_image__']
    for i in range(len(sg['objects'])):
        triplets.append([i, in_image, len(sg['objects'])])

    triplets = torch.LongTensor(triplets)
    return triplets


def extract_bounding_boxes(scene):
    """
    Get for each scene the bounding box
    :param scene:
    :return:
    """
    objs = scene['objects']
    rotation = scene['directions']['right']

    x_list = []
    y_list = []
    h_list = []
    w_list = []

    for i, obj in enumerate(objs):
        [x, y, z] = obj['pixel_coords']

        [x1, y1, z1] = obj['3d_coords']

        cos_theta, sin_theta, _ = rotation

        x1 = x1 * cos_theta + y1 * sin_theta
        y1 = x1 * -sin_theta + y1 * cos_theta

        height_d = 6.9 * z1 * (15 - y1) / 2.0
        height_u = height_d
        width_l = height_d
        width_r = height_d

        if obj['shape'] == 'cylinder':
            d = 9.4 + y1
            h = 6.4
            s = z1

            height_u *= (s * (h / d + 1)) / \
                ((s * (h / d + 1)) - (s * (h - s) / d))
            height_d = height_u * (h - s + d) / (h + s + d)

            width_l *= 11 / (10 + y1)
            width_r = width_l

        if obj['shape'] == 'cube':
            height_u *= 1.3 * 10 / (10 + y1)
            height_d = height_u
            width_l = height_u
            width_r = height_u

        y_min_coord = (y - height_d) / 320.
        y_max_coord = (y + height_u) / 320.
        x_man_coord = (x + width_r) / 480.
        x_min_coord = (x - width_l) / 480.

        x_list.append(x_min_coord)
        y_list.append(y_min_coord)
        h_list.append(y_max_coord - y_min_coord)
        w_list.append(x_man_coord - x_min_coord)

    return x_list, y_list, w_list, h_list


def pad_collate_fn(batch):
    '''
    boxes: (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
    '''
    # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
    imgs, objs, boxes, triples, obj_to_img, triple_to_img, image_ids = clevr_collate_fn(
        self.args.vocab, batch, padding=True)

    max_n_objs = objs.shape[1]

    triples[:, :, 0] += triple_to_img * max_n_objs
    triples[:, :, 2] += triple_to_img * max_n_objs
    objs = rearrange(objs, 'b s -> (b s)')
    boxes = rearrange(boxes, 'b s bc -> (b s) bc')
    triples = rearrange(triples, 'b s t -> (b s) t')
    obj_to_img = rearrange(obj_to_img, 'b s -> (b s)')
    triple_to_img = rearrange(triple_to_img, 'b s -> (b s)')

    return imgs, objs, boxes, triples, obj_to_img, triple_to_img


def dgl_clevr_collate_fn(batch, pos_enc_dim=8, vocab=None):

    all_imgs, all_graphs, all_boxes = [], [], []
    max_objs = 0

    for sample in batch:
        max_objs = max(max_objs, sample[1].shape[0])

    for i, (img, objs, boxes, triples) in enumerate(batch):
        all_imgs.append(img[None])
        if objs.dim() == 0 or triples.dim() == 0:
            continue
        O, T = objs.size(0), triples.size(0)

        pad_size = max_objs - O
        # If pad size is 0, no element is concatenated (zero-dim tensor)
        objs = torch.cat((objs, torch.LongTensor([0] * pad_size)))

        boxes = torch.cat((boxes, torch.FloatTensor(
            [[-0.5, -0.5, -0.1, -0.1]] * pad_size)))
        all_boxes.append(boxes)
        triples = triples.clone()
        graph = construct_dgl_graph(objs, triples, pos_enc_dim)
        all_graphs.append(graph)

    all_imgs = torch.cat(all_imgs)
    all_graphs = dgl.batch(all_graphs)
    all_boxes = torch.cat(all_boxes)

    out = (all_imgs, all_graphs, all_boxes)
    return out
