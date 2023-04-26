import json
import os
import random
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import PIL
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from utils.data import *
from utils.sg2im.utils import Resize

from data_modules.base import BaseDataModule
from data_modules.loader import *


class VGDataModule(BaseDataModule):
    def __init__(self, base_path="data", **kwargs):
        super(VGDataModule, self).__init__(base_path)

        self.pos_enc_dim = kwargs.pop('pos_enc_dim', 32)
        train_h5 = kwargs.pop('train_h5', 'train.h5')
        val_h5 = kwargs.pop('val_h5', 'val.h5')

        with open(os.path.join(self.data_dir, 'vocab.json')) as f:
            vocab = json.load(f)

        self.train_ds = VgSceneGraphDataset(
            vocab, train_h5, base_path, **kwargs)
        self.val_ds = VgSceneGraphDataset(vocab, val_h5, base_path, **kwargs)
        self.test_ds = self.val_ds
        self.collate_fn = partial(
            dgl_coco_collate_fn, pos_enc_dim=self.pos_enc_dim)

    @property
    def vocab(self):
        return self.train_ds.vocab

    @property
    def n_objs(self):
        return len(self.vocab['object_idx_to_name'])

    @property
    def n_rels(self):
        return len(self.vocab['pred_idx_to_name'])


class VgSceneGraphDataset(Dataset):
    def __init__(self, vocab, h5_path, image_dir, image_size=(256, 256),
                 normalize_images=True, max_objects=10, max_samples=None,
                 include_relationships=True, use_orphaned_objects=True):
        super(VgSceneGraphDataset, self).__init__()

        self.image_dir = image_dir
        self.image_size = image_size
        self.vocab = vocab
        self.num_objects = len(vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.max_samples = max_samples
        self.include_relationships = include_relationships

        transform = [Resize(image_size), T.ToTensor()]
        if normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)

        self.data = {}
        with h5py.File(Path(image_dir) / h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))

        self.image_paths = [p.decode('utf-8') for p in self.image_paths]

    def __len__(self):
        num = self.data['object_names'].size(0)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
            (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
            means that (objs[i], p, objs[j]) is a triple.
        """
        img_path = os.path.join(self.image_dir, self.image_paths[index])

        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        # H, W = self.image_size

        # Figure out which objects appear in relationships and which don't
        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(
            range(self.data['objects_per_image'][index].item()))
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        if len(obj_idxs) > self.max_objects - 1:
            obj_idxs = random.sample(obj_idxs, self.max_objects)
        if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
            num_to_add = self.max_objects - 1 - len(obj_idxs)
            num_to_add = min(num_to_add, len(obj_idxs_without_rels))
            obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
        O = len(obj_idxs) + 1

        objs = torch.LongTensor(O).fill_(-1)

        boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
        obj_idx_mapping = {}
        for i, obj_idx in enumerate(obj_idxs):
            objs[i] = self.data['object_names'][index, obj_idx].item()
            x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
            x0 = float(x) / WW
            y0 = float(y) / HH
            x1 = float(x + w) / WW
            y1 = float(y + h) / HH
            boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
            obj_idx_mapping[obj_idx] = i

        # The last object will be the special __image__ object
        objs[O - 1] = self.vocab['object_name_to_idx']['__image__']

        triples = []
        for r_idx in range(self.data['relationships_per_image'][index].item()):
            if not self.include_relationships:
                break
            s = self.data['relationship_subjects'][index, r_idx].item()
            p = self.data['relationship_predicates'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            s = obj_idx_mapping.get(s, None)
            o = obj_idx_mapping.get(o, None)
            if s is not None and o is not None:
                triples.append([s, p, o])

        # Add dummy __in_image__ relationships for all objects
        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([i, in_image, O - 1])

        triples = torch.LongTensor(triples)
        return image, objs, boxes, triples


def vg_collate_fn(batch):
    """
    Collate function to be used when wrapping a VgSceneGraphDataset in a
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
    # batch is a list, and each element is (image, objs, boxes, triples)
    all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (img, objs, boxes, triples) in enumerate(batch):
        all_imgs.append(img[None])
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_imgs, all_objs, all_boxes, all_triples,
           all_obj_to_img, all_triple_to_img)
    return out


def vg_uncollate_fn(batch):
    """
    Inverse operation to the above.
    """
    imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
    out = []
    obj_offset = 0
    for i in range(imgs.size(0)):
        cur_img = imgs[i]
        o_idxs = (obj_to_img == i).nonzero().view(-1)
        t_idxs = (triple_to_img == i).nonzero().view(-1)
        cur_objs = objs[o_idxs]
        cur_boxes = boxes[o_idxs]
        cur_triples = triples[t_idxs].clone()
        cur_triples[:, 0] -= obj_offset
        cur_triples[:, 2] -= obj_offset
        obj_offset += cur_objs.size(0)
        out.append((cur_img, cur_objs, cur_boxes, cur_triples))
    return out


def dgl_coco_collate_fn(batch, pos_enc_dim=12):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving object categories
    - boxes: FloatTensor of shape (O, 4)
    - masks: FloatTensor of shape (O, M, M)
    - triples: LongTensor of shape (T, 3) giving triples
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - triple_to_img: LongTensor of shape (T,) mapping triples to images
    """
    all_imgs, all_graphs, all_boxes = [], [], []
    obj_offset = 0
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
        boxes = torch.cat((boxes, torch.LongTensor([[0, 0, 1, 1]] * pad_size)))
        all_boxes.append(boxes)
        # If pad size is 0, no element is concatenated (zero-dim tensor)
        objs = torch.cat((objs, torch.LongTensor([0] * pad_size)))

        triples = triples.clone()
        graph = construct_dgl_graph(objs, triples, pos_enc_dim)
        all_graphs.append(graph)

    all_imgs = torch.cat(all_imgs)
    all_boxes = torch.cat(all_boxes)
    all_graphs = dgl.batch(all_graphs)

    out = (all_imgs, all_graphs, all_boxes)
    return out
