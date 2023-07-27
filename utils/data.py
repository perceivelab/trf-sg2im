import dgl
import torchvision.transforms as T
import torchvision.transforms.functional as F

from modules.pos_enc import laplacian_positional_encoding


def token_pair_from_bbox(bbox, no_sections=90):
    bbox = bbox.clamp(0, 1)
    return tokenize_coordinates(bbox[..., 0], bbox[..., 1], no_sections=no_sections), \
        tokenize_coordinates(
            bbox[..., 2], bbox[..., 3], no_sections=no_sections)


def tokenize_coordinates(x: float, y: float, no_sections: float) -> int:
    """
    Express 2d coordinates with one number.
    Example: assume self.no_tokens = 16, then no_sections = 4:
    0  0  0  0
    0  0  #  0
    0  0  0  0
    0  0  0  x
    Then the # position corresponds to token 6, the x position to token 15.
    @param x: float in [0, 1]
    @param y: float in [0, 1]
    @return: discrete tokenized coordinate
    """
    x_discrete = (x * (no_sections - 1)).round().int()
    y_discrete = (y * (no_sections - 1)).round().int()
    return y_discrete * no_sections + x_discrete


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def idx_to_name(idxs, vocab):
    if isinstance(idxs, list):
        return [vocab['object_idx_to_name'][idx] for idx in idxs]
    elif isinstance(idxs, int):
        return vocab['object_idx_to_name'][idxs]


def normalize(x, low=0, up=1):
    return (up - low) * (x - x.min()) / (x.max() - x.min()) + low


def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def inv_normalize(img):
    if img.min() == 0 and img.max() == 1:  # Image already normalized
        pass
    elif img.min() == -1 and img.max() == 1:  # GAN Normalization
        img = normalize(img)
    else:  # ImageNet Normalization
        img = F.normalize(img, mean=[0, 0, 0], std=INV_IMAGENET_STD)
        img = F.normalize(img, mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0])
    return img


def construct_dgl_graph(objects, triples, pos_enc_dim=12, triplet_type=None):
    s, p, o = triples.chunk(3, dim=1)
    p = p.squeeze(1)
    s, o = s.squeeze(), o.squeeze()
    g = dgl.graph((s, o))
    pad_size = len(objects) - g.num_nodes()
    g.add_nodes(pad_size)
    g.ndata['feat'] = objects
    g.edata['feat'] = p
    if triplet_type is not None:
        g.edata['type'] = triplet_type
    laplacian_positional_encoding(g, pos_enc_dim)
    return g


def get_hw(img_size, downsampling_factors):
    for ds in downsampling_factors:
        img_size = img_size // ds
    return img_size
