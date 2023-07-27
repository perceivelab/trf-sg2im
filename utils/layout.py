import torch
import torch.nn.functional as F
from einops import rearrange

from utils.data import token_pair_from_bbox


def encode_layout(boxes, labels, no_sections=90, max_objects=30, pad_value=8191, dummy_category=0):
    ###Shuffle objects and boxes############
    device = boxes.device
    idxs = torch.linspace(
        0, labels.shape[1]-1, labels.shape[1], device=device).long()
    idxs = idxs[torch.randperm(idxs.shape[0], device=device)]
    boxes = boxes[:, idxs].to(device)
    labels = labels[:, idxs].to(device)
    ########################################
    tokenized_boxes = token_pair_from_bbox(boxes, no_sections=no_sections)
    cond = torch.stack([labels, *tokenized_boxes], 2)
    cond = cond.detach()
    pad = (0, 0, 0, max_objects - cond.shape[1], 0, 0)
    cond = F.pad(cond, pad, value=pad_value)
    cond = rearrange(cond, 'b s n -> b (s n)')
    extra = torch.tensor(token_pair_from_bbox(torch.tensor(
        [0, 0, 1, 1]), no_sections=no_sections)).to(cond.device).unsqueeze(0)
    extra = extra.repeat(cond.shape[0], 1)
    cond = torch.cat([cond, extra], dim=1)
    return cond
