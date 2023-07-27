import json
from utils.visualize import *
import yaml
from tqdm import tqdm
from utils.sg2im.coco import build_coco_dsets, coco_collate_fn
import json
from utils.misc import fix_seed
from attrdict import AttrDict
from pathlib import Path
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils
import numpy as np
import PIL

def seg_to_mask(seg, width=1.0, height=1.0):
	"""
	Tiny utility for decoding segmentation masks using the pycocotools API.
	"""
	if type(seg) == list:
		rles = mask_utils.frPyObjects(seg, height, width)
		rle = mask_utils.merge(rles)
	elif type(seg['counts']) == list:
		rle = mask_utils.frPyObjects(seg, height, width)
	else:
		rle = seg
	return mask_utils.decode(rle)

def main():
    with open("args.yaml") as y:
        args = AttrDict(yaml.safe_load(y))
    args.data_dir = Path(args.data_dir)


    fix_seed(42)
    data_path = Path('data/coco')
    args.data_dir /= 'coco'
    collate_fn = coco_collate_fn
    config_file = 'utils/sg2im/coco'

    with open(f"{config_file}.json") as conf:
        args.data = AttrDict(json.load(conf))

    # with open(data_path / 'vocab.json') as v:
        # vocab = json.load(v)
        # node_vocab = vocab['object_name_to_idx']
        # rel_vocab = vocab['pred_name_to_idx']
        # rel2name = vocab['pred_idx_to_name']
        # node2name = vocab['object_idx_to_name']


    # Data Loading
    vocab, train_dset, val_dset = build_coco_dsets(args.data)
    dsets = {'train': train_dset, 'val': val_dset}
    # dsets = {'val': val_dset}

    mask_size = 16

    for split, dset in dsets.items():
        sg_dicts = {}
        for i, el in tqdm(enumerate(dset)):
            sg_dict = {}
            img_id = dset.image_ids[i]
            filename = dset.image_id_to_filename[img_id]
            # with open(f'{split}_images.txt', 'a') as o:
            #     o.write(filename + '\n')
            sg_dict['img_id'] = img_id
            sg_dict['filename'] = filename
            with open(data_path / 'images' / (split +'2017') / filename, 'rb') as f:
                with PIL.Image.open(f) as image:
                    WW, HH = image.size
            images, objs, boxes, masks, triples = el
            sg_dict['objs'] = objs.tolist()
            # subjects, predicates, objects = triples.chunk(3, dim=1)
            sg_dict['triples'] = []

            for t in triples:
                # s,p,o = t.tolist()
                # subject = vocab['object_idx_to_name'][objs[s]]
                # predicate = vocab['pred_idx_to_name'][p]
                # obj = vocab['object_idx_to_name'][objs[o]]
                sg_dict['triples'].append([t.tolist()])

            sg_dict['boxes'] = []

            for b in boxes:
                sg_dict['boxes'].append([b.tolist()])

            sg_dict['masks'] = []
            for object_data in dset.image_id_to_objects[img_id]:
                segmentation = object_data['segmentation']
                box = object_data['bbox']
                masks = []
                # for segmentation in segmentations:

                # This will give a numpy array of shape (HH, WW)
                mask = seg_to_mask(segmentation, WW, HH)
                x, y, w, h = box
                # Crop the mask according to the bounding box, being careful to
                # ensure that we don't crop a zero-area region
                mx0, mx1 = int(round(x)), int(round(x + w))
                my0, my1 = int(round(y)), int(round(y + h))
                mx1 = max(mx0 + 1, mx1)
                my1 = max(my0 + 1, my1)
                mask = mask[my0:my1, mx0:mx1]
                mask = imresize(255.0 * mask, (mask_size, mask_size),
                                                mode='constant')
                # mask = torch.from_numpy((mask > 128).astype(np.int64))
                mask = (mask > 128).astype(np.int64).tolist()
                    
                # masks.append(mask)

                sg_dict['masks'].append(mask) # Bbox coordinates are needed in the collate function

            sg_dicts[img_id] = sg_dict

        ann_json = json.dumps(sg_dicts)
        with open(f'{data_path / "annotations" /  split}.json', 'w') as out:
            json.dump(json.loads(ann_json), out)
            # images, objects, boxes, masks, triples, obj_to_img, triple_to_img, paths = coco_collate_fn([el], padding=False, return_paths=True)


if __name__ == '__main__':
    main()