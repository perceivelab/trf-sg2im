from datetime import datetime
from pathlib import Path

import logzero
import torch
import torchvision.transforms.functional as TF
import wandb
import yaml
from logzero import logger
from torchvision.transforms.functional import resize
from torchvision.utils import draw_bounding_boxes, save_image


class Logger:

    def __init__(self, run_name, vocab):

        self.run_name = Path(run_name)
        self.set_dirs()
        self.vocab = vocab
        self.obj_idx_to_name = {k: v for k,
                                v in enumerate(vocab['object_idx_to_name'])}
        logzero.logfile(self.log_dir / Path('output.log'))

    def set_dirs(self):
        current_date = datetime.now()
        formatted_date = f'{current_date.year}-{current_date.month}-{current_date.day}-{current_date.hour}h{current_date.minute}'
        test_name = f'{formatted_date}'

        self.log_dir = 'logs' / self.run_name / Path(test_name)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f'Saving logs into {self.log_dir}')

    def info(self, msg):
        logger.info(msg)

    def warning(self, msg):
        logger.warning(msg)

    def log_images(self, images, phase, title, step, size=512):
        wnb_images = []
        for i, im in enumerate(images):
            size = size if size is not None else im.shape[-1]
            im = TF.resize(im, size)
            im = (im - im.min()) / (im.max() - im.min())
            save_image(im, self.log_dir / Path(f'{step}_{i}_{title}.png'))
            im = TF.to_pil_image(im)
            wnb_images.append(wandb.Image(resize(im, [size, size])))

        wandb.log({f'{phase}/{title}': wnb_images, 'epoch': step})

    def log_images_w_boxes(self, images, boxes, labels, phase, title):
        assert len(images) == len(boxes) == len(
            labels), 'Images, boxes and labels must have the same length'
        wnb_images = []
        for im, im_boxes, im_labels in zip(images, boxes, labels):
            box_data = []
            class_names = []
            for i, (box, label) in enumerate(zip(im_boxes.squeeze(), im_labels)):
                x0, y0, x1, y1 = box.squeeze().tolist()
                class_name = self.vocab['object_idx_to_name'][label]
                if label == 0:
                    continue
                class_names.append(class_name)
                box_data.append({
                    "position": {
                        "minX": x0,
                        "maxX": x1,
                        "minY": y0,
                        "maxY": y1
                    },
                    "class_id": label.item() if isinstance(label, torch.Tensor) else label,
                    "box_caption": class_name,
                })
            im = resize(im, [512, 512])
            wnb_images.append(wandb.Image(im, boxes={f'predictions{i}':
                                                     {'box_data': box_data}}))
            im_boxes = im_boxes[im_labels != 0]
            im = (im - im.min()) / (im.max() - im.min())
            im *= 255
            im = draw_bounding_boxes(
                im.type(torch.uint8), im_boxes*512, class_names, width=3)
            TF.to_pil_image(im).save(
                self.log_dir / Path(f'{phase}_{title}.png'))

        wandb.log({f'{phase}/{title}': wnb_images})

    def log_hparams(self, hparams):
        with open(self.log_dir / Path('hparams.yaml'), 'w') as out:
            out.write(yaml.dump(hparams))

    def log_metric(self, title, metric, step):
        wandb.log({title: metric, 'epoch': step})

    def log_lr(self, lr, epoch):
        wandb.log({'train/lr': lr, "epoch": epoch})

    def save_ckpt(self, ckpt, path='checkpoint.pt', on_wandb=True):
        path = Path(path)
        logger.info(f'Saving {path} in {self.log_dir}')
        torch.save(ckpt, self.log_dir / path)
        if on_wandb:
            wandb.save(str(self.log_dir / path))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
