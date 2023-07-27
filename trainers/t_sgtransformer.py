import dgl
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.ops import complete_box_iou_loss
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from trainers.t_base import TrainerBase
from utils.graph import get_labels_from_graph
from utils.misc import instantiate_from_config
from utils.visualize import draw_scene_graph


class TrainerSGTransformer(TrainerBase):

    def __init__(self, base_lr, resume=None, scheduler=None, **kwargs):

        sgtrf_config = OmegaConf.load(kwargs.pop('sgtransformer_config'))
        sgtrf_config.params.update(
            {'n_objs': kwargs['n_objs'], 'n_rels': kwargs['n_rels'], 'pos_enc_dim': kwargs['pos_enc_dim']})
        self.sgtrf = instantiate_from_config(sgtrf_config)
        self.opt = torch.optim.Adam(self.sgtrf.parameters(), lr=base_lr)
        super().__init__(resume, scheduler, **kwargs)
        self.box_beta = kwargs.pop('box_beta', 20)
        self.losses = ["box", "iou", 'label', "total"]

        sched_params = kwargs.pop('lr_scheduler', {})
        self.get_scheduler(scheduler, self.opt, **sched_params)

        self.sgtrf = self.sgtrf.to(self.device)

    def _make_ckpt(self, epoch):
        ckpt = {
            'epoch': epoch,
            'sgtrf': self.sgtrf.state_dict(),
            'img_trf': self.img_trf.state_dict() if hasattr(self, 'img_trf') else None,
            'opt': self.opt.state_dict(),
        }

        return ckpt

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.sgtrf.load_state_dict(checkpoint['sgtrf'])
        self.start_epoch = checkpoint['epoch']
        if "opt" in checkpoint:
            self.opt.load_state_dict(checkpoint['opt'])

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.logger.info(f'Training [{epoch} / {self.epochs}]')
            train_losses = self.train_one_epoch()

            {self.logger.log_metric(f'train/{k}_loss', v, epoch)
             for k, v in train_losses.items()}
            self.logger.info(f'Epoch {epoch}: {train_losses["total"]}')

            ckpt = self._make_ckpt(epoch)

            self.logger.info(f'Validation [{epoch} / {self.epochs}]')
            val_losses = self.evaluate(epoch)

            if (epoch + 1) % self.save_every == 0:
                if val_losses['total'] < self.best_loss:
                    self.best_loss = val_losses['total']
                    self.logger.save_ckpt(ckpt, 'best.pt')

                self.logger.save_ckpt(ckpt)

    def train_one_epoch(self):
        epoch_losses = {k: 0 for k in self.losses}
        self.sgtrf.train()
        for batch in tqdm(self.train_loader):
            losses = self.training_step(batch)
            epoch_losses = {k: v + losses[k] for k, v in epoch_losses.items()}
        epoch_losses = {k: v / len(self.train_loader)
                        for k, v in epoch_losses.items()}
        return epoch_losses

    def training_step(self, batch):

        images, graph, boxes = batch
        labels = get_labels_from_graph(graph)

        pred_boxes, pred_logits = self.run_model(graph)

        boxes = rearrange(boxes, '(b s) e -> b s e', b=images.shape[0])
        box_loss = F.mse_loss(pred_boxes, boxes)
        iou_loss = complete_box_iou_loss(pred_boxes, boxes, reduction='mean')
        label_loss = F.cross_entropy(pred_logits, labels)

        loss = box_loss + iou_loss + label_loss

        losses = {'box': box_loss.item(), 'label': label_loss.item(),
                  "iou": iou_loss.item(), "total": loss.item()}

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sgtrf.parameters(), max_norm=1.0)
        self.opt.step()
        self.scheduler_step()

        return losses

    def evaluate(self, epoch=0):
        # Run one epoch of evaluation
        losses_epoch, layout, images, graphs = self.eval_one_epoch()

        # Log evaluation metrics
        for key in losses_epoch.keys():
            self.logger.log_metric(f'eval/{key}', losses_epoch[key], epoch)

        # Log sample images
        if (epoch + 1) % self.log_every == 0:
            sgs = [to_tensor(draw_scene_graph(g, self.vocab).resize(
                [512, 512])) for g in dgl.unbatch(graphs)]
            self.logger.log_images(
                sgs, phase='eval', title='scene_graph', step=epoch)

            empty_img = torch.ones_like(images)
            self.logger.log_images_w_boxes(
                empty_img, layout['gt_boxes'], layout['gt_labels'], "eval", "gt")
            self.logger.log_images_w_boxes(
                empty_img, layout['pred_boxes'], layout['pred_labels'], "eval", "pred")
        return losses_epoch

    def eval_one_epoch(self):
        epoch_losses = {k: 0 for k in self.losses}
        self.sgtrf.eval()
        layout = {'gt': [], 'pred': []}
        for batch in tqdm(self.val_loader):
            images, graph, boxes = batch
            labels = get_labels_from_graph(graph)
            with torch.no_grad():
                pred_boxes, pred_logits = self.run_model(graph)
                pred_labels = torch.argmax(pred_logits, dim=1)

                boxes = rearrange(boxes, '(b s) e -> b s e', b=images.shape[0])
                box_loss = F.mse_loss(pred_boxes, boxes)
                iou_loss = complete_box_iou_loss(
                    pred_boxes, boxes, reduction='mean')
                label_loss = F.cross_entropy(pred_logits, labels)

                loss = box_loss + iou_loss + label_loss

                losses = {'box': box_loss.item(), 'label': label_loss.item(),
                          "iou": iou_loss.item(), "total": loss.item()}

                epoch_losses = {k: v + losses[k]
                                for k, v in epoch_losses.items()}
        layout['gt_boxes'] = boxes
        layout['pred_boxes'] = pred_boxes
        layout['gt_labels'] = labels
        layout['pred_labels'] = pred_labels
        epoch_losses = {k: v / len(self.val_loader)
                        for k, v in epoch_losses.items()}
        return epoch_losses, layout, images, graph

    def run_model(self, graph):

        pred_boxes, pred_logits = self.sgtrf(graph)
        pred_logits = rearrange(pred_logits, 'b s e -> b e s')
        pred_boxes = pred_boxes.clamp(min=0, max=1)
        return pred_boxes, pred_logits
