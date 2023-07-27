import torch
from torch import nn
from torch.optim.lr_scheduler import (CosineAnnealingLR, ExponentialLR,
                                      OneCycleLR, ReduceLROnPlateau)

from utils.logging import Logger


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)


class TrainerBase:

    def __init__(self, resume, scheduler=None, **kwargs):
        self.start_epoch = 0
        self.epochs = kwargs.pop('epochs', 300)
        self.vocab = kwargs.pop('vocab')
        self.log_every = kwargs.pop('log_every')
        self.save_every = kwargs.pop('save_every')
        self.train_loader = kwargs.pop('train_loader')
        self.val_loader = kwargs.pop('val_loader')
        self.device = kwargs.pop('device', 'cuda')
        self.logger = kwargs.pop('logger', None)
        self.init_checkpoint(resume)
        self.best_loss = float('inf')

    def increment_epoch(self):
        self.epoch += 1

    def get_trainable_parameters(self):
        return [p for n, p in self.model.named_parameters() if 'vqvae' not in n]

    def get_trainable_modules(self):
        return [(n, p) for n, p in self.model.named_modules() if 'vqvae' not in n]

    def scheduler_step(self, value=None):
        if not self.scheduler:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(value)
        else:
            self.scheduler.step()

    def get_scheduler(self, scheduler, opt, **kwargs):
        if not scheduler:
            self.scheduler = None
        elif scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                opt, patience=10, min_lr=1e-6)
        elif scheduler == 'exponential':
            self.scheduler = ExponentialLR(opt, gamma=0.9)
        elif scheduler == 'cosine':
            epochs = kwargs.pop('epochs', 300)
            self.scheduler = CosineAnnealingLR(
                opt, T_max=epochs, eta_min=1e-6)
        elif scheduler == 'onecycle':
            max_lr = kwargs.pop('max_lr', 1e-2)
            epochs = kwargs.pop('epochs', 300)
            steps_per_epoch = kwargs.pop('steps_per_epoch', 0)
            self.scheduler = OneCycleLR(
                opt, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch)

    def init_checkpoint(self, resume):
        if resume:
            self.load_checkpoint(resume)

        self.best_ckpt = {
            'epoch': 0,
            'loss': float('inf'),
        }

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.start_epoch = checkpoint['epoch']
        if 'sgtrf' in checkpoint:
            self.logger.info(f'Loading SGTransformer from {ckpt_path}')
            m, u = self.sgtrf.load_state_dict(
                checkpoint['sgtrf'], strict=False)
            self.logger.warning(f'Missing keys: {m}')
            self.logger.warning(f'Unexpected keys: {u}')
            self.sgtrf_opt.load_state_dict(checkpoint['sgtrf_opt'])
        if 'img_trf' in checkpoint:
            self.logger.info(f'Loading Image Transformer from {ckpt_path}')
            m, u = self.img_trf.load_state_dict(
                checkpoint['img_trf'], strict=False)
            self.logger.warning(f'Missing keys: {m}')
            self.logger.warning(f'Unexpected keys: {u}')
            self.img_trf_opt.load_state_dict(checkpoint['img_trf_opt'])

    def train_one_epoch(self, train_loader):
        pass

    def training_step(self, batch):
        pass

    def update_best_ckpt(self, epoch_loss, epoch):
        if epoch_loss < self.best_ckpt['loss']:
            # Save best model at min val loss
            self.best_ckpt['loss'] = epoch_loss
            self.best_ckpt['epoch'] = epoch
            self.best_ckpt['sgtrf'] = self.sgtrf.state_dict()
            self.best_ckpt['opt'] = self.opt.state_dict()
            changed = True
            return changed
