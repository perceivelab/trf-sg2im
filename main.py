import wandb
import yaml
from attrdict import AttrDict
from omegaconf import OmegaConf

from utils.arg_parser import get_args_parser, setup
from utils.logging import Logger
from utils.misc import count_parameters, fix_seed, instantiate_from_config


def main(args):

    fix_seed(args.seed)

    config = OmegaConf.load(args.config)

    # Data Loading
    data_module = instantiate_from_config(config.datamodule)
    train_loader = data_module.train_dataloader(
        args.batch_size, args.device, args.num_workers, args.persistent_workers)
    val_loader = data_module.val_dataloader(
        args.batch_size, args.device, args.num_workers, args.persistent_workers)

    args.data.n_objs = data_module.n_objs
    args.data.n_rels = data_module.n_rels
    args.vocab = data_module.vocab

    logger = Logger(args.run_name, args.vocab)
    logger.log_hparams(args)

    tconf = OmegaConf.to_container(config.trainer)
    tparams = tconf['params']

    tparams['n_objs'] = data_module.n_objs
    tparams['n_rels'] = data_module.n_rels
    tparams['pos_enc_dim'] = data_module.pos_enc_dim
    tparams['lr_scheduler']['epochs'] = args.epochs
    tparams['lr_scheduler']['steps_per_epoch'] = len(train_loader)
    tparams['vocab'] = args.vocab
    tparams['resume'] = args.resume
    tparams['device'] = args.device
    tparams['logger'] = logger
    tparams['train_loader'] = train_loader
    tparams['val_loader'] = val_loader
    tparams['log_every'] = args.log_every_n_epochs
    tparams['save_every'] = args.save_every_n_epochs
    tparams['use_gt_layout'] = args.use_gt_layout

    trainer = instantiate_from_config(tconf)

    print('*'*70)
    logger.info(f'Debugging mode: {args.debug}')
    logger.info(f'Logging model information')
    logger.info(
        f'SGTransformer type: {trainer.sgtrf.__class__.__name__ if hasattr(trainer, "sgtrf") else None}')
    logger.info(
        f'SGTransformer total parameters {(count_parameters(trainer.sgtrf) / 1e6 if hasattr(trainer, "sgtrf") else 0):.2f}M')
    logger.info(
        f'VQVAE type: {trainer.vqvae.__class__.__name__ if hasattr(trainer, "vqvae") else None}')
    logger.info(
        f'VQVAE total parameters {(count_parameters(trainer.vqvae) / 1e6 if hasattr(trainer, "vqvae") else 0):.2f}M')
    logger.info(
        f'Image Transformer type: {trainer.img_trf.__class__.__name__ if hasattr(trainer, "img_trf") else None}')
    logger.info(
        f'Image Transformer total parameters {(count_parameters(trainer.img_trf) / 1e6 if hasattr(trainer, "img_trf") else 0):.2f}M')
    print('*'*70)

    trainer.train()


if __name__ == '__main__':

    args = get_args_parser().parse_args()

    with open(args.config, "r") as stream:
        conf = yaml.safe_load(stream)

    args.data = AttrDict(conf['datamodule']['params'])

    setup(args)

    run_mode = 'disabled' if args.debug else 'online'

    wandb.init(
        project='trf-sg2im',
        entity='rensortino',
        config=conf,
        name=args.run_name,
        mode=run_mode
    )

    main(args)
