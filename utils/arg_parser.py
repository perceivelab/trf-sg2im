import argparse
import os


def get_args_parser():
    parser = argparse.ArgumentParser()

    # * Training Parameters
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--sgtrf_ckpt', default='',
                        help='Checkpoint for SGTransformer')
    parser.add_argument('--img_trf_ckpt', default='',
                        help='Checkpoint for Image Transformer')
    parser.add_argument(
        '--config', default='sgformer_coco.yaml', help='Path to config file')
    parser.add_argument('--box_beta', type=float,
                        help='Coefficient for scaling box loss')

    # * Optimizer
    parser.add_argument('--base_lr', default=2e-4, type=float)
    parser.add_argument('--max_lr', default=6e-4, type=float)
    parser.add_argument('--opt', default="adam",
                        choices=['adam', 'sgd'], type=str)
    parser.add_argument('--lr_scheduler', default="cosine",
                        choices=['none', 'plateau', 'exponential', 'cosine', 'onecycle'], type=str)

    # * Transformer
    parser.add_argument('--top_k', type=int,
                        help="Sample size for top-k sampling")
    parser.add_argument('--use_gt_layout', action="store_true",
                        help="Use GT Layout to condition the image transformer")

    # * Dataset parameters
    parser.add_argument('--data_dir', default="data/",
                        help="Directory where data is stored")
    parser.add_argument('--num_workers', default=os.cpu_count(), type=int)
    parser.add_argument('--persistent_workers', action='store_true')

    # * Logger parameters
    parser.add_argument('--run_name', default='placeholder',
                        help='descriptive name of run')
    parser.add_argument('--log_every_n_epochs', type=int, default=5,
                        help='Frequency to log images')
    parser.add_argument('--save_every_n_epochs', type=int, default=10,
                        help='Frequency to save checkpoints')

    # * Other parameters
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    return parser


def setup(args):
    if args.debug:
        args.run_name = 'debug'
        args.log_every_n_epochs = 1
        args.test_every_n_epochs = 1
