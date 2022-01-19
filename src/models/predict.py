import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from src.module import LNNP
from src.data import data, qm9
from src import priors, models
from src.data.data import DataModule
from src.models import output_modules
from src.models.utils import rbf_class_mapping, act_class_mapping
from src.utils import LoadFromFile, LoadFromCheckpoint, save_argparse, number


def get_predict_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--load-model', action=LoadFromCheckpoint,
                        help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    parser.add_argument('--num-epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--inference-batch-size', default=None, type=int, help='Batchsize for validation and tests.')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr-patience', type=int, default=10,
                        help='Patience for lr-schedule. Patience per eval-interval of validation')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-factor', type=float, default=0.8, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-warmup-steps', type=int, default=0,
                        help='How many steps to warm-up over. Defaults to 0 for no warm-up')
    parser.add_argument('--early-stopping-patience', type=int, default=30,
                        help='Stop training after this many epochs without improvement')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay strength')
    parser.add_argument('--ema-alpha-y', type=float, default=1.0,
                        help='The amount of influence of new losses on the exponential moving average of y')
    parser.add_argument('--ema-alpha-dy', type=float, default=1.0,
                        help='The amount of influence of new losses on the exponential moving average of dy')
    parser.add_argument('--ngpus', type=int, default=-1,
                        help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--save-dir', '-l', default='models/', help='Model save directory')
    parser.add_argument('--splits', default=None, help='Npz with splits idx_train, idx_val, idx_test')
    parser.add_argument('--train-size', type=number, default=None,
                        help='Percentage/number of samples in training set (None to use all remaining samples)')
    parser.add_argument('--val-size', type=number, default=0.05,
                        help='Percentage/number of samples in validation set (None to use all remaining samples)')
    parser.add_argument('--test-size', type=number, default=0.1,
                        help='Percentage/number of samples in test set (None to use all remaining samples)')
    parser.add_argument('--test-interval', type=int, default=10,
                        help='Test interval, one test per n epochs (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save interval, one save per n epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')
    parser.add_argument('--redirect', type=bool, default=False, help='Redirect stdout and stderr to save_dir/log')

    # dataset specific
    parser.add_argument('--dataset', default=None, type=str, help='Name of the torch_geometric dataset')
    parser.add_argument('--dataset-root', default='~/data', type=str,
                        help='Data storage directory (not used if dataset is "CG")')
    parser.add_argument('--dataset-arg', default=None, type=str,
                        help='Additional dataset argument, e.g. target property for QM9 or molecule for MD17')
    parser.add_argument('--coord-files', default=None, type=str, help='Custom coordinate files glob')
    parser.add_argument('--embed-files', default=None, type=str, help='Custom embedding files glob')
    parser.add_argument('--energy-files', default=None, type=str, help='Custom energy files glob')
    parser.add_argument('--force-files', default=None, type=str, help='Custom force files glob')
    parser.add_argument('--energy-weight', default=1.0, type=float,
                        help='Weighting factor for energies in the loss function')
    parser.add_argument('--force-weight', default=1.0, type=float,
                        help='Weighting factor for forces in the loss function')

    # model architecture
    parser.add_argument('--model', type=str, default='graph-network', help='Which model to train')
    parser.add_argument('--output-model', type=str, default='Scalar', choices=output_modules.__all__,
                        help='The type of output model')
    parser.add_argument('--prior-model', type=str, default=None, choices=priors.__all__,
                        help='Which prior model to use')

    # architectural args
    parser.add_argument('--embedding-dimension', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of interaction layers in the model')
    parser.add_argument('--num-rbf', type=int, default=64, help='Number of radial basis functions in model')
    parser.add_argument('--activation', type=str, default='silu', choices=list(act_class_mapping.keys()),
                        help='Activation function')
    parser.add_argument('--rbf-type', type=str, default='expnorm', choices=list(rbf_class_mapping.keys()),
                        help='Type of distance expansion')
    parser.add_argument('--trainable-rbf', type=bool, default=False,
                        help='If distance expansion functions should be trainable')
    parser.add_argument('--neighbor-embedding', type=bool, default=False,
                        help='If a neighbor embedding should be applied before interactions')
    parser.add_argument('--aggr', type=str, default='add',
                        help='Aggregation operation for CFConv filter output. Must be one of \'add\', \'mean\', or \'max\'')

    # Transformer specific
    parser.add_argument('--distance-influence', type=str, default='both', choices=['keys', 'values', 'both', 'none'],
                        help='Where distance information is included inside the attention')
    parser.add_argument('--attn-activation', default='silu', choices=list(act_class_mapping.keys()),
                        help='Attention activation function')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')

    # other args
    parser.add_argument('--derivative', default=False, type=bool,
                        help='If true, take the derivative of the prediction w.r.t coordinates')
    parser.add_argument('--cutoff-lower', type=float, default=0.0, help='Lower cutoff in model')
    parser.add_argument('--cutoff-upper', type=float, default=5.0, help='Upper cutoff in model')
    parser.add_argument('--atom-filter', type=int, default=-1, help='Only sum over atoms with Z > atom_filter')
    parser.add_argument('--max-z', type=int, default=100,
                        help='Maximum atomic number that fits in the embedding matrix')
    parser.add_argument('--max-num-neighbors', type=int, default=32,
                        help='Maximum number of neighbors to consider in the network')
    parser.add_argument('--standardize', type=bool, default=False,
                        help='If true, multiply prediction by dataset std and add mean')
    parser.add_argument('--reduce-op', type=str, default='add', choices=['add', 'mean'],
                        help='Reduce operation to apply to atomic predictions')
    # fmt: on

    args = parser.parse_args()

    if args.redirect:
        sys.stdout = open(os.path.join(args.save_dir, "models"), "w")
        sys.stderr = sys.stdout
        logging.getLogger("pytorch_lightning").addHandler(
            logging.StreamHandler(sys.stdout)
        )

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size

    save_argparse(args, os.path.join(args.save_dir, "input.yaml"), exclude=["conf"])

    return args


def main():
    args = get_predict_args()
    # initialize data module
    data = DataModule(args)
    data.prepare_data()
    data.setup("predict")

    model = LNNP.load_from_checkpoint(checkpoint_path="models/model.ckpt")
    model.freeze()
    idx = data.test_dataset.indices
    x = data.test_dataset.dataset[idx.to(int)]

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        num_nodes=args.num_nodes,
        default_root_dir=args.save_dir,
        auto_lr_find=False,
        precision=args.precision,
    )
    y_hat = trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
# run it with python src/models/predict.py --conf config/predict_config.yaml