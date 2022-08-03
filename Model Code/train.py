"""
@file train.py

Main training script used to run all versions (ODEVAE, ODEVAEGRU, ODEVAEIM).

Available models are:
- ode_vae
- ode_vae_gru
- ode_vae_im

Available datasets are:
- normal
- pacing

Experiments are saved in the structure 'experiments/model/exptype/version_num/' based on passed in arguments.
"""
import argparse
import pytorch_lightning

from scipy.io import loadmat
from torch.utils.data import DataLoader
from data.data_loader import DynamicsDataset
from util.utils import get_exp_versions

from ode_vae import ODEVAE
from ode_vae_gru import ODEVAEGRU
from ode_vae_im import ODEVAEIM


def parse_args():
    """ General arg parsing for non-model parameters """
    parser = argparse.ArgumentParser()

    # Experiment ID
    parser.add_argument('--exptype', type=str, default='train', help='name of the exp folder')
    parser.add_argument('--model', type=str, default='ode_vae_im', help='which model to choose')
    parser.add_argument('--version', type=str, default='pacing', help='which dataset version to use')

    # Learning hyperparameters
    parser.add_argument('--num_epochs', type=int, default=501, help='number of epochs to run over')
    parser.add_argument('--batch_size', type=int, default=16, help='size of batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')

    # Dimensions of different components
    parser.add_argument('--dim', type=int, default=100, help='dimension of the image data')

    # Tuning parameters
    parser.add_argument('--r_beta', type=float, default=10, help='multiplier for x0 bce term in loss')

    # Network dimensions
    parser.add_argument('--latent_dim', type=int, default=12, help='latent dimension of the z vector field')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the ODE func')
    parser.add_argument('--num_hidden', type=int, default=200, help='number of nodes per hidden layer in ODE func')

    parser.add_argument('--train_len', type=int, default=24, help='how many samples to use in reconstruction')
    parser.add_argument('--z_amort', type=int, default=3, help='how many X samples to use in z0 inference')
    return parser


def get_model(parser):
    """ Handles returning the model class related to the one specified in the arg parser """
    model = parser.parse_args().model

    if model == "ode_vae":
        model = ODEVAE
    elif model == "ode_vae_gru":
        model = ODEVAEGRU
    elif model == "ode_vae_im":
        model = ODEVAEIM
    else:
        raise NotImplementedError("Model class {} incorrect.".format(model))

    return model


if __name__ == '__main__':
    # Parse and save cmd args
    parser = parse_args()

    # Get the model class object for the given model
    model = get_model(parser)

    # Add model-specific arguments to the Trainer
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser = model.add_model_specific_args(parser)

    # Parse args and manually specify GPU ranks to train on
    arg = parser.parse_args()
    arg.generation_len = arg.train_len + arg.z_amort
    arg.gpus = [0]

    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125)

    # Get version numbers
    global top, exptop
    top, exptop = get_exp_versions(arg.model, arg.exptype)

    # Input generation
    traindata = DynamicsDataset(version=arg.version, length=arg.generation_len, split='train')
    trainset = DataLoader(traindata, batch_size=arg.batch_size, shuffle=True, num_workers=0)
    last_train_idx = (traindata.bsps.shape[0] // arg.batch_size) - 1

    valdata = DynamicsDataset(version=arg.version, length=arg.generation_len, split='val')
    valset = DataLoader(valdata, batch_size=arg.batch_size, shuffle=False, num_workers=0)

    # Load in the H matrix
    H_matrix = loadmat("data/H.mat")["H"]

    # Init trainer
    trainer = pytorch_lightning.Trainer.from_argparse_args(arg, max_epochs=arg.num_epochs,
                                                           check_val_every_n_epoch=25,
                                                           auto_select_gpus=True)

    # Initialize model
    model = model(arg, H_matrix, top, exptop, last_train_idx,
                  (traindata.bsps_max, traindata.bsps_min), (traindata.tmps_max, traindata.tmps_min))

    # Start model training
    trainer.fit(model, trainset, valset)
