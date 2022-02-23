"""
@file ode_ssm_dynamics.py
@author Ryan Missel

PyTorch Lightning implementation of the ODE-VAE-GRU that contains an intervention dynamics function f(a) that is
corrected by a physics-informed loss encoder. This model propagates the pure dynamics forwards then corrects both
of them with individual GRU update cells.

Deterministic version in that the parameters W of the NODE are straight optimized.
"""
import os
import torch
import shutil
import argparse
import numpy as np
import torch.nn as nn
import pytorch_lightning
import torch.nn.functional as F

from torchdiffeq import odeint
from torch.utils.data import DataLoader
from util.layers import Flatten, UnFlatten
from data.data_loader import DynamicsDataset
from util.plotting import plot_recon_lightning
from util.utils import get_act, get_exp_versions


class DeterministicODEFunction(nn.Module):
    def __init__(self, args):
        """
        Represents a global NODE function whose weights are deterministic
        :param args: script arguments to use for initialization
        """
        super(DeterministicODEFunction, self).__init__()

        # Parameters
        self.args = args

        # Array that holds dimensions over hidden layers
        self.latent_dim = args.latent_dim
        self.layers_dim = [self.latent_dim] + args.num_layers * [args.num_hidden] + [self.latent_dim]

        # Build activation layers and layer normalization
        self.acts = []
        self.layer_norms = []
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act('leaky_relu') if i < args.num_layers else get_act('linear'))
            self.layer_norms.append(nn.LayerNorm(n_out, device=0) if True and i < args.num_layers else nn.Identity())

        # Build up initial distributions of weights and biases
        self.weights, self.biases = nn.ParameterList([]), nn.ParameterList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            # Weights
            self.weights.append(torch.nn.Parameter(torch.randn([n_in, n_out]), requires_grad=True))
            self.biases.append(torch.nn.Parameter(torch.randn([n_out]), requires_grad=True))

    def forward(self, t, x):
        """ Wrapper function for the odeint calculation """
        for norm, a, w, b in zip(self.layer_norms, self.acts, self.weights, self.biases):
            x = a(norm(F.linear(x, w.T, b)))
        return x


class InterventionODEFunction(nn.Module):
    def __init__(self, args):
        """
        Represents a global NODE function whose weights are deterministic
        :param args: script arguments to use for initialization
        """
        super(InterventionODEFunction, self).__init__()

        # Parameters
        self.args = args

        # Array that holds dimensions over hidden layers
        self.intervention_dim = args.intervention_dim
        self.layers_dim = [self.intervention_dim] + args.num_layers * [args.intervention_hidden] + [self.intervention_dim]

        # Build activation layers and layer normalization
        self.acts = []
        self.layer_norms = []
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act('leaky_relu') if i < args.num_layers else get_act('linear'))
            self.layer_norms.append(nn.LayerNorm(n_out, device=0) if True and i < args.num_layers else nn.Identity())

        # Build up initial distributions of weights and biases
        self.weights, self.biases = nn.ParameterList([]), nn.ParameterList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            # Weights
            self.weights.append(torch.nn.Parameter(torch.randn([n_in, n_out]), requires_grad=True))
            self.biases.append(torch.nn.Parameter(torch.randn([n_out]), requires_grad=True))

    def forward(self, t, x):
        """ Wrapper function for the odeint calculation """
        for norm, a, w, b in zip(self.layer_norms, self.acts, self.weights, self.biases):
            x = a(norm(F.linear(x, w.T, b)))
        return x


class CombinedODEFunction(nn.Module):
    def __init__(self, args, a_func):
        """
        Represents a global NODE function whose weights are deterministic
        :param args: script arguments to use for initialization
        """
        super(CombinedODEFunction, self).__init__()
        # Z Dynamics function
        self.z_func = DeterministicODEFunction(args)
        self.load_in_weights()

        # A dynamics function
        self.a_func = a_func

    def load_in_weights(self):
        """ Function to handle loading in the pre-trained normal dynamics """
        pass

    def forward(self, t, x):
        """ Wrapper function for the odeint calculation """
        x = self.z_func(x) + self.a_func(x)
        return x


class DGSSM(pytorch_lightning.LightningModule):
    def __init__(self, args, last_train_idx):
        super().__init__()
        self.save_hyperparameters(args)

        # Args
        self.args = args
        self.last_train_idx = last_train_idx

        # Losses
        self.bce = nn.BCELoss(reduction='none')

        # Dynamics functions
        self.ode_a_dynamics = InterventionODEFunction(args)
        self.combined_func = CombinedODEFunction(args, self.ode_a_dynamics)

        """ z0 encoder and update mechanisms for the z dynamics """
        self.z_encoder = nn.Sequential(
            nn.Conv2d(self.args.z_amort, 64, kernel_size=(3, 3), stride=2, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(),

            Flatten(),
            nn.Linear(128 * 6 * 5, self.args.latent_dim)
        )

        self.z_jump_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=2, padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),

            Flatten(),
            nn.Linear(64 * 6 * 5, self.args.latent_dim)
        )

        self.z_gru = nn.GRUCell(input_size=self.args.latent_dim, hidden_size=self.args.latent_dim)

        """ a0 encoder and update mechanisms for the a dynamics """
        self.a_encoder = nn.Sequential(
            nn.Conv2d(self.args.z_amort, 64, kernel_size=(3, 3), stride=2, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(),

            Flatten(),
            nn.Linear(128 * 6 * 5, self.args.latent_dim)
        )

        self.a_jump_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),

            nn.Linear(64, self.args.intervention_dim)
        )

        self.a_gru = nn.GRUCell(input_size=self.args.intervention_dim, hidden_size=self.args.intervention_dim)

        """ Two decoders, pre-trained one and the final optimized one """
        self.decoder = nn.Sequential(
            # First perform two linear scaling layers
            nn.Linear(self.args.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ELU(),

            # Then transform to image and tranpose convolve
            UnFlatten(8),
            nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=3),
            nn.BatchNorm2d(16),
            nn.ELU(),

            nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=2, padding=(2, 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),

            nn.ConvTranspose2d(8, 1, kernel_size=(4, 4), stride=2),
            nn.Sigmoid()
        )

        # Decoding network to get the reconstructed trajectory
        self.final_decoder = nn.Sequential(
            # First perform two linear scaling layers
            nn.Linear(self.args.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ELU(),

            # Then transform to image and tranpose convolve
            UnFlatten(8),
            nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=3),
            nn.BatchNorm2d(16),
            nn.ELU(),

            nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=2, padding=(2, 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),

            nn.ConvTranspose2d(8, 1, kernel_size=(4, 4), stride=2),
            nn.Sigmoid()
        )

    def intervention_loss(self, x_hat, y):
        """
        Handles getting the physics informed loss between the reconstrcuted x of normal dynamics through
        the equation L = MSE(Hx_hat, Yk)
        :param x_hat: reconstructed TMP
        """
        return self.mse(self.H * x_hat, y)

    def forward(self, x):
        """
        Forward function of the network that handles locally embedding the given sample into the C codes,
        generating the z posterior that defines mixture weightings, and finding the winning components for
        each sample
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        :return: reconstructions of the trajectory and generation
        """
        batch_size, generation_len = x.shape[0], x.shape[1]

        # Get q(z0 | X) and sample z0
        z0 = self.z_encoder(x[:, :self.args.z_amort])
        a0 = self.a_encoder(x[:, :self.args.z_amort])

        # Evaluate model forward over T to get L latent reconstructions
        timesteps = torch.linspace(0, generation_len - 1, generation_len - 1).to(self.device)

        # Evaluate the normal dynamics over the full time-period
        zt_normal = odeint(self.dynanmics_ode_func, z0, t, method='rk4', options={'step_size': 0.25})  # [T,q]
        zt_normal = zt_normal.permute([1, 0, 2])

        # Decode
        x_normal = self.decoder(zt_normal.contiguous().view([batch_size * generation_len, z0.shape[1]]))  # L*N*T,nc,d,d
        x_normal = x_normal.view([batch_size, generation_len, 100, 100])  # L,N,T,nc,d,d

        # Get physics-informed loss between normal dynamics reconstruction and BSP at all timesteps
        ls = self.intervention_loss(x_normal, y)

        # Evaluate forward over timestep
        last_a = a0
        last_z = z0

        zts = [z0.unsqueeze(0)]     # TODO - intervention application to z0? GRU update cell?
        for t in timesteps[1:]:
            """ Step 1: Propagate both functions forwards (combined function f(z) + f(a)) """
            # Propagate a and z forward the current dynamics prediction
            a_pred = odeint(self.ode_a_dynamics, last_a, t=torch.tensor([0, 1], dtype=torch.float),
                            method='rk4', options={'step_size': 0.25})[-1, :]

            z_pred = odeint(self.combined_func, last_z, t=torch.tensor([0, 1], dtype=torch.float),
                            method='rk4', options={'step_size': 0.25})[-1, :]

            """ Step 2: Update A via PI-Loss """
            # Decode

            # Get z encoding from data and update the z state
            z_enc = self.z_jump_encoder(x[:, t.cpu().numpy()].unsqueeze(1))
            z_corrected = self.jump_cell(z_enc, z_pred)

            # Get a encoding from PI-loss and update the a state
            a_diff = self.intervention_loss()
            a_enc = self.a_jump_encoder()
            z_corrected = self.jump_cell(z_enc, z_pred)

            # Add to trajectory, update last z
            last_z = z_corrected
            zts.append(z_corrected.unsqueeze(0))

        # Stack trajectory and decode
        zt = torch.vstack(zt).permute([1, 0, 2])
        Xrec = self.decoder(zt.contiguous().view([batch_size * generation_len, z0.shape[1]]))  # L*N*T,nc,d,d
        Xrec = Xrec.view([batch_size, generation_len, 100, 100])  # L,N,T,nc,d,d
        return Xrec

    def configure_optimizers(self):
        """ Define optimizers and schedulers used in training """
        optim = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[300, 600, 900], gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, verbose=True)
        return [optim], [scheduler]

    def training_step(self, batch, batch_idx):
        """ One training step for a given batch """
        _, bsp, tmp = batch
        bsp = bsp[:, :self.args.generation_len]

        # Get embedded C and z posterior with reconstructions
        preds = self(bsp)

        # Get loss and update weights
        bce_r, bce_g = self.bce(preds[:, :1], tmp[:, :1]).sum([2, 3]).view([-1]).mean(), \
                       self.bce(preds[:, 1:], tmp[:, 1:]).sum([2, 3]).view([-1]).mean()

        # Build the full loss
        loss = (self.args.r_beta * bce_r) + bce_g

        # Logging
        self.log("bce_r_loss", self.args.r_beta * bce_r, prog_bar=True)
        self.log("bce_g_loss", bce_g, prog_bar=True)

        if batch_idx >= self.last_train_idx:
            return {"loss": loss, "preds": preds.detach(), "tmps": tmp.detach()}
        else:
            return {"loss": loss}

    def training_epoch_end(self, outputs):
        """ Every 10 epochs, get reconstructions on batch of data """
        if self.current_epoch % 10 == 0:
            # Make image dir in lightning experiment folder if it doesn't exist
            if not os.path.exists('lightning_logs/version_{}/images/'.format(top)):
                os.mkdir('lightning_logs/version_{}/images/'.format(top))
                shutil.copy("stochastic_ode_ssm.py", "lightning_logs/version_{}/".format(top))

            # Using the last batch of this
            plot_recon_lightning(outputs[-1]["tmps"][:5], outputs[-1]["preds"][:5],
                                 self.args.dim, self.args.train_len,
                                 'lightning_logs/version_{}/images/recon{}train.png'.format(top, self.current_epoch))

            # Copy experiment to relevant folder
            if arg.exptype is not None:
                if os.path.exists("experiments/{}/{}/version_{}".format(self.args.model, self.args.exptype, exptop)):
                    shutil.rmtree("experiments/{}/{}/version_{}".format(self.args.model, self.args.exptype, exptop))
                shutil.copytree("lightning_logs/version_{}/".format(top),
                            "experiments/{}/{}/version_{}".format(self.args.model, self.args.exptype, exptop))

    def validation_step(self, batch, batch_idx):
        """ One validation step for a given batch """
        _, bsp, tmp = batch
        bsp = bsp[:, :self.args.generation_len]

        with torch.no_grad():
            # Get predicted trajectory from the model
            preds = self(bsp)

            # Get reconstruction loss
            bce_r, bce_g = self.bce(preds[:, :1], tmp[:, :1]).sum([2, 3]).view([-1]).mean(), \
                           self.bce(preds[:, 1:], tmp[:, 1:]).sum([2, 3]).view([-1]).mean()

            # Build the full loss
            loss = (self.args.r_beta * bce_r) + bce_g

        # Logging
        self.log("zval_bce_r_loss", self.args.r_beta * bce_r, prog_bar=True)
        self.log("zval_bce_g_loss", bce_g, prog_bar=True)
        return {"val_loss": loss, "val_preds": preds.detach(), "val_tmps": tmp.detach()}

    def validation_epoch_end(self, outputs):
        """ Every 10 epochs, get reconstructions on batch of data """
        if self.current_epoch % 10 == 0:
            # Make image dir in lightning experiment folder if it doesn't exist
            if not os.path.exists('lightning_logs/version_{}/images/'.format(top)):
                os.mkdir('lightning_logs/version_{}/images/'.format(top))

            # Using the last batch of this
            plot_recon_lightning(outputs[-1]["val_tmps"][:5], outputs[-1]["val_preds"][:5], self.args.dim,
                                 self.args.train_len, 'lightning_logs/version_{}/images/recon{}val.png'.format(top, self.current_epoch))

            # Save all val_reconstructions to npy file
            recons = None
            for tup in outputs:
                if recons is None:
                    recons = tup["val_preds"]
                else:
                    recons = torch.vstack((recons, tup["val_preds"]))

            np.save("lightning_logs/version_{}/recons.npy".format(top), recons.cpu().numpy())

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Model specific parameter group used for PytorchLightning integration """
        parser = parent_parser.add_argument_group("MoGSSM")
        parser.add_argument('--latent_dim', type=int, default=16, help='latent dimension of the z vector field')
        parser.add_argument('--intervention_dim', type=int, default=16, help='latent dimension of the a vector field')

        parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the ODE func')
        parser.add_argument('--num_hidden', type=int, default=100, help='number of nodes per hidden layer in ODE func')
        parser.add_argument('--intervention_hidden', type=int, default=50, help='number of nodes per hidden layer in ODE func')

        parser.add_argument('--num_filt', type=int, default=16, help='number of filters in the CNNs')

        parser.add_argument('--train_len', type=int, default=10,
                            help='how many X samples to use in model initialization')
        # parser.add_argument('--generation_len', type=int, default=13, help='total length to generate')

        parser.add_argument('--z_amort', type=int, default=3, help='how many X samples to use in z0 inference')
        return parent_parser


def parse_args():
    """ General arg parsing for non-model parameters """
    parser = argparse.ArgumentParser()

    # Experiment ID
    parser.add_argument('--exptype', type=str, default='vt_dynamics_random_bernoulli', help='name of the exp folder')
    parser.add_argument('--checkpt', type=str, default='None', help='checkpoint to resume training from')
    parser.add_argument('--model', type=str, default='jump', help='which model to choose')

    parser.add_argument('--random', type=bool, default=True, help='whether to have randomized sequence starts')

    # Learning hyperparameters
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs to run over')
    parser.add_argument('--batch_size', type=int, default=16, help='size of batch')
    parser.add_argument('--data_size', type=int, default=1008, help='size of batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')

    # Dimensions of different components
    parser.add_argument('--dim', type=int, default=100, help='dimension of the image data')

    # Tuning parameters
    parser.add_argument('--r_beta', type=float, default=10, help='multiplier for x0 bce term in loss')
    return parser


if __name__ == '__main__':
    # Parse and save cmd args
    parser = parse_args()
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser = DGSSM.add_model_specific_args(parser)
    arg = parser.parse_args()
    arg.gpus = [0]      # Choose the GPU ID to run on here
    arg.generation_len = 13 if arg.random else 32

    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125)

    # Get version numbers
    global top, exptop
    top, exptop = get_exp_versions(arg.model, arg.exptype)

    # Input generation
    traindata = DynamicsDataset(vt=True, split='train', random=arg.random)
    trainset = DataLoader(traindata, batch_size=arg.batch_size, shuffle=True, num_workers=2)
    last_train_idx = (traindata.bsps.shape[0] // arg.batch_size) - 1

    valdata = DynamicsDataset(vt=True, split='val', random=arg.random)
    valset = DataLoader(valdata, batch_size=arg.batch_size, shuffle=False, num_workers=2)

    # Init trainer
    trainer = pytorch_lightning.Trainer.from_argparse_args(arg, max_epochs=arg.num_epochs, auto_select_gpus=True)

    # Initialize model
    model = DGSSM(arg, last_train_idx)

    # Choose whether to restart from a given version checkpoint or have new training
    if arg.checkpt == 'None':
        trainer.fit(model, trainset, valset)
    else:
        trainer.fit(
            model, trainset, valset,
            ckpt_path="lightning_logs/version_{}/checkpoints/{}".format(
                arg.checkpt, os.listdir("lightning_logs/version_{}/checkpoints/".format(arg.checkpt))[0])
        )
