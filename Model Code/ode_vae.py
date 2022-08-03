"""
@file ode_vae.py

PyTorch Lightning implementation of the ODE-VAE that has a global neural ODE that operates
on the VAE embedding in an encoder-decoder-like structure for time-series data
"""
import os
import torch
import shutil
import numpy as np
import torch.nn as nn
import pytorch_lightning
import torch.nn.functional as F

from util.utils import get_act
from torchdiffeq import odeint
from util.plotting import plot_recon_lightning
from util.layers import Flatten, UnFlatten, Gaussian
from torch.distributions import Normal, kl_divergence as kl


class DeterministicODEFunction(nn.Module):
    def __init__(self, args):
        """
        Represents a global NODE function whose weights are deterministic
        :param args: script arguments to use for initialization
        """
        super(DeterministicODEFunction, self).__init__()

        # Parameters
        self.args = args
        self.latent_dim = args.latent_dim
        self.layers_dim = [self.latent_dim] + args.num_layers * [args.num_hidden] + [self.latent_dim]

        # Build activation layers and layer normalization
        self.acts = []
        self.layer_norms = []
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act('leaky_relu') if i < args.num_layers else get_act('tanh'))
            self.layer_norms.append(nn.LayerNorm(n_out, device=0) if True and i < args.num_layers else nn.Identity())

        # Initialize weights and biases
        self.weights, self.biases = nn.ParameterList([]), nn.ParameterList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.weights.append(torch.nn.Parameter(torch.randn([n_in, n_out]), requires_grad=True))
            self.biases.append(torch.nn.Parameter(torch.randn([n_out]), requires_grad=True))

    def forward(self, t, x):
        """ ODE-Network to output vector field derivative """
        for norm, a, w, b in zip(self.layer_norms, self.acts, self.weights, self.biases):
            x = a(norm(F.linear(x, w.T, b)))
        return x


class ODEVAE(pytorch_lightning.LightningModule):
    def __init__(self, args, _, top, exptop, last_train_idx, __, ___, ____):
        super().__init__()

        # Args
        self.args = args
        self.top = top
        self.exptop = exptop
        self.last_train_idx = last_train_idx

        # ODE class holding weights and forward propagation
        self.ode_func = DeterministicODEFunction(args)

        # Losses
        self.lossf = nn.BCEWithLogitsLoss(reduction='none')

        # Z0 encoder to initialize the vector field
        self.z_encoder = nn.Sequential(
            nn.Conv2d(self.args.z_amort, 64, kernel_size=(3, 3), stride=2, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(),

            Flatten(),
            Gaussian(128 * 3 * 3, self.args.latent_dim)
        )

        # Holds generated z0 means and logvars for use in KL calculations
        self.z_means = None
        self.z_logvs = None

        # Decoding network to get the reconstructed trajectory
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
        )

        # Set of activations to use on the network output
        self.act = nn.Identity()
        self.out_act = nn.Sigmoid()

    def kl_z_term(self):
        """
        KL Z term, KL[q(z0|X) || N(0,1)]
        :return: mean klz across batch
        """
        batch_size = self.z_means.shape[0]
        mus, logvars = self.z_means.view([-1]), self.z_logvs.view([-1])  # N, 2

        q = Normal(mus, torch.exp(0.5 * logvars))
        N = Normal(torch.zeros(len(mus), device=mus.device),
                   torch.ones(len(mus), device=mus.device))

        klz = kl(q, N).view([batch_size, -1]).sum([1]).mean()
        return klz

    def forward(self, x):
        """
        Forward function of the network that handles locally embedding the given sample into the C codes,
        generating the z posterior that defines mixture weightings, and finding the winning components for
        each sample
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        :return: reconstructions of the trajectory and generation
        """
        batch_size, generation_len = x.shape[0], x.shape[1]

        # Get z0
        self.z_means, self.z_logvs, z0 = self.z_encoder(x[:, :self.args.z_amort])

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, generation_len - 1, generation_len).to(self.device)

        # Evaluate forward over timestep
        zt = odeint(self.ode_func, z0, t, method='rk4', options={'step_size': 0.25})  # [T,q]
        zt = zt.permute([1, 0, 2])

        # Decode
        Xrec = self.decoder(zt.contiguous().view([batch_size * generation_len, z0.shape[1]]))  # L*N*T,nc,d,d
        Xrec = Xrec.view([batch_size, generation_len, 100, 100])  # L,N,T,nc,d,d
        return Xrec

    def configure_optimizers(self):
        """ Define optimizers and schedulers used in training """
        optim = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100, 250, 400], gamma=0.5)
        return [optim], [scheduler]

    def training_step(self, batch, batch_idx):
        """ One training step for a given batch """
        _, bsp, tmp = batch
        bsp = bsp[:, :self.args.generation_len]

        # Get prediction and sigmoid-activated predictions
        preds = self.act(self(bsp))
        sig_preds = self.out_act(preds)

        # Get loss and update weights
        bce_r, bce_g = self.lossf(preds[:, :1], tmp[:, :1]).sum([2, 3]).view([-1]).mean(), \
                       self.lossf(preds[:, 1:], tmp[:, 1:]).sum([2, 3]).view([-1]).mean()

        # Get KL Z loss
        klz = self.kl_z_term()

        # Build the full loss
        loss = (self.args.r_beta * bce_r) + bce_g + (self.args.z_beta * klz)

        # Logging
        self.log("bce_r_loss", self.args.r_beta * bce_r, prog_bar=True)
        self.log("bce_g_loss", bce_g, prog_bar=True)
        self.log("klz_loss", (self.args.z_beta * klz), prog_bar=True)

        if batch_idx >= self.last_train_idx:
            return {"loss": loss, "preds": sig_preds.detach(), "tmps": tmp.detach()}
        else:
            return {"loss": loss}

    def training_epoch_end(self, outputs):
        """ Every 10 epochs, get reconstructions on batch of data """
        if self.current_epoch % 25 == 0:
            # Make image dir in lightning experiment folder if it doesn't exist
            if not os.path.exists('lightning_logs/version_{}/images/'.format(self.top)):
                os.mkdir('lightning_logs/version_{}/images/'.format(self.top))
                shutil.copy("ode_vae.py", "lightning_logs/version_{}/".format(self.top))

            # Using the last batch of this
            plot_recon_lightning(outputs[-1]["tmps"][:5], outputs[-1]["preds"][:5], self.args.dim, self.args.z_amort,
                                 'lightning_logs/version_{}/images/recon{}train.png'.format(self.top, self.current_epoch))

            # Copy experiment to relevant folder
            if self.args.exptype is not None:
                if os.path.exists("experiments/{}/{}/version_{}/".format(self.args.model, self.args.exptype, self.exptop)):
                    shutil.rmtree("experiments/{}/{}/version_{}/".format(self.args.model, self.args.exptype, self.exptop))
                shutil.copytree("lightning_logs/version_{}/".format(self.top),
                            "experiments/{}/{}/version_{}".format(self.args.model, self.args.exptype, self.exptop))

        if self.current_epoch % 100 == 0:
            torch.save(self.state_dict(), "lightning_logs/version_{}/checkpoints/save{}.ckpt".format(self.top, self.current_epoch))

    def validation_step(self, batch, batch_idx):
        """ One validation step for a given batch """
        _, bsp, tmp = batch
        bsp = bsp[:, :self.args.generation_len]

        with torch.no_grad():
            # Get predicted trajectory from the model
            preds = self.act(self(bsp))
            sig_preds = self.out_act(preds)

            # Get reconstruction loss
            bce_r, bce_g = self.lossf(preds[:, :1], tmp[:, :1]).sum([2, 3]).view([-1]).mean(), \
                           self.lossf(preds[:, 1:], tmp[:, 1:]).sum([2, 3]).view([-1]).mean()

            # Build the full loss
            loss = (self.args.r_beta * bce_r) + bce_g

            # Logging
            self.log("val_bce_r_loss", self.args.r_beta * bce_r, prog_bar=True)
            self.log("val_bce_g_loss", bce_g, prog_bar=True)

        return {"val_loss": loss, "val_preds": sig_preds.detach(), "val_tmps": tmp.detach()}

    def validation_epoch_end(self, outputs):
        """ Every 10 epochs, get reconstructions on batch of data """
        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists('lightning_logs/version_{}/images/'.format(self.top)):
            os.mkdir('lightning_logs/version_{}/images/'.format(self.top))
            shutil.copy("ode_vae.py", "lightning_logs/version_{}/".format(self.top))

        # Using the last batch of this
        ridx = np.random.randint(0, len(outputs), 1)[0]
        plot_recon_lightning(outputs[ridx]["val_tmps"][:5], outputs[ridx]["val_preds"][:5],
                             self.args.dim, self.args.z_amort,
                             'lightning_logs/version_{}/images/recon{}val.png'.format(self.top, self.current_epoch))

        # Save all val_reconstructions to npy file
        recons = None
        for tup in outputs:
            if recons is None:
                recons = tup["val_preds"]
            else:
                recons = torch.vstack((recons, tup["val_preds"]))

        np.save("lightning_logs/version_{}/recons.npy".format(self.top), recons.cpu().numpy())

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Model specific parameter group used for PytorchLightning integration """
        parser = parent_parser.add_argument_group("ODEVAE")
        parser.add_argument('--z_beta', type=float, default=1e-2, help='multiplier for z kl term in loss')
        return parent_parser
