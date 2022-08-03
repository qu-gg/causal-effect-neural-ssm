"""
@file ode_vae_im.py

PyTorch Lightning implementation of the ODE-VAE-IM that contains an intervention dynamics function f(a) that is
corrected by a physics-informed loss encoder.
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
            self.layer_norms.append(nn.LayerNorm(n_out).to(self.args.gpus[0]) if True and i < args.num_layers else nn.Identity())

        # Build up initial distributions of weights and biases
        self.weights, self.biases = nn.ParameterList([]), nn.ParameterList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
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
            self.layer_norms.append(nn.LayerNorm(n_out).to(self.args.gpus[0]) if True and i < args.num_layers else nn.Identity())

        # Build up initial distributions of weights and biases
        self.weights, self.biases = nn.ParameterList([]), nn.ParameterList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.weights.append(torch.nn.Parameter(torch.randn([n_in, n_out]), requires_grad=True))
            self.biases.append(torch.nn.Parameter(torch.randn([n_out]), requires_grad=True))

    def forward(self, t, x):
        """ Wrapper function for the odeint calculation """
        for norm, a, w, b in zip(self.layer_norms, self.acts, self.weights, self.biases):
            x = a(norm(F.linear(x, w.T, b)))
        return x


class CombinedODEFunction(nn.Module):
    def __init__(self, args, z_func):
        """
        Represents a global NODE function whose weights are deterministic
        :param args: script arguments to use for initialization
        """
        super(CombinedODEFunction, self).__init__()

        # Sizes of vector fields
        self.z_size = args.latent_dim
        self.a_size = args.intervention_dim
        self.c_size = args.latent_dim + args.intervention_dim

        # Z Dynamics function
        self.z_func = z_func

        # A dynamics function
        self.a_func = InterventionODEFunction(args)

        # TO-DO - test if linear combination layer on combined function makes a difference
        self.lin_combine = nn.Linear(self.z_size + self.a_size, self.z_size)

    def forward(self, t, input_concat):
        # Separate latent fields
        z, a = input_concat[:, :self.z_size], input_concat[:, self.z_size:self.c_size]

        # Perform the combined dynamics forward pass
        d_z = self.z_func(None, z)
        d_a = self.a_func(None, a)
        # d_z = torch.add(d_z, d_a)
        d_z = self.lin_combine(torch.concat((d_z, a), dim=1))

        # Concatenate to pass forward to
        return torch.concat([d_z, d_a], dim=1)


class ODEVAE(pytorch_lightning.LightningModule):
    def __init__(self, args):
        super().__init__()

        # Args
        self.args = args

        # ODE class holding weights and forward propagation
        self.ode_func = DeterministicODEFunction(args)

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


class ODEVAEIM(pytorch_lightning.LightningModule):
    def __init__(self, args, H, top, exptop, last_train_idx, bsp_range, tmp_range, new_train=True):
        super().__init__()

        # Args
        self.args = args
        self.top = top
        self.exptop = exptop
        self.last_train_idx = last_train_idx
        self.H = torch.from_numpy(H).requires_grad_(False).to(self.args.gpus[0]).float()

        # Sizes of vector fields
        self.z_size = args.latent_dim
        self.a_size = args.intervention_dim

        # Min/Max for BSP/TMP
        self.bsp_max, self.bsp_min = bsp_range
        self.tmp_max, self.tmp_min = tmp_range

        # Losses
        self.lossf = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')

        # Dynamics functions
        self.normal_net = ODEVAE(args)
        if new_train:
            self.normal_net.load_state_dict(
                torch.load("experiments/ode_vae/normal_trained/version_1/checkpoints/epoch350.ckpt",
                           map_location=self.device)["state_dict"]
            )

        self.normal_net.ode_func.requires_grad_(False)
        self.normal_net.z_encoder.requires_grad_(False)
        self.normal_net.decoder.requires_grad_(True)

        self.combined_func = CombinedODEFunction(args, self.normal_net.ode_func)

        """ Update mechanisms for the a dynamics """
        self.a_jump_encoder = nn.Sequential(
            nn.Conv2d(self.args.a_amort, 32, kernel_size=(3, 3), stride=2, padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),

            Flatten(),
            nn.Linear(64 * 3 * 3, self.args.latent_dim)
        )

        self.a_gru = nn.GRUCell(input_size=self.args.intervention_dim, hidden_size=self.args.intervention_dim)

        # Which set of activations to use on the network output
        self.act = nn.Identity()
        self.out_act = nn.Sigmoid()

    def intervention_loss(self, x_hat, y):
        """
        Handles getting the physics informed loss between the reconstructed x of normal dynamics through
        the equation L = MSE(Hx_hat, Yk)
        :param x_hat: reconstructed TMP
        """
        batch_size = x_hat.shape[0]

        # First denormalize inputs back to original data space
        y = ((self.bsp_max - self.bsp_min) * y) + self.bsp_min
        x_hat = ((self.tmp_max - self.tmp_min) * x_hat) + self.tmp_min

        # Get the residual
        return self.mse(x_hat.view([batch_size, -1]) @ self.H.T, y.view([batch_size, -1]))

    def forward(self, y):
        """
        Forward function of the network that handles locally embedding the given sample into the C codes,
        generating the z posterior that defines mixture weightings, and finding the winning components for
        each sample
        :param y: BSP data observation, which is a timeseries [BS, Timesteps, Dim1, Dim2]
        :return: reconstructions of the trajectory and generation
        """
        batch_size, generation_len, bsp_dim = y.shape[0], y.shape[1], y.shape[2]

        # Get z0 and a0
        _, _, z0 = self.normal_net.z_encoder(y[:, :self.args.z_amort])
        a0 = torch.zeros([batch_size, self.args.intervention_dim]).to(self.device)

        # Evaluate model forward over T to get L latent reconstructions
        timesteps = torch.linspace(1, generation_len - 1 - self.args.a_amort, generation_len - 1 - self.args.a_amort).to(self.device)

        # Evaluate forward over timestep
        a_pred, last_z = a0, z0
        zt = []
        for tidx in timesteps:
            t = int(tidx.cpu().numpy())

            """ Step 1: Get jumped intervention a """
            # Get normal dynamics prediction for k:k+a_amort
            z_pred = odeint(self.normal_net.ode_func, last_z, t=torch.tensor([0] + [i for i in range(1, self.args.a_amort + 1)], dtype=torch.float),
                            method='rk4', options={'step_size': 0.25})[1:, :]

            # Decode to get normal x prediction
            x_pred = self.normal_net.decoder(z_pred.contiguous().view([batch_size * self.args.a_amort, z0.shape[1]]))
            x_pred = self.out_act(x_pred)

            # Get intervention loss with H and a encoding
            intv_loss = self.intervention_loss(x_pred, y[:, t:t+self.args.a_amort]).view([batch_size, self.args.a_amort, bsp_dim, bsp_dim])
            a_enc = self.a_jump_encoder(intv_loss)

            # Predict a forwards and jump it with a_enc
            a = self.a_gru(a_enc, a_pred)

            """ Step 2: Propagate f(z) forwards with intervention """
            # Perform combined prediction forwads
            combined_pred = odeint(self.combined_func, torch.concat([last_z, a], dim=1),
                                   t=torch.tensor([0, 1], dtype=torch.float),
                                   method='rk4', options={'step_size': 0.25})[-1, :]

            # Split vector field into appropriate individual ones and update variables
            last_z = combined_pred[:, :self.z_size]
            a_pred = combined_pred[:, self.z_size:]

            # Append to variables
            zt.append(last_z.unsqueeze(0))

        # Stack trajectory and decode
        zt = torch.vstack(zt).permute([1, 0, 2])
        Xrec = self.normal_net.decoder(zt.contiguous().view([batch_size * (generation_len - 1 - self.args.a_amort), z0.shape[1]]))
        Xrec = Xrec.view([batch_size, generation_len - 1 - self.args.a_amort, 100, 100])
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
        tmp = tmp[:, 1:self.args.generation_len - self.args.a_amort]

        # Get prediction and sigmoid-activated predictions
        preds = self.act(self(bsp))
        sig_preds = self.out_act(preds)

        # Get loss and update weights
        loss = self.lossf(preds, tmp).sum([2, 3]).view([-1]).mean()

        # Logging
        self.log("bce_g_loss", loss, prog_bar=True)

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
                shutil.copy("ode_vae_im.py", "lightning_logs/version_{}/".format(self.top))

            # Using the last batch of this
            plot_recon_lightning(outputs[-1]["tmps"][:5], outputs[-1]["preds"][:5], self.args.dim, self.args.z_amort,
                                 'lightning_logs/version_{}/images/recon{}train.png'.format(self.top, self.current_epoch))

            # Copy experiment to relevant folder
            if self.args.exptype is not None:
                if os.path.exists("experiments/{}/{}/version_{}/".format(self.args.model, self.args.exptype, self.exptop)):
                    shutil.rmtree("experiments/{}/{}/version_{}/".format(self.args.model, self.args.exptype, self.exptop))
                shutil.copytree("lightning_logs/version_{}/".format(self.top),
                            "experiments/{}/{}/version_{}".format(self.args.model, self.args.exptype, self.exptop))

        if self.current_epoch % 200 == 0:
            torch.save(self.state_dict(), "lightning_logs/version_{}/checkpoints/save{}.ckpt".format(self.top, self.current_epoch))

    def validation_step(self, batch, batch_idx):
        """ One validation step for a given batch """
        with torch.no_grad():
            _, bsp, tmp = batch
            bsp = bsp[:, :self.args.generation_len]
            tmp = tmp[:, 1:self.args.generation_len - self.args.a_amort]

            # Get predicted trajectory from the model
            preds = self.act(self(bsp))
            sig_preds = self.out_act(preds)

            # Get loss and update weights
            loss = self.lossf(preds, tmp).sum([2, 3]).view([-1]).mean()

            # Logging
            self.log("val_bce_g_loss", loss, prog_bar=True)

        return {"val_loss": loss, "val_preds": sig_preds.detach(), "val_tmps": tmp.detach()}

    def validation_epoch_end(self, outputs):
        """ Every 10 epochs, get reconstructions on batch of data """
        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists('lightning_logs/version_{}/images/'.format(self.top)):
            os.mkdir('lightning_logs/version_{}/images/'.format(self.top))
            shutil.copy("ode_vae_im.py", "lightning_logs/version_{}/".format(self.top))

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
        parser = parent_parser.add_argument_group("ODEVAEIM")
        parser.add_argument('--intervention_dim', type=int, default=12, help='latent dimension of the a vector field')
        parser.add_argument('--intervention_hidden', type=int, default=200, help='number of nodes per hidden layer in ODE func')
        parser.add_argument('--a_amort', type=int, default=3, help='how many X+1 samples to use in a_enc inference')
        return parent_parser
