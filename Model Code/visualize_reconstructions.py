"""
@file visualize_reconstructions.py

Handles outputting visualizations of each model into a number of side-by-side predictions of
ground truth and the predictions. As the val/test sets are not shuffled, comparisons across the
sets are ordered.
"""
import os
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning

from scipy.io import loadmat
from train import parse_args, get_model
from torch.utils.data import DataLoader
from data.data_loader import DynamicsDataset
from util.plotting import plot_recon_lightning

# Parse and save cmd args
parser = parse_args()

# Get the model class object for the given model
model = get_model(parser)

# Add model-specific arguments to the Trainer
parser = pytorch_lightning.Trainer.add_argparse_args(parser)
parser = model.add_model_specific_args(parser)

# Parse args and manually specify GPU ranks to train on
args = parser.parse_args()
args.generation_len = args.train_len + args.z_amort
args.gpus = [0]

# Get experiment version
version = os.listdir("experiments/{}/{}/".format(args.model, args.exptype))[-1]

# Load in the H matrix
H_matrix = loadmat("data/H.mat")["H"]

# Set up dataloader for test split
valdata = DynamicsDataset(version="pacing", length=27, split='test')
valset = DataLoader(valdata, batch_size=6, shuffle=False, num_workers=0)

# Initialize model and load in its state dict
model = model(args, H_matrix, None, None, 0,
              (valdata.bsps_max, valdata.bsps_min), (valdata.tmps_max, valdata.tmps_min), False)

sd = torch.load("experiments/{}/{}/{}/checkpoints/{}".format(args.model, args.exptype, version,
    os.listdir("experiments/{}/{}/{}/checkpoints/".format(args.model, args.exptype, version))[-1]),
                map_location=torch.device("cpu"))["state_dict"]
model.load_state_dict(sd)
model = model.to(args.gpus[0])

# Make dirs if not already there
if not os.path.exists("vals/{}".format(args.model)):
    os.mkdir("vals/{}/".format(args.model))

# Losses
bce = nn.BCELoss(reduction='none')
recons = None
with torch.no_grad():
    for idx, batch in enumerate(valset):
        # Get loss and update weights
        if args.model == "ode_vae_im":
            _, bsp, tmp = batch
            bsp = bsp[:, :args.generation_len].cuda()
            tmp = tmp[:, 1:args.generation_len - args.a_amort]

            # Get prediction and sigmoid-activated predictions
            preds = model.act(model(bsp)).cpu()
            sig_preds = model.out_act(preds)

            # Get loss and update weights
            loss = model.lossf(preds, tmp).sum([2, 3]).view([-1]).mean()

        elif args.model == "ode_vae_gru":
            # Get batch components
            _, bsp, tmp = batch
            bsp = bsp[:, :args.generation_len].cuda()
            tmp = tmp[:, :-2]

            preds = model.out_act(model(bsp)).cpu()
            sig_preds = preds

            # Get reconstruction loss
            bce_r, bce_g = bce(preds[:, :1], tmp[:, :1]).sum([2, 3]).view([-1]).mean(), \
                           bce(preds[:, 1:], tmp[:, 1:]).sum([2, 3]).view([-1]).mean()
            loss = (10 * bce_r) + bce_g

        elif args.model == "ode_vae":
            # Get batch components
            _, bsp, tmp = batch
            bsp = bsp[:, :args.generation_len].cuda()
            tmp = tmp

            preds = model.out_act(model(bsp)).cpu()
            sig_preds = preds

            # Get reconstruction loss
            bce_r, bce_g = bce(preds[:, :1], tmp[:, :1]).sum([2, 3]).view([-1]).mean(), \
                           bce(preds[:, 1:], tmp[:, 1:]).sum([2, 3]).view([-1]).mean()
            loss = (10 * bce_r) + bce_g
        else:
            raise NotImplementedError("Model not implemented.")

        # Print out loss of batch
        print("Loss over {}: {}".format(idx, loss.item()))

        # Stack reconstructions
        if recons is None:
            recons = sig_preds
        else:
            recons = torch.vstack((recons, sig_preds))

        # Plot sequences of batch
        plot_recon_lightning(tmp[:5], sig_preds[:5], 100, 3, "vals/{}/local_val{}.png".format(args.model, idx))

np.save("vals/{}/recons.npy".format(args.model), recons.cpu().numpy())
