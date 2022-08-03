"""
@file data_loader.py

Handles building the datasets for both an initial static starting position and a
model that samples random initial starting positions of the wave propagation
"""
import os
import torch
import numpy as np

from tqdm import tqdm
from scipy.io import loadmat
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DynamicsDataset(Dataset):
    """
    Load in the BSP and TMP data from the raw .mat files
    Loads static starting positions of the sequence
    """
    def __init__(self, data_size=9999, version="normal", length=13, split='train', newload=False):
        """
        :param data_size: how many samples to load in, default all
        :param split: which split (train/test) to load in for this dataset object
        :param newload: whether to generate the stacked files
        """
        self.length = length

        # Get prefix and the ending safe index for this VT dataset
        if version == "pacing":
            prefix = "Pacing"
        elif version == "normal":
            prefix = "Normal"
        else:
            raise NotImplementedError("Incorrect data version {}.".format(version))

        # On a new load, stack all the individual mat files, normalize them, and split to train/val/test
        if newload:
            # Process BSP
            bsp_idxs = []
            bsp_timings = []

            bsps = None
            for f in tqdm(os.listdir("{}/{}_BSP/".format(prefix, prefix))):
                bsp_idxs.append(f.split("_")[1])
                bsp = loadmat("{}/{}_BSP/{}".format(prefix, prefix, f))["bsp"]

                # Get timing of intervention
                bsp_timings.append(
                    np.ceil(int(f.split("_")[-1][:-4]) / 50)
                )

                if bsps is None:
                    bsps = np.expand_dims(bsp, axis=0)
                else:
                    bsps = np.vstack((bsps, np.expand_dims(bsp, axis=0)))

            # Process TMP
            tmp_idxs = []
            tmps = None
            for f in tqdm(os.listdir("{}/{}_TMP/".format(prefix, prefix))):
                tmp_idxs.append(f.split("_")[1])
                tmp = loadmat("{}/{}_TMP/{}".format(prefix, prefix, f))["tmp"]

                if tmps is None:
                    tmps = np.expand_dims(tmp, axis=0)
                else:
                    tmps = np.vstack((tmps, np.expand_dims(tmp, axis=0)))

            # Save ranges of sets to denormalize
            np.save("{}/{}_bsps_range.npy".format(prefix, version), [np.max(bsps), np.min(bsps)], allow_pickle=True)
            np.save("{}/{}_tmps_range.npy".format(prefix, version), [np.max(tmps), np.min(tmps)], allow_pickle=True)

            # Normalize the entire datasets
            bsps = (bsps - np.min(bsps)) / (np.max(bsps) - np.min(bsps))
            tmps = (tmps - np.min(tmps)) / (np.max(tmps) - np.min(tmps))

            # Split into train, val, test sets
            train_x, split_x, train_y, split_y = train_test_split(bsps, tmps, train_size=0.6,
                                                                  random_state=155, shuffle=True)
            val_x, test_x, val_y, test_y = train_test_split(split_x, split_y, train_size=0.5,
                                                            random_state=155, shuffle=True)

            # Save on new load
            np.save("{}/{}_bsps_train.npy".format(prefix, version), train_x, allow_pickle=True)
            np.save("{}/{}_tmps_train.npy".format(prefix, version), train_y, allow_pickle=True)

            np.save("{}/{}_bsps_val.npy".format(prefix, version), val_x, allow_pickle=True)
            np.save("{}/{}_tmps_val.npy".format(prefix, version), val_y, allow_pickle=True)

            np.save("{}/{}_bsps_test.npy".format(prefix, version), test_x, allow_pickle=True)
            np.save("{}/{}_tmps_test.npy".format(prefix, version), test_y, allow_pickle=True)

        # Otherwise just load in the given split type
        else:
            bsps = np.load("data/{}/{}_bsps_{}.npy".format(prefix, version, split), allow_pickle=True)
            tmps = np.load("data/{}/{}_tmps_{}.npy".format(prefix, version, split), allow_pickle=True)

            if version == "pacing":
                bsps_range = np.load("data/{}/{}_bsps_range.npy".format(prefix, version), allow_pickle=True)
                self.bsps_max, self.bsps_min = bsps_range[0], bsps_range[1]

                tmps_range = np.load("data/{}/{}_tmps_range.npy".format(prefix, version), allow_pickle=True)
                self.tmps_max, self.tmps_min = tmps_range[0], tmps_range[1]

        # Transform into tensors and change to float type
        tmps = (tmps > 0.4).astype('float64')

        # Convert to Tensors and restrict to the given number of samples
        self.bsps = torch.from_numpy(bsps).to(device=torch.Tensor().device)[:data_size].float()
        self.tmps = torch.from_numpy(tmps).to(device=torch.Tensor().device)[:data_size].float()

    def __len__(self):
        return len(self.bsps)

    def __getitem__(self, idx):
        return torch.Tensor([idx]), self.bsps[idx, :self.length], self.tmps[idx, :self.length]
