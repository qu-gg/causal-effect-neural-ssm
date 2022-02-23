"""
@file data_loader.py
@author Ryan Missel

Handles building the datasets for both an initial static starting position as well as a
model that samples random initial starting positions of the wave propagation

Requires Normal_BSP and Normal_TMP folders to be in the same directory, however can generate
independent files with newload=True
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.io import loadmat
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DynamicsDataset(Dataset):
    """
    Load in the BSP and TMP data from the raw .mat files
    Loads static starting positions of the sequence
    """
    def __init__(self, data_size=9999, vt=False, version="normal", split='train', newload=False, random=False):
        """
        :param data_size: how many samples to load in, default all
        :param split: which split (train/test) to load in for this dataset object
        :param newload: whether to generate the stacked files
        """
        self.random = random

        # Get prefix and the ending safe index for this VT dataset
        if version == "block":
            prefix = "Block"
            self.end_idx = 31
        elif version == "pacing":
            prefix = "Pacing"
            self.end_idx = -1
        elif version == "normal":
            prefix = "Normal"
            self.end_idx = 15
        else:
            raise NotImplementedError("Incorrect version {}".format(version))

        # On a new load, stack all the individual mat files, normalize them, and split to train/val/test
        if newload:
            # Process BSP
            bsp_idxs = []
            bsps = None
            for f in tqdm(os.listdir("{}/{}_BSP/".format(prefix, prefix))):
                bsp_idxs.append(f.split("_")[1])
                bsp = loadmat("{}/{}_BSP/{}".format(prefix, prefix, f))["bsp"]

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

            # Normalize the entire datasets
            bsps = (bsps - np.min(bsps)) / (np.max(bsps) - np.min(bsps))
            tmps = (tmps - np.min(tmps)) / (np.max(tmps) - np.min(tmps))

            # Split into train, val, test sets
            train_x, split_x, train_y, split_y = train_test_split(bsps, tmps, train_size=0.6, random_state=155,
                                                                  shuffle=True)
            val_x, test_x, val_y, test_y = train_test_split(split_x, split_y, train_size=0.5, random_state=155,
                                                            shuffle=True)

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

        # Transform into tensors and change to float type
        tmps = (tmps > 0.4).astype('float32')

        self.bsps = torch.from_numpy(bsps).to(device=torch.Tensor().device)[:data_size]
        self.tmps = torch.from_numpy(tmps).to(device=torch.Tensor().device)[:data_size]

        self.bsps = self.bsps.float()
        self.tmps = self.tmps.float()

    def __len__(self):
        return len(self.bsps) * 5

    def __getitem__(self, idx):
        # Get a random starting position in the sequence for this sample
        if self.random:
            idx = idx % len(self.bsps)

            # First get random starting indices for the sequences
            starting_idxs = np.random.randint(5, self.end_idx, 1)[0]

            # Then get a slice of 10 timesteps from the given start
            return torch.Tensor([idx]), \
                   self.bsps[idx, starting_idxs:starting_idxs + 13], \
                   self.tmps[idx, starting_idxs:starting_idxs + 13]

        # Get the full sequence from the starting position
        else:
            return torch.Tensor([idx]), self.bsps[idx], self.tmps[idx]


if __name__ == '__main__':
    dataset = DynamicsDataset(5000, vt=True, version="normal", split='train', newload=False)
    print(dataset.bsps.shape)
    print(dataset.tmps.shape)

    test = dataset.__getitem__(0)[1].cpu().numpy()
    test2 = dataset.__getitem__(5)[1].cpu().numpy()

    # for i in range(0, 28, 2):
    #     plt.imshow(dataset.__getitem__(0)[1][i])
    #     plt.title("BSP {}".format(0))
    #     plt.show()

    for i in range(0, 44, 2):
        plt.imshow(dataset.__getitem__(0)[2][i], cmap='gray')
        plt.title("BSP 0, step {}".format(i))
        plt.show()
        plt.pause(0.2)

    for i in range(0, 44, 2):
        plt.imshow(dataset.__getitem__(5)[2][i], cmap='gray')
        plt.title("BSP 5, step {}".format(i))
        plt.show()
        plt.pause(0.2)

