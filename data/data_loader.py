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
    def __init__(self, data_size=9999, vt=False, split='train', newload=False, random=False):
        """
        :param data_size: how many samples to load in, default all
        :param split: which split (train/test) to load in for this dataset object
        :param newload: whether to generate the stacked files
        """
        self.random = random

        if newload:
            # Process BSP
            bsp_idxs = []
            bsps = None
            for f in tqdm(os.listdir("VT_BSP/")):
                bsp_idxs.append(f.split("_")[1])
                bsp = loadmat("VT_BSP/{}".format(f))["bsp"]

                if bsps is None:
                    bsps = np.expand_dims(bsp, axis=0)
                else:
                    bsps = np.vstack((bsps, np.expand_dims(bsp, axis=0)))

            # Process TMP
            tmp_idxs = []
            tmps = None
            for f in tqdm(os.listdir("VT_TMP/")):
                tmp_idxs.append(f.split("_")[1])
                tmp = loadmat("VT_TMP/{}".format(f))["tmp"]

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
            np.save("vt_bsps_train.npy", train_x, allow_pickle=True)
            np.save("vt_tmps_train.npy", train_y, allow_pickle=True)

            np.save("vt_bsps_val.npy", val_x, allow_pickle=True)
            np.save("vt_tmps_val.npy", val_y, allow_pickle=True)

            np.save("vt_bsps_test.npy", test_x, allow_pickle=True)
            np.save("vt_tmps_test.npy", test_y, allow_pickle=True)

        # Otherwise just load in the given split type
        elif vt is False:
            bsps = np.load("data/bsps_{}.npy".format(split), allow_pickle=True)
            tmps = np.load("data/tmps_{}.npy".format(split), allow_pickle=True)
        else:
            bsps = np.load("data/vt_bsps_{}.npy".format(split), allow_pickle=True)
            tmps = np.load("data/vt_tmps_{}.npy".format(split), allow_pickle=True)

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
            # starting_idxs = np.random.randint(5, 112, 1)[0]
            starting_idxs = np.random.randint(0, 19, 1)[0]

            # Then get a slice of 10 timesteps from the given start
            return torch.Tensor([idx]), \
                   self.bsps[idx, starting_idxs:starting_idxs + 13], \
                   self.tmps[idx, starting_idxs:starting_idxs + 13]

        # Get the full sequence from the starting position
        else:
            return torch.Tensor([idx]), self.bsps[idx], self.tmps[idx]


if __name__ == '__main__':
    dataset = DynamicsDataset(5000, vt=True, split='train', newload=False)
    print(dataset.bsps.shape)
    print(dataset.tmps.shape)

    for i in range(0, 120, 2):
        plt.imshow(dataset.__getitem__(0)[1][i])
        plt.title("BSP {}".format(0))
        plt.show()

    # for i in range(10):
    #     plt.imshow(dataset.__getitem__(i)[2][0])
    #     plt.title("TMP {}".format(i))
    #     plt.show()