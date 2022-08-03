"""
@file plotting.py

Holds general plotting functions for reconstructions of the bouncing ball dataset
"""
import numpy as np
from PIL import Image


def plot_recon_lightning(X, Xrec, img_size, genstart, out_loc):
    """
    Plotting function to handle plotting sequence pairs of ground truth and reconstruction on top of each other
    :param X: ground truth data
    :param Xrec: reconstructed predictions
    :param img_size: how big each image is in the sequence
    :param genstart: how many timesteps are used in parameter initialization
    :param out_loc: where to save the generated photo
    """
    X = X.cpu().numpy()
    Xrec = Xrec.cpu().numpy()

    [num_sample, time_steps, _, _] = X.shape
    blank = 0

    panel = np.ones((img_size * 2 * num_sample + blank * (num_sample + 2), img_size * time_steps + 2 * blank)) * 255
    panel = np.uint8(panel)
    panel = Image.fromarray(panel)

    selected_idx = np.random.choice(X.shape[0], num_sample, replace=False)
    selected_idx = sorted(selected_idx)

    for num, idx in enumerate(selected_idx):
        selected_inps = X[idx]
        selected_rcns = Xrec[idx, :genstart]
        selected_gens = Xrec[idx, genstart:]

        selected_inps = np.uint8(np.clip(selected_inps, 1e-3, 0.999) * 255)
        selected_rcns = np.uint8(np.clip(selected_rcns, 1e-3, 0.999) * 255)
        selected_gens = np.uint8(np.clip(selected_gens, 1e-3, 0.999) * 255)

        img = np.zeros((img_size * 2, genstart * img_size)).astype(np.uint8)
        for i in range(genstart):
            img[:img_size, i * img_size: (i + 1) * img_size] = selected_inps[i]
            img[img_size:img_size*2, i * img_size: (i + 1) * img_size] = selected_rcns[i]

        img = Image.fromarray(img)
        panel.paste(img, (blank, blank * (num + 1) + num * img_size * 2))

        img_gen = np.zeros((img_size * 2, (time_steps - genstart) * img_size)).astype(np.uint8)
        for i in range(time_steps - genstart):
            img_gen[:img_size, i * img_size: (i + 1) * img_size] = selected_inps[i + genstart]
            img_gen[img_size:img_size*2, i * img_size: (i + 1) * img_size] = selected_gens[i]

        img_gen = Image.fromarray(img_gen)
        panel.paste(img_gen, (blank * 2 + img_size * genstart, blank * (num + 1) + num * img_size * 2))

    panel.save(out_loc)
