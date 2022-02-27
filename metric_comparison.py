"""
@file metric_comparison.py
@author

Handles loading in each of the reconstructions across each model and comparing metrics at each timestep
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage.metrics import structural_similarity


settype = "val"
print("SET TYPE: {}".format(settype))
gt = np.load('data/Pacing/pacing_tmps_{}.npy'.format(settype))
gt = gt.reshape([gt.shape[0], 28, -1])
gt_size = gt.shape[0]


""" Functions """
def mse(x, x_hat):
    return np.mean(np.mean((x - x_hat) ** 2, axis=2), axis=0)


def mae(x, x_hat):
    return np.mean(np.mean(np.abs(x - x_hat), axis=2), axis=0)


def ssim(x, x_hat):
    ssim_avg = 0

    for xi, xi_hat in zip(x, x_hat):
        ssim_const = structural_similarity(xi, xi_hat, data_range=x.max() - x.min())
        ssim_avg += ssim_const

    ssim_avg /= x.shape[0]
    return ssim_avg


def bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(np.mean(term_0+term_1, axis=2), axis=0)


def ccs(x, x_hat):
    ccs = []
    for gt_i, ov_i in zip(x, x_hat):
        sample_ccs = []

        for i in range(gt_i.shape[1]):
            sample_ccs.append(np.corrcoef(gt_i[:, i], ov_i[:, i], rowvar=False)[0, 1])

        ccs.append(sample_ccs)

    ccs = np.stack(ccs)
    ccs = np.mean(ccs, axis=0)      # 141, 10000 -> 10000
    ccs = np.mean(ccs, axis=0)      # 10000 -> 1
    return ccs


def ccs_spatial(x, x_hat):
    ccs = []
    for gt_i, ov_i in zip(x, x_hat):
        sample_ccs = []

        for i in range(gt_i.shape[0]):
            sample_ccs.append(np.corrcoef(gt_i[i, :], ov_i[i, :])[0, 1])

        ccs.append(sample_ccs)

    ccs = np.stack(ccs)
    print(ccs.shape)
    ccs = np.mean(ccs, axis=0)      # 141, 26 -> 26
    print(ccs.shape)
    ccs = np.mean(ccs, axis=0)      # 26 -> 1
    return ccs


""" ODE-VAE """
odevae_recons = np.load('vals/odevae_{}/recons.npy'.format(settype)).reshape([gt_size, 27, -1])
print(odevae_recons.shape)
# print("ODE-VAE Temporal CC: {}".format(ccs(gt[:, :27], odevae_recons)))
# print("ODE-VAE Spatial CC: {}".format(ccs_spatial(gt[:, :27], odevae_recons)))

odevae_mse = mse(gt[:, :27], odevae_recons)
odevae_mae = mae(gt[:, :27], odevae_recons)
odevae_bce = bce(gt[:, :27], odevae_recons)


""" ODE-VAE-GRU """
odevaegru_recons = np.load('vals/odevaegru_{}/recons.npy'.format(settype)).reshape([gt_size, 25, -1])
print(odevaegru_recons.shape)
# print("ODE-VAE-GRU Temporal CC: {}".format(ccs(gt[:, :25], odevaegru_recons)))
# print("ODE-VAE-GRU Spatial CC: {}".format(ccs_spatial(gt[:, :25], odevaegru_recons)))

odevaegru_mse = mse(gt[:, :25], odevaegru_recons)
odevaegru_mae = mae(gt[:, :25], odevaegru_recons)
odevaegru_bce = bce(gt[:, :25], odevaegru_recons)


""" ODE-VAE-IM """
odevaeim_recons = np.load('vals/odevaeim_{}/recons.npy'.format(settype)).reshape([gt_size, 26, -1])
print(odevaeim_recons.shape)
# print("ODE-VAE-IM Temporal CC: {}".format(ccs(gt[:, 1:27], odevaeim_recons)))
# print("ODE-VAE-IM Spatial CC: {}".format(ccs_spatial(gt[:, 1:27], odevaeim_recons)))

odevaeim_mse = mse(gt[:, :26], odevaeim_recons)
odevaeim_mae = mae(gt[:, :26], odevaeim_recons)
odevaeim_bce = bce(gt[:, :26], odevaeim_recons)


""" ODE-VAE-GRU-IM """
odevae_gru_im_recons = np.load('vals/odevae_gru_im_{}/recons.npy'.format(settype)).reshape([gt_size, 26, -1])
print(odevae_gru_im_recons.shape)
print("ODE-VAE-IM Temporal CC: {}".format(ccs(gt[:, 1:27], odevae_gru_im_recons)))
print("ODE-VAE-IM Spatial CC: {}".format(ccs_spatial(gt[:, 1:27], odevae_gru_im_recons)))

odevae_gru_im_mse = mse(gt[:, :26], odevae_gru_im_recons)
odevae_gru_im_mae = mae(gt[:, :26], odevae_gru_im_recons)
odevae_gru_im_bce = bce(gt[:, :26], odevae_gru_im_recons)


plt.plot(range(odevae_mse.shape[0]), odevae_mse)
plt.plot(range(odevaegru_mse.shape[0]), odevaegru_mse)
plt.plot(range(odevaeim_mse.shape[0]), odevaeim_mse)
plt.title("MSE over each Timestep")
plt.legend(['ODE-VAE', 'ODE-VAE-GRU', 'ODE-VAE-IM'])
plt.show()

plt.plot(range(odevae_mse.shape[0]), odevae_mae)
plt.plot(range(odevaegru_mae.shape[0]), odevaegru_mae)
plt.plot(range(odevaeim_mse.shape[0]), odevaeim_mae)
plt.title("MAE over each Timestep")
plt.legend(['ODE-VAE', 'ODE-VAE-GRU', 'ODE-VAE-IM'])
plt.show()

plt.plot(range(odevae_mse.shape[0]), odevae_bce)
plt.plot(range(odevaegru_mse.shape[0]), odevaegru_bce)
plt.plot(range(odevaeim_mse.shape[0]), odevaeim_bce)
plt.title("BCE over each Timestep")
plt.legend(['ODE-VAE', 'ODE-VAE-GRU', 'ODE-VAE-IM'])
plt.savefig("vals/BCEtimesteps.png")
plt.show()

# plt.plot(range(odevaegru_mse.shape[0]), odevaegru_ssim)
# plt.title("SSIM Error over each Timestep")
# plt.show()
