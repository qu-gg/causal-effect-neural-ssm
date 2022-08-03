"""
@file utils.py

Utility functions across files
"""
import os
import torch.nn as nn


def get_act(act="relu"):
    """
    Return torch function of a given activation function
    :param act: activation function
    :return: torch object
    """
    if act == "relu":
        return nn.ReLU()
    elif act == "leaky_relu":
        return nn.LeakyReLU(0.1)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":
        return nn.Identity()
    elif act == 'softplus':
        return nn.modules.activation.Softplus()
    elif act == 'softmax':
        return nn.Softmax()
    else:
        return None


def get_exp_versions(model, exptype):
    """ Return the version number for the latest lightning log and experiment num """
    # Find version folder path
    top = 0

    if not os.path.exists("lightning_logs/"):
        os.mkdir("lightning_logs/")
        os.mkdir("lightning_logs/version_0/")

    for folder in os.listdir("lightning_logs/"):
        try:
            num = int(folder.split("_")[-1])
            top = num if num > top else top
        except ValueError:
            continue

    top += 1
    print("Version {}".format(top))

    # Set up paths if they don't exist
    if not os.path.exists("experiments/"):
        os.mkdir("experiments/")

    if not os.path.exists("experiments/{}".format(model)):
        os.mkdir("experiments/{}/".format(model))

    if not os.path.exists("experiments/{}/{}".format(model, exptype)):
        os.mkdir("experiments/{}/{}".format(model, exptype))

    # Find version folder path
    exptop = 0
    for folder in os.listdir("experiments/{}/{}/".format(model, exptype)):
        try:
            num = int(folder.split("_")[-1])
            exptop = num if num > exptop else exptop
        except ValueError:
            continue

    exptop += 1
    print("Exp Top {}".format(exptop))
    return top, exptop
