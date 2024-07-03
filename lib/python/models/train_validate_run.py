#!/usr/bin/env python


import sys
sys.path.append('../helper/')

import os
import numpy as np
from sklearn.model_selection import KFold
import time
import multiprocessing
from joblib import Parallel, delayed
from ml_collections import config_flags
from absl import app
from absl import flags

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import models
import train_validate as tnvl
import helper_functions as hpfn

_FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'configuration file', lock_config=True)
flags.DEFINE_integer('D_cnn', None, 'number of cnn layers', lower_bound=1)
flags.DEFINE_integer('C', None, 'number of channels', lower_bound=1)
flags.DEFINE_integer('R', None, 'repeat for different initializations', lower_bound=1)
flags.mark_flags_as_required(['config', 'D_cnn', 'C', 'R'])


def main(_):
    ######## Data #######
    # Run get_train_test_data.py first, and then load the data.
    train_samples = np.load(_FLAGS.config.training_data_path + f'train_samples.npy')
    train_targets = np.load(_FLAGS.config.training_data_path + f'train_targets.npy')
    if _FLAGS.config.contrast_stats == 'inverted':
        train_samples = -train_samples
    
    print(f'The shape of the training samples are {train_samples.shape}.')
    print(f'The shape of the training targets are {train_targets.shape}.')

    # Get an instance of the dataset class, and feed the loaded data.
    transformed_dataset = hpfn.RigidMotionDataset_ln(targets=train_targets, 
                                                     samples=train_samples,
                                                     transform=hpfn.ToTensor(),
                                                     T = _FLAGS.config.T)

    ####### train #######

    # folder that stores the results
    _FLAGS.config.save_folder = _FLAGS.config.save_folder + f'Dcnn{_FLAGS.D_cnn}_C{_FLAGS.C}_' + _FLAGS.config.activationf + f'_R{_FLAGS.R}/'

    # Make the folder that stores the results
    if not os.path.exists(_FLAGS.config.save_folder):
        os.makedirs(_FLAGS.config.save_folder)

    # model
    network_model = models.CNNSpaceInvLRSymm(D_cnn=_FLAGS.D_cnn, C=_FLAGS.C, od=_FLAGS.config.od, activationf=_FLAGS.config.activationf, T=_FLAGS.config.T)
    loss_function = _FLAGS.config.loss_function
    tnvl.train_run(_FLAGS.config, network_model, transformed_dataset, loss_function)


if __name__ == '__main__':
    app.run(main)
    
    








