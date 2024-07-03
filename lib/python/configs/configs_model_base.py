#!/usr/bin/env python

######################
# Configurations for models.
######################
import sys
sys.path.append('../models/')
sys.path.append('../configs/')

from ml_collections import config_dict
import torch.nn as nn
import importlib

import models
import configs_data_natural as confdn
importlib.reload(confdn)

configsdata = confdn.get_config()

def get_config():
  """
  With spacial invariance and left-right symmetry
  """
  config = config_dict.ConfigDict()

  config.T = configsdata.T
  
  config.D_cnn_list = [1, 2, 3, 4] # list of depth
  config.C_list = [2, 4] # list of number of independent channels
  config.k = 3 # the default longer dimension of the kernel (in multiples of 5 deg)
  config.od = 1 # output dimension
  config.activationf = 'ReLU' # activation functions
  config.num_epochs = 300
  config.batch_size = 100
  config.lr = 1e-3
  config.l1_factor = 0
  config.l2_factor = 0
  config.constraints_weight = models.weight_constraint_positive()
  config.constraints_bias = models.bias_constraint_negative()
  config.constrained_weight_layers = []
  config.constrained_bias_layers = []
  config.loss_function = nn.MSELoss()
  config.Repeat = 500 # repeat for different initializations
  
  return config
