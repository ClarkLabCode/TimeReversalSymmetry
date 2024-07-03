#!/usr/bin/env python

######################
# Configurations for data generation.
######################
import sys
sys.path.append('../configs/')

from ml_collections import config_dict
import importlib

import configs_data_natural as confdn
importlib.reload(confdn)

configsdata = confdn.get_config()

def get_config():
  config = config_dict.ConfigDict()

  config.data_folder = configsdata.data_folder
  config.gamma1 = configsdata.gamma1
  config.delta_t = configsdata.delta_t
  config.acc_length = configsdata.acc_length
  config.vel_length = configsdata.vel_length
  config.velocity_scale = configsdata.velocity_scale
  config.acc_mean = configsdata.acc_mean
  config.acc_std = configsdata.acc_std
  config.L = configsdata.L
  config.T = configsdata.T

  config.contrast_scale = 1.25
  config.N_images = 50000
  config.training_data_path = config.data_folder + f'random_pattern/processed_data_gaussian/'
  config.testing_data_path = config.training_data_path
  
  return config
