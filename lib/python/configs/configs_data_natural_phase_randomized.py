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
  config.N_vel_per_img = configsdata.N_vel_per_img
  config.sample_per_image = configsdata.sample_per_image

  config.contrast_scale = 0.33
  config.scope = 'NA'
  config.folder_processed = config.data_folder + f'panoramic/processed_data_natural_phase_randomized/'
  config.training_data_path = config.data_folder + f'panoramic/processed_data_natural_phase_randomized/train_test_wide_field/'
  config.testing_data_path = config.data_folder + f'panoramic/processed_data_natural_phase_randomized/train_test_wide_field/'
  
  return config
