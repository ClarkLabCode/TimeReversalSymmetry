#!/usr/bin/env python

######################
# Configurations for data generation.
######################
from ml_collections import config_dict
import numpy as np


def get_config():
  config = config_dict.ConfigDict()

  ###### for Grace cluster
  config.data_folder = '/home/bz242/project/data/'
  ####### for office desktop
  config.data_folder = '/mnt/d/data/'

  ####### data generation
  config.gamma1 = np.log(2.) / 0.2 # half-life is 0.2 second
  config.delta_t = 0.01 # simulation step is 0.01 second
  config.acc_length = 200 # length of the acceleration array
  config.vel_length = 49 # length of the velocity array, does not matter as long as it is bigger than 49.
  config.velocity_scale = 100 # one standard deviation for the velocity, in degree/s
  config.acc_mean = 0 # mean of the acceleration, in degree/s^2
  config.acc_std = config.velocity_scale * np.sqrt(2 * config.gamma1 / (1 - np.exp(-2 * config.gamma1 * config.acc_length * config.delta_t)) / config.delta_t) # standard deviation of the acceleration, in degree/s^2
  config.scope = '30deg' # scope in space when calculating the contrast, either 'full', '15deg', '30deg', or '60deg'.
  config.L = 72 # width of the input array
  config.T = 50 # movie length in time (in multiples of 10 ms)
  config.N_vel_per_img = 100 # different velocities for one image
  config.sample_per_image = 4 # number of 1-by-72 samples from one movie
  config.folder_processed = config.data_folder + f'panoramic/processed_data_natural/'
  config.training_data_path = config.data_folder + f'panoramic/processed_data_natural/train_test_wide_field/'
  config.testing_data_path = config.data_folder + f'panoramic/processed_data_natural/train_test_wide_field/'
  
  return config
