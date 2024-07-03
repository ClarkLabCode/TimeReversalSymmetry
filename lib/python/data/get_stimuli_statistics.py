#!/usr/bin/env python

'''
This script calculates stimuli statistics.
'''


import sys
sys.path.append('../helper/')

import os
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis

import helper_functions as hpfn

def get_statisics(train_samples):
    h1, bins1 = np.histogram(train_samples[:, -1, :, :].flatten(), bins=int(np.sqrt(len(train_samples[:, -1, :, :].flatten()))), density=True)
    mean1 = np.round(train_samples[:, -1, :, :].flatten().mean(), 2)
    std1 = np.round(train_samples[:, -1, :, :].flatten().std(), 2)
    skew1 = np.round(skew(train_samples[:, -1, :, :].flatten()), 2)
    kurtosis1 = np.round(kurtosis(train_samples[:, -1, :, :].flatten()), 2)

    return h1, bins1, mean1, std1, skew1, kurtosis1

save_folder = '../results/variables_for_paper/'

# ####### Normal
# training_data_path = '/mnt/d/data/panoramic/processed_data_natural/train_test_wide_field/'
# train_samples = np.load(training_data_path + f'train_samples.npy')
# train_targets = np.load(training_data_path + f'train_targets.npy')
        
# h1, bins1, mean1, std1, skew1, kurtosis1 = get_statisics(train_samples)
# save_path = save_folder + 'stimuli_statistics_natural'
# np.savez(save_path, h1, bins1, [mean1, std1, skew1, kurtosis1])


####### Phase randomized
training_data_path = '/mnt/d/data/panoramic/processed_data_natural_phase_randomized/train_test_wide_field/'
train_samples = np.load(training_data_path + f'train_samples.npy')
train_targets = np.load(training_data_path + f'train_targets.npy')
        
h1, bins1, mean1, std1, skew1, kurtosis1 = get_statisics(train_samples)
save_path = save_folder + 'stimuli_statistics_natural_phase_randomized'
np.savez(save_path, h1, bins1, [mean1, std1, skew1, kurtosis1])


# ####### Gaussian
# training_data_path = '/mnt/d/data/random_pattern/processed_data_gaussian/'
# train_samples = np.load(training_data_path + f'train_samples.npy')
# train_targets = np.load(training_data_path + f'train_targets.npy')
        
# h1, bins1, mean1, std1, skew1, kurtosis1 = get_statisics(train_samples)
# save_path = save_folder + 'stimuli_statistics_gaussian'
# np.savez(save_path, h1, bins1, [mean1, std1, skew1, kurtosis1])




        
        
        
        
        
        
        