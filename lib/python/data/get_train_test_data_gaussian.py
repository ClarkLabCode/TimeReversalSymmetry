#!/usr/bin/env python

'''
This script generates sythetic random data for motion detection.
'''


import sys
sys.path.append('../helper/')

import os
import numpy as np
import random
from scipy.stats import levy_stable
from ml_collections import config_flags
from absl import app
from absl import flags
from tqdm import tqdm

import helper_functions as hpfn

_FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'configuration file', lock_config=True)
flags.mark_flags_as_required(['config'])


def main(_):
    N_images = _FLAGS.config.N_images
    contrast_scale = _FLAGS.config.contrast_scale

    data_folder = _FLAGS.config.data_folder
    gamma1 = _FLAGS.config.gamma1 # half-life is 0.2 second
    delta_t = _FLAGS.config.delta_t # simulation step is 0.01 second
    acc_length = _FLAGS.config.acc_length # length of the acceleration array
    vel_length = _FLAGS.config.vel_length # length of the velocity array
    vscal = _FLAGS.config.velocity_scale # one standard deviation for the velocity, in degree/s
    acc_mean = _FLAGS.config.acc_mean # mean of the acceleration, in degree/s^2
    acc_std = _FLAGS.config.acc_std # standard deviation of the acceleration, in degree/s^2
    folder_processed = _FLAGS.config.training_data_path
    if not os.path.exists(folder_processed + ''):
        os.makedirs(folder_processed + '')

    vel_array_list = []
    for nv in range(N_images):
        acc_array = np.random.normal(acc_mean, acc_std, acc_length)
        vel_array = hpfn.get_filtered_OU_1d(gamma1, delta_t, acc_array, vel_length)
        vel_array_list.append(vel_array)
    vel_array_list = np.array(vel_array_list)
    print(f'The shape of the velocity array is {vel_array_list.shape}.')
    savepath = folder_processed + 'vel_array_list.npy'
    np.save(savepath, vel_array_list)


    ####### Generate the movies #######
    # Import velocity traces
    vel_array_list = np.load(folder_processed + 'vel_array_list.npy')

    K_row = 1
    K_col = 720
    pix_per_deg = K_col / 360
    FWHM = 5 # in degree
    sigma_for_gaussian = np.round(FWHM/(2*np.sqrt(2*np.log(2)))*pix_per_deg, 1) # smoothing gaussian

    sample_data = np.zeros((2*N_images, 50, 1, 72))
    sample_targets = np.zeros((2*N_images, 1))
    ind = 0
    for ii in tqdm(range(N_images)):
        img = np.random.normal(scale=contrast_scale, size=(K_row, K_col))
        img_filtered = hpfn.get_filtered_spacial_row(img, sigma_for_gaussian)
        _, shift_array_pix = hpfn.get_shift_array(vel_array_list[ii], delta_t, img_filtered)

        # Normal order
        img_processed = []
        for shift in shift_array_pix:
            img_roll = np.roll(img_filtered, shift, axis=1)
            img_resized = hpfn.get_resized(img_roll, n_row=K_row)
            img_processed.append(img_resized)
        img_processed = np.array(img_processed)
        sample_data[ind] = img_processed[:, :, :]
        sample_targets[ind] = vel_array_list[ii][-1:]
        ind = ind + 1
            
        # Reversed order to enforce symmetry
        sample_data[ind] = np.flip(img_processed[:, :, :], axis=-1)
        sample_targets[ind] = -vel_array_list[ii][-1:]
        ind = ind + 1

    sample_data = np.array(sample_data)
    sample_targets = np.array(sample_targets)
    print(f'The shape of the sample data is {sample_data.shape}.')
    print(f'The shape of the sample targets is {sample_targets.shape}.')
    np.save(folder_processed + 'sample_data.npy', sample_data)
    np.save(folder_processed + 'sample_targets.npy', sample_targets)

        
    ####### Divid data into training and testing set #######
    N_train = int(0.8 * N_images * 2)
    train_samples = sample_data[:N_train]
    train_targets = sample_targets[:N_train]
    test_samples = sample_data[N_train:]
    test_targets = sample_targets[N_train:]

    # balance the on and off signals
    train_samples = np.concatenate((train_samples, -train_samples))
    train_targets = np.concatenate((train_targets, train_targets))
    test_samples = np.concatenate((test_samples, -test_samples))
    test_targets = np.concatenate((test_targets, test_targets))


    np.save(folder_processed + 'train_samples', train_samples)
    np.save(folder_processed + 'train_targets', train_targets)

    np.save(folder_processed + 'test_samples', test_samples)
    np.save(folder_processed + 'test_targets', test_targets)

    print(f'The shape of the train data is {train_samples.shape}.')
    print(f'The shape of the train targets is {train_targets.shape}.')
    print(f'The shape of the test data is {test_samples.shape}.')
    print(f'The shape of the test targets is {test_targets.shape}.')
        

if __name__ == '__main__':
    app.run(main)       
        
        
        
        
        
        
        
        
        