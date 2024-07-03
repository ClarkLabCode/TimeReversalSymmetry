#!/usr/bin/env python

'''
This script generates the data for motion detection.
This is not designed to run in parallel, and could take about half an hour to run.
'''


import sys
sys.path.append('../helper/')

import os
import numpy as np
import glob
import random
import time
from tqdm import tqdm
from ml_collections import config_flags
from absl import app
from absl import flags

import helper_functions as hpfn

_FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'configuration file', lock_config=True)
flags.mark_flags_as_required(['config'])


def main(_):
    start_time = time.time()

    data_folder = _FLAGS.config.data_folder
    folder_nat = data_folder + 'panoramic/data_natural_only_filtered/'
    items = glob.glob(folder_nat+'*.npy')
    random.shuffle(items)
    N_images = len(items)
    print('number of images are: ', N_images)

    img=np.load(items[0])
    K_row = img.shape[0] # in degree
    K_col = img.shape[1] # in degree


    ####### Generate a list of velocities #######
    gamma1 = _FLAGS.config.gamma1 # half-life is 0.2 second
    delta_t = _FLAGS.config.delta_t # simulation step is 0.01 second
    acc_length = _FLAGS.config.acc_length # length of the acceleration array
    vel_length = _FLAGS.config.vel_length # length of the velocity array
    scal = _FLAGS.config.velocity_scale # one standard deviation for the velocity, in degree/s
    acc_mean = _FLAGS.config.acc_mean # mean of the acceleration, in degree/s^2
    acc_std = _FLAGS.config.acc_std # standard deviation of the acceleration, in degree/s^2
    N_vel_per_img = _FLAGS.config.N_vel_per_img # different velocities for one image
    N_vel = N_vel_per_img * N_images # total different velocities # total different velocities
    scope = _FLAGS.config.scope # scope in space when calculating the contrast
    folder_processed = _FLAGS.config.folder_processed
    if not os.path.exists(folder_processed + ''):
        os.makedirs(folder_processed + '')

    vel_ind = -1

    vel_array_list = []
    for nv in range(N_vel):
        acc_array = np.random.normal(acc_mean, acc_std, acc_length)
        vel_array = hpfn.get_filtered_OU_1d(gamma1, delta_t, acc_array, vel_length)
        vel_array_list.append(vel_array)
    vel_array_list = np.array(vel_array_list)
    print(f'The shape of the velocity array is {vel_array_list.shape}.')
    savepath = folder_processed + 'vel_array_list.npy'
    np.save(savepath, vel_array_list)


    ####### Subsample the images and generate the movies #######
    # Import velocity traces
    vel_array_list = np.load(folder_processed + 'vel_array_list.npy')

    # Normal order
    if not os.path.exists(folder_processed + 'whole_frame/'):
        os.makedirs(folder_processed + 'whole_frame/')
    ind = 0
    for item in tqdm(items):
        img = np.load(item)
        for ii in range(N_vel_per_img):
            _, shift_array_pix = hpfn.get_shift_array(vel_array_list[ind], delta_t, img)
            img_processed = []
            for shift in shift_array_pix:
                img_roll = np.roll(img, shift, axis=1)
                img_resized = hpfn.get_resized(img_roll)
                img_processed.append(img_resized)
            img_processed = np.array(img_processed)
            np.save(folder_processed + 'whole_frame/img_processed_{}.npy'.format(ind+1), img_processed)
            ind = ind + 1
            
    # Reversed order to enforce symmetry
    if not os.path.exists(folder_processed + 'whole_frame_reversed/'):
        os.makedirs(folder_processed + 'whole_frame_reversed/')
    ind = 0
    for item in tqdm(items):
        img = np.load(item)
        for ii in range(N_vel_per_img):
            _, shift_array_pix = hpfn.get_shift_array(vel_array_list[ind], delta_t, img)
            img_processed = []
            for shift in shift_array_pix:
                img_roll = np.roll(np.flip(img, axis=1), -shift, axis=1)
                img_resized = hpfn.get_resized(img_roll)
                img_processed.append(img_resized)
            img_processed = np.array(img_processed)
            np.save(folder_processed + 'whole_frame_reversed/img_reversed_{}.npy'.format(ind+1), img_processed)
            ind = ind + 1
            

    ####### Pair the movies with the velocities #######
    # Import velocity traces
    vel_array_list = np.load(folder_processed + 'vel_array_list.npy')
    print(vel_array_list.shape)
    N = len(vel_array_list)
    print('Half sample size is ', N)

    # Normal samples
    samples1 = []
    for n in range(N):
        path = folder_processed + f'whole_frame/img_processed_{n+1}.npy'
        samples1.append([path, vel_array_list[n][vel_ind]])

    # Reversed samples
    samples2 = []
    for n in range(N):
        path = folder_processed + f'whole_frame_reversed/img_reversed_{n+1}.npy'
        samples2.append([path, -vel_array_list[n][vel_ind]])
        
        
    ####### Divid data into training and testing set #######
    N_train = int(180 * N_vel_per_img * 2)
    samples_training1 = samples1[:int(N_train/2)]
    samples_training2 = samples2[:int(N_train/2)]
    samples_testing1 = samples1[int(N_train/2):]
    samples_testing2 = samples2[int(N_train/2):]

    N_train = int(len(samples_training1)+len(samples_training2))
    N_test = int(len(samples_testing1)+len(samples_testing2))
    print('Training sample size is ', N_train)
    print('Testing sample size is ', N_test)
            
            
    ####### Data normalization, training #######
    if not os.path.exists(folder_processed + 'whole_frame_normalized_train/'):
        os.makedirs(folder_processed + 'whole_frame_normalized_train/')
    y_train_all = []
    for nt in tqdm(range(int(N_train/2))):
        # normal samples
        X_train = np.load(samples_training1[nt][0])
        X_train = hpfn.get_contrast(X_train, scope)
        y_train = samples_training1[nt][1]
        y_train_all.append(y_train)
        np.save(folder_processed + 'whole_frame_normalized_train/X_train_{}.npy'.format(int(nt*2)), X_train)
        # reversed samples
        X_train = np.load(samples_training2[nt][0])
        X_train = hpfn.get_contrast(X_train, scope)
        y_train = samples_training2[nt][1]
        y_train_all.append(y_train)
        np.save(folder_processed + 'whole_frame_normalized_train/X_train_{}.npy'.format(int(nt*2+1)), X_train)
    y_train_all = np.array(y_train_all)
    np.save(folder_processed + 'y_train_all.npy', y_train_all)


    ####### Data normalization, testing #######
    if not os.path.exists(folder_processed + 'whole_frame_normalized_test/'):
        os.makedirs(folder_processed + 'whole_frame_normalized_test/')
    y_test_all = []
    for nt in tqdm(range(int(N_test/2))):
        # normal samples
        X_test = np.load(samples_testing1[nt][0])
        X_test = hpfn.get_contrast(X_test, scope)
        y_test = samples_testing1[nt][1]
        y_test_all.append(y_test)
        np.save(folder_processed + 'whole_frame_normalized_test/X_test_{}.npy'.format(int(nt*2)), X_test)
        # reversed samples
        X_test = np.load(samples_testing2[nt][0])
        X_test = hpfn.get_contrast(X_test, scope)
        y_test = samples_testing2[nt][1]
        y_test_all.append(y_test)
        np.save(folder_processed + 'whole_frame_normalized_test/X_test_{}.npy'.format(int(nt*2+1)), X_test)
    y_test_all = np.array(y_test_all)
    np.save(folder_processed + 'y_test_all.npy', y_test_all)

    print(f'Total time took is {time.time()-start_time}.')
        
        
if __name__ == '__main__':
    app.run(main)
        
        
        
        
        
        
        
        