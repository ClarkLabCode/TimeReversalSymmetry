#!/usr/bin/env python

'''
This script prepares the data for training and testing. 
The whole training and testing dataset can be loaded into RAM at the same time.
'''


import sys
sys.path.append('../helper/')

import os
import numpy as np
from tqdm import tqdm
from ml_collections import config_flags
from absl import app
from absl import flags

_FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'configuration file', lock_config=True)
flags.mark_flags_as_required(['config'])


def main(_):

    n_row = 1
    n_col = _FLAGS.config.L
    T = _FLAGS.config.vel_length + 1
    scal = _FLAGS.config.velocity_scale # one standard deviation for the velocity, in degree/s
    sample_per_image = _FLAGS.config.sample_per_image # number of 1-by-72 samples from one movie

    data_folder = _FLAGS.config.data_folder
    folder_processed = _FLAGS.config.folder_processed
    if not os.path.exists(folder_processed + 'train_test_wide_field/'):
        os.makedirs(folder_processed + 'train_test_wide_field/')

    # Save config info to a log
    log_file = folder_processed + 'train_test_wide_field/log.txt'
    with open(log_file, 'w') as f:
        f.write('Data parameters:\n')
        f.write('----------------------------------------\n')
        f.write(f'Number of rows: {n_row}\n')
        f.write(f'Number of columns: {n_col}\n')
        f.write(f'Total length of the sample in time: {T}\n')
        f.write(f'scale of the velocity: {scal}\n')
        f.write(f'samples per image: {sample_per_image}\n')


    ####### Training data #######
    targets_file = folder_processed + 'y_train_all.npy'
    root_dir = folder_processed + 'whole_frame_normalized_train/'
    targets = np.load(targets_file)
    N = len(targets)

    train_samples = np.zeros((N*sample_per_image, T, n_row, n_col))
    train_targets = np.zeros((N*sample_per_image, 1))
    ind = 0
    for n in range(N):
        movie_name = os.path.join(root_dir, f'X_train_{n}.npy')
        movie = np.load(movie_name)
        target = targets[n]
        for ii in range(sample_per_image):
            row_start = np.random.randint(0, movie.shape[1]-n_row, 1)[0]
            col_start = 0
            sample = movie[:, row_start:row_start+n_row, col_start:col_start+n_col]
            train_samples[ind] = sample
            train_targets[ind] = target
            ind = ind + 1

    print(f'Array shape of the targets: {train_targets.shape}')
    print(f'Array shape of the samples: {train_samples.shape}')

    np.save(folder_processed + 'train_test_wide_field/train_samples', train_samples)
    np.save(folder_processed + 'train_test_wide_field/train_targets', train_targets)


    ####### Testing data #######
    targets_file = folder_processed + 'y_test_all.npy'
    root_dir = folder_processed + 'whole_frame_normalized_test/'
    targets = np.load(targets_file)
    N = len(targets)

    test_samples = np.zeros((N*sample_per_image, T, n_row, n_col))
    test_targets = np.zeros((N*sample_per_image, 1))
    ind = 0
    for n in tqdm(range(N)):
        movie_name = os.path.join(root_dir, f'X_test_{n}.npy')
        movie = np.load(movie_name)
        target = targets[n]
        for ii in range(sample_per_image):
            row_start = np.random.randint(0, movie.shape[1]-n_row, 1)[0]
            col_start = 0
            sample = movie[:, row_start:row_start+n_row, col_start:col_start+n_col]
            test_samples[ind] = sample
            test_targets[ind] = target
            ind = ind + 1

    print(f'Array shape of the targets: {test_targets.shape}')
    print(f'Array shape of the samples: {test_samples.shape}')

    np.save(folder_processed + 'train_test_wide_field/test_samples', test_samples)
    np.save(folder_processed + 'train_test_wide_field/test_targets', test_targets)


if __name__ == '__main__':
    app.run(main)


