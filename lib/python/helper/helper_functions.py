#!/usr/bin/env python

"""
This script contains helper functions for the data/model of the motion detection.
"""


import numpy as np
import glob
import matplotlib.image as mpimg
import cv2
from scipy.ndimage import gaussian_filter
from scipy import fft
import random
from scipy.stats import skew
from scipy.stats import kurtosis

import torch
from torch.utils.data import Dataset


####################
####### Data #######
####################

# Define the dataset class.
class RigidMotionDataset_ln(Dataset):
    """Rigid Motion dataset."""

    def __init__(self, targets, samples, transform=None, T=30):
        """
        Args:
            targets: numpy array
            samples: numpy array
            transform (callable, optional): Optional transform to be applied
                on a sample.
            T: the length in time (one for 10 ms)
        """
        self.targets = targets
        self.samples = samples
        self.transform = transform
        self.T = T

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        movie = self.samples[idx][-self.T:] # take the last T frames
        target = self.targets[idx]
        sample = {'movie': movie, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
# Transform functions
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        movie, target = sample['movie'], sample['target']

        return {'movie': torch.from_numpy(movie).float(),
                'target': torch.from_numpy(target).float()}


################################
####### helper functions #######
################################

def get_standardized_row(input_array):
    """
    Standardize an array.
    """
    mu = np.expand_dims(np.mean(input_array, axis=-1), axis=-1)
    std = np.expand_dims(np.std(input_array, axis=-1), axis=-1)

    assert std.all()
    output_array = (input_array - mu) / std
    
    return output_array


def get_normalized_row(input_array):
    """
    Normalize an array to be within [-1, 1]
    """
    min_v = np.expand_dims(np.min(input_array, axis=-1), axis=-1)
    max_v = np.expand_dims(np.max(input_array, axis=-1), axis=-1)

    assert (max_v - min_v).all()
    output_array = (input_array - min_v) / (max_v - min_v)
    output_array = (output_array - 0.5) * 2
    
    return output_array 


def get_standardized(input_array):
    """
    Standardize an array.
    """
    mu = np.mean(input_array)
    std = np.std(input_array)

    assert std > 0
    output_array = (input_array - mu) / std
    
    return output_array


def get_normalized(input_array):
    """
    Normalize an array to be within [-1, 1]
    """
    min_v = np.min(input_array)
    max_v = np.max(input_array)

    assert max_v > min_v
    output_array = (input_array - min_v) / (max_v - min_v)
    output_array = (output_array - 0.5) * 2
    
    return output_array   


def get_contrast(input_array, scope):
    """
    Get the contrast of an array.
    """
    if scope == 'full':
        mean_v = np.mean(input_array)
        output_array = (input_array - mean_v) / mean_v
    elif scope == 'row':
        mean_v = np.expand_dims(np.mean(input_array, axis=-1), axis=-1)
        output_array = (input_array - mean_v) / mean_v
    elif scope == '15deg':
        T, K_row, K_col = input_array.shape
        input_array_filtered = np.zeros((T, K_row, K_col))
        pix_per_deg = K_col / 360
        FWHM = 15 # in degree
        sigma_for_gaussian = np.round(FWHM/(2*np.sqrt(2*np.log(2)))*pix_per_deg, 1) # smoothing gaussian
        pad_size = int(4*sigma_for_gaussian) # this comes from the fact that the gaussian is truncated at 4*std
        for t in range(T):
            input_array_filtered[t] = get_filtered_spacial(input_array[t], pad_size, sigma_for_gaussian)
        output_array = (input_array - input_array_filtered) / input_array_filtered
    elif scope == '30deg':
        T, K_row, K_col = input_array.shape
        input_array_filtered = np.zeros((T, K_row, K_col))
        pix_per_deg = K_col / 360
        FWHM = 30 # in degree
        sigma_for_gaussian = np.round(FWHM/(2*np.sqrt(2*np.log(2)))*pix_per_deg, 1) # smoothing gaussian
        pad_size = int(4*sigma_for_gaussian) # this comes from the fact that the gaussian is truncated at 4*std
        for t in range(T):
            input_array_filtered[t] = get_filtered_spacial(input_array[t], pad_size, sigma_for_gaussian)
        output_array = (input_array - input_array_filtered) / input_array_filtered
    elif scope == '60deg':
        T, K_row, K_col = input_array.shape
        input_array_filtered = np.zeros((T, K_row, K_col))
        pix_per_deg = K_col / 360
        FWHM = 60 # in degree
        sigma_for_gaussian = np.round(FWHM/(2*np.sqrt(2*np.log(2)))*pix_per_deg, 1) # smoothing gaussian
        pad_size = int(4*sigma_for_gaussian) # this comes from the fact that the gaussian is truncated at 4*std
        for t in range(T):
            input_array_filtered[t] = get_filtered_spacial(input_array[t], pad_size, sigma_for_gaussian)
        output_array = (input_array - input_array_filtered) / input_array_filtered
        
    return output_array   


def get_filtered_spacial(input_frame, pad_size, sigma_for_gaussian):
    """
    Filter the input_frame with a Gaussian filter spacially.
    """
    K_row = input_frame.shape[0]
    K_col = input_frame.shape[1]
    padded_frame = np.zeros((K_row+2*pad_size, K_col+2*pad_size))
    padded_frame[pad_size:-pad_size, pad_size:-pad_size] = input_frame
    padded_frame[pad_size:-pad_size, :pad_size] = input_frame[:, -pad_size:]
    padded_frame[pad_size:-pad_size, -pad_size:] = input_frame[:, :pad_size]
    filtered_frame = gaussian_filter(padded_frame, sigma_for_gaussian)
    output_frame = filtered_frame[pad_size:-pad_size, pad_size:-pad_size]
    
    return output_frame


def get_filtered_spacial_row(input_frame, sigma_for_gaussian, filter_mode='wrap'):
    """
    Filter the input_frame with a Gaussian filter spacially.
    """
    K_row = input_frame.shape[0]
    K_col = input_frame.shape[1]
    output_frame = np.zeros((K_row, K_col))
    for kk in range(K_row):
        output_frame[kk, :] = gaussian_filter(input_frame[kk, :], sigma_for_gaussian, mode=filter_mode)
    
    return output_frame


def get_resized(input_array, n_row=20, n_col=72):
    """
    Resize the input_array.
    """
    step_row = input_array.shape[0] / n_row
    step_col = input_array.shape[1] / n_col

    row_sample = (np.arange(n_row) * step_row + step_row / 2).astype(int)
    col_sample = (np.arange(n_col) * step_col + step_col / 2).astype(int)
    resized_array = input_array[row_sample, :]
    resized_array = resized_array[:, col_sample]

    return resized_array


def get_resized_cv2(input_array, new_size=(72, 20)):
    """
    Resize the input_array using cv2.
    """
    resized_array = cv2.resize(input_array, dsize=new_size, interpolation=cv2.INTER_AREA)

    return resized_array


def get_filtered_OU_1d(gamma1, delta_t, input_array, vel_length, initial_val=0):
    """
    This function filters a 1d input array in an Ornstein-Uhlenbeck fashion.
    This is to generate correlated a velocity trace.
    ________
    Args:
    gamma1 - time scale
    delta_t - time resolution
    input_array - input array of accelerations
    vel_length - choose the last vel_length elements to avoid correlations with 
                 the initial value, which is the length of the velocity trace.
    initial_val - initial value
    """
    output_array = [initial_val]
    e_factor = np.exp(-gamma1*delta_t)
    for ind, ele in enumerate(input_array):
        ele_out = e_factor * output_array[ind] + (1 - e_factor) * ele / gamma1
        output_array.append(ele_out)
    output_array = np.array(output_array[-vel_length:])
    
    return output_array


def get_shift_array(vel_array, delta_t, img):
    """
    Get the arrays that store the shift sizes at each time point.
    The shift sizes can be in degrees or in pixels.
    """
    pix_per_deg = img.shape[1] / 360
    shift_array_deg = [0]
    shift_array_pix = [0]
    for ind, vel in enumerate(vel_array):
        shift = shift_array_deg[ind] + vel * delta_t
        shift_array_deg.append(shift)
        shift_array_pix.append(int(np.round(shift*pix_per_deg)))
        
    return shift_array_deg, shift_array_pix


def add_object_to_image_periodic_boundary(img_array, x, l, row, image_patch):
    """
    Add an object to an image with periodic boundary.
    _________
    Args:
    img_array - the image
    x - position of the left end of the object, in degree (leftmost (rightmost) edge is 0 (360).).
    l - length of the object, in degree
    row - which row in the image the object is at.
    image_patch - image patch as the object.
    """
    L1 = img_array.shape[0]
    L2 = img_array.shape[1]
    xl = int(np.floor((x % 360) / 360 * L2))
    xr = xl + int(np.floor(l / 360 * L2))
    xr = xr % L2
    yu = row
    yd = row + int(np.floor(5 / 97 * L1))
    mask_in_x = np.zeros(L2)
    if xl < xr:
        img_array[yu:yd, xl:xr] = image_patch[:, :]
        mask_in_x[xl:xr] = 1
    else:
        img_array[yu:yd, xl:] = image_patch[:, :(L2-xl)]
        img_array[yu:yd, :xr+1] = image_patch[:, -xr-1:]
        mask_in_x[xl:] = 1
        mask_in_x[:xr] = 1
        
    return img_array, mask_in_x


def get_sin_wave(k, v, L, T, phi=0, dt=0.01, resize=72):
    """
    Args:
    k: wave number, rad/deg
    v: deg/sec
    L: length of the long dimension, deg
    T: length of time, sec
    phi: phase
    dt: temporal resolution, sec
    """
    x = np.arange(1, L+1)
    scene_1d = np.sin(k * (x + phi))
    scene_1d = scene_1d.reshape((1, -1))
    scene_1d = get_filtered_spacial_row(scene_1d, 5)
    
    xt_plot = np.zeros((T, 1, resize))
    for t in range(T):
        xt_plot[t, 0] = get_resized(np.roll(scene_1d, int(np.round(v * dt * t)), axis=1), n_row=1, n_col=resize)

    return xt_plot


def get_2d_gliders(L, T, corr_order, parity, trend=False):
    """
    Args:
    L: spacial dimension
    T: temporal dimension
    corr_order: 'two point' or 'three point'
    parity: -1 or 1
    trend: 'diverging' or 'converging', for 'three point' only
    """
    xt_plot = np.zeros((T, 1, L))
    xt_plot[0, 0, :] = ((np.random.random(L) > 0.5).astype(float) - 0.5) * 2
    xt_plot[:, 0, 0] = ((np.random.random(T) > 0.5).astype(float) - 0.5) * 2
    
    if corr_order == 'two point':
        for t in range(1, T):
            for l in range(1, L):
                xt_plot[t, 0, l] = xt_plot[t-1, 0, l-1] * parity
    elif corr_order == 'three point':
        if trend == 'diverging':
            for t in range(1, T):
                for l in range(1, L):
                    xt_plot[t, 0, l] = xt_plot[t-1, 0, l-1] * xt_plot[t, 0, l-1] * parity
        if trend == 'converging':
            for t in range(1, T):
                for l in range(1, L):
                    xt_plot[t, 0, l] = xt_plot[t-1, 0, l-1] * xt_plot[t-1, 0, l] * parity
                    
    return xt_plot


def up_sampling(xt_plot, a=1, b=1):
    """
    Args:
    xt_plot: xt plot, numpy array
    a: upsample scale along each column
    b: upsample scale along each row
    """
    xt_plot_up = xt_plot.repeat(a, axis=0).repeat(b, axis=-1)
    
    return xt_plot_up

def down_sampling(xt_plot_up, a=1, b=1):
    """
    Args:
    xt_plot_up: xt plot, numpy array
    a: downsample scale along each column
    b: downsample scale along each row
    """
    xt_plot = xt_plot_up[::a, :, ::b]
    
    return xt_plot

def additional_processing(xt_plot, a, b, shift, sigma_for_gaussian, scaling=1):
    xt_plot_up = up_sampling(xt_plot, a, b)
    shape_info = xt_plot_up.shape
    xt_plot_up = xt_plot_up.squeeze()
    xt_plot_up = get_filtered_spacial_row(xt_plot_up, sigma_for_gaussian)
    xt_plot_up = np.roll(xt_plot_up, shift, axis=-1)
#     xt_plot_up = hpfn.get_standardized_row(xt_plot_up)
    xt_plot_up = xt_plot_up.reshape(shape_info)
    xt_plot = down_sampling(xt_plot_up, a, b) * scaling
    
    return xt_plot

def get_elongated(xt_plot, c=10, T=50):
    xt_plot_e = up_sampling(xt_plot, a=c, b=1)
    xt_plot_e_out = np.zeros((c, ) + xt_plot.shape)
    for cc in range(c):
        if cc == 0:
            xt_plot_e_out[cc] = xt_plot_e[-T-cc:, :, :]
        else:
            xt_plot_e_out[cc] = xt_plot_e[-T-cc:-cc, :, :]
    
    return xt_plot_e_out


def get_spectrum_shuffled(input_array):
    array_fft = fft.fft2(input_array)
    for ii in range(10):
        np.random.shuffle(array_fft)
        np.random.shuffle(array_fft.T)
    array_ifft = fft.ifft2(array_fft)
    
    return array_ifft


def get_phase_randomized_correlated(input_array, rand_type='Gaussian', delete_zero_freq=False):
    array_fft = fft.fft2(input_array)
    array_fft_length = np.absolute(array_fft)
    if delete_zero_freq:
        array_fft_length[0, 0] = 0

    if rand_type == 'uniform':
        rand_phase = np.random.random(array_fft.shape) 
    elif rand_type == 'Gaussian':
        rand_phase = np.random.normal(loc=0., scale=1, size=array_fft.shape)
    rand_phase = fft.fft2(rand_phase)
    rand_phase = rand_phase / np.absolute(rand_phase)
    array_fft_rand = array_fft_length * rand_phase
    array_ifft = fft.ifft2(array_fft_rand)
    
    return array_ifft


def generate_random_phases(K_row, K_col):
    """
    So far, only works for both K_row and K_col being odd.
    """
    random_phases = np.zeros((K_row, K_col))
    # First row
    random_phases[0, 1:] = (np.random.random(K_col-1) - 0.5) * np.pi * 2
    # First column
    random_phases[1:, 0] = (np.random.random(K_row-1) - 0.5) * np.pi * 2
    # positive, positive region
    random_phases[1:int((K_row-1)/2+1), 1:int((K_col-1)/2+1)] = (np.random.random((int((K_row-1)/2), int((K_col-1)/2))) - 0.5) * np.pi * 2
    # positive, negative region
    random_phases[1:int((K_row-1)/2+1), int((K_col-1)/2+1):] = (np.random.random((int((K_row-1)/2), int((K_col-1)/2))) - 0.5) * np.pi * 2
    # negative, negative region
    random_phases[int((K_row-1)/2+1):, int((K_col-1)/2+1):] = -random_phases[1:int((K_row-1)/2+1), 1:int((K_col-1)/2+1)][::-1, ::-1]
    # negative, positive region
    random_phases[int((K_row-1)/2+1):, 1:int((K_col-1)/2+1)] = -random_phases[1:int((K_row-1)/2+1), int((K_col-1)/2+1):][::-1, ::-1]
    
    return random_phases

    
def get_phase_randomized(input_array, delete_zero_freq=False):
    array_fft = fft.fft2(input_array)
    array_fft_length = np.absolute(array_fft)
    if delete_zero_freq:
        array_fft_length[0, 0] = 0
    
    K_row, K_col = array_fft.shape
    rand_phase = generate_random_phases(K_row, K_col)
    array_fft_rand = array_fft_length * np.exp(1j*rand_phase)
    array_ifft = fft.ifft2(array_fft_rand)
    
    return array_ifft


def get_one_checker_board_with_corr(K_row, K_col, scl, size=5):
    img = np.zeros((K_row, K_col))
    checker_size = int(K_col / 360 * size)
    krow = int(K_row / checker_size)
    kcol = int(K_col / checker_size)
    
    for indr in range(krow):
        for indc in range(kcol):
            value = np.sign(np.random.random() - 0.5)
            img[indr*checker_size:(indr+1)*checker_size, indc*checker_size:(indc+1)*checker_size] = value
    img = img * scl
    
    return img


####### For plotting #######
def get_hist(arr, bin_range=False, density=False):
    arr = arr.flatten()
    bins = int(np.sqrt(arr.shape[0]))
    bins = np.minimum(bins, 1000)
    if bin_range:
        hist, bin_edges = np.histogram(arr, bins, range=bin_range, density=density)
    else:
        hist, bin_edges = np.histogram(arr, bins, density=density)
    bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
    
    return hist, bin_centers


def get_sigmoid(a, b, x):
    y = 1. / (1. + np.exp(-a*x-b))
    
    return y


def get_cumulants(arr, round_deci=3):
    mean_ = np.round(arr.flatten().mean(), round_deci)
    std_ = np.round(arr.flatten().std(), round_deci)
    skew_ = np.round(skew(arr.flatten()), round_deci)
    kurtosis_ = np.round(kurtosis(arr.flatten()), round_deci)

    return mean_, std_, skew_, kurtosis_



















