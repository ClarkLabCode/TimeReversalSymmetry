#!/usr/bin/env python

"""
This script generates variables that are used to test the models on the synthetic data, sawtooth and moving edges. It puts all test results 
for different model architectures and different initializations together. ESI: edge selective index.
"""

import os
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import explained_variance_score

l1_regu = 0
l2_regu = 0
k1 = 3
k2 = 3
D_dense = 0
W = 0

rsquare_thres = 0.8

for C in [2, 4]:
    for D_cnn in [1, 2, 3, 4]:
        R_performance_array_phase_random = []

        for R in tqdm(range(500)):
            # phase randomized
            model_folder = f'../results/trained_models_natural_phase_randomized/'
            model_path = model_folder + f'Dcnn{D_cnn}_C{C}_ReLU_R{R+1}/'
            if os.path.exists(model_path + f'y_pred_all.pth'):
                y_test_all = torch.load(model_path + f'y_test_all.pth')
                y_pred_all = torch.load(model_path + f'y_pred_all.pth')
                explained_variance = np.round(explained_variance_score(y_test_all, y_pred_all), 2)
                direction_accuracy = np.round((np.sign(y_test_all) == np.sign(y_pred_all)[:, 0]).sum()/len(y_pred_all), 2)

                if explained_variance >= rsquare_thres:
                    R_performance_array_phase_random.append([R+1, explained_variance, direction_accuracy])

        np.save(f'../results/variables_for_paper/R_performance_array_natural_phase_random_C{C}_D{D_cnn}_thres{rsquare_thres}', R_performance_array_phase_random)
