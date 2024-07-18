#!/usr/bin/env python

# Example code from 
# https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/
# By Christian Versloot
# but with modifications


import sys
sys.path.append('../helper/')

import os
import numpy as np
import glob
import time
from datetime import datetime
from joblib.externals.loky.backend.context import get_context

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import models
import helper_functions as hpfn


# Function to reset the weights before the training
def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


# training function
def train_run(config, network_model, transformed_dataset, loss_function):

    # Configuration options
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    lr = config.lr
    l1_factor = config.l1_factor
    l2_factor = config.l2_factor
    constraints_weight = config.constraints_weight
    constraints_bias = config.constraints_bias
    constrained_weight_layers = config.constrained_weight_layers
    constrained_bias_layers = config.constrained_bias_layers
    training_data_path = config.training_data_path
    save_folder = config.save_folder

    # Save config info to a log file
    log_file = save_folder + 'log.txt'
    with open(log_file, 'w') as f:
        f.write('Model setup:\n')
        f.write('----------------------------------------\n')
        f.write(f'network_model: {network_model}\n')
        f.write(f'constraints_weight: {constraints_weight}\n')
        f.write(f'constraints_bias: {constraints_bias}\n')
        f.write(f'constrained_weight_layers: {constrained_weight_layers}\n')
        f.write(f'constrained_bias_layers: {constrained_bias_layers}\n')
        f.write(f'num_epochs: {num_epochs}\n')
        f.write(f'batch_size: {batch_size}\n')
        f.write(f'lr: {lr}\n')
        f.write(f'training data path: {training_data_path}\n')

    # Define data loaders for training
    trainloader = torch.utils.data.DataLoader(
                      transformed_dataset, 
                      batch_size=batch_size)

    # Init the neural network
    network_model.apply(reset_weights)
    # Re-Init the constrained parameters
    for constrained_weight_layer in constrained_weight_layers:
        nn.init.uniform_(network_model._modules[constrained_weight_layer].weight, 0, 1)
    for constrained_bias_layer in constrained_bias_layers:
        nn.init.uniform_(network_model._modules[constrained_bias_layer].bias, -1, 0)

    # Saving the initialized model
    save_path = save_folder + f'model_init.pth'
    torch.save(network_model.state_dict(), save_path)

    # Initialize optimizer
    optimizer = torch.optim.Adam(network_model.parameters(), lr=lr, weight_decay=0)

    # List to save train losses
    train_loss = []
    
    ####### Training #######
    print('Training:')

    # Run the training loop for defined number of epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        
        ####### Training #######
        train_loss_epoch = 0
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader):
            # Get inputs
            inputs, targets = data['movie'], data['target']

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = network_model(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)

            # Compute l1 and l2 regularization
            l1_regu = 0
            l2_regu = 0
            if l1_factor > 0:
                l1_penalty = nn.L1Loss(size_average=False, reduction='sum')
                for param in network_model.parameters():
                    target_regu = torch.zeros_like(param)
                    l1_regu = l1_regu + l1_penalty(param, target_regu)
            if l2_factor > 0:
                l2_penalty = nn.MSELoss(size_average=False, reduction='sum')
                for param in network_model.parameters():
                    target_regu = torch.zeros_like(param)
                    l2_regu = l2_regu + l2_penalty(param, target_regu)

            # Add regularization to loss
            loss = loss + l1_factor * l1_regu + l2_factor * l2_regu

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Apply constraints
            for constrained_weight_layer in constrained_weight_layers:
                network_model._modules[constrained_weight_layer].apply(constraints_weight)
            for constrained_bias_layer in constrained_bias_layers:
                network_model._modules[constrained_bias_layer].apply(constraints_bias)

            train_loss_epoch = train_loss_epoch + loss.item()

        train_loss_epoch = train_loss_epoch / (i+1)
        train_loss.append(train_loss_epoch)

        # Print statistics
        print(f'Training loss for epoch {epoch+1} is {train_loss_epoch}')
        print(f'Time for epoch {epoch+1} is {time.time()-start_time}')

    # Saving the model
    save_path = save_folder + f'model.pth'
    torch.save(network_model.state_dict(), save_path)
    
    # Saving the loss function, training
    save_path = save_folder + f'train_loss.pth'
    torch.save(train_loss, save_path)



