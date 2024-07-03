#!/usr/bin/env python

"""
Various neural network models for rigid motion estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSpaceInvLRSymm(nn.Module):
    """
      CNN with spacial invariance and left-right symmetry.
    """
    def __init__(self, D_cnn=1, C=2, k=3, od=1, activationf='ReLU', T=50):
        """
        D_cnn: depth, number of cnn layers.
        C: number of independent channels. Actual number is 2 * C.
        k: the longer dimension of the kernel (in multiples of 5 deg)
        od: output dimenstion.
        activationf: activation function.
        L: the longer dimension of the input (in multiples of 5 deg)
        T: length in time (in multiples of 10 ms)
        """
        super().__init__()
        self.D_cnn = D_cnn
        self.C = C
        self.activationf = activationf

        # cnn layers
        self.cnn_layers = nn.ModuleList([])
        # first cnn layer
        self.cnn_layers.append(nn.Conv2d(T, C, (1, k), padding=(0, int((k-1)/2)), padding_mode='circular'))
        # other cnn layers
        for d in range(D_cnn-1):
            self.cnn_layers.append(nn.Conv2d(C, C, (1, k), padding=(0, int((k-1)/2)), padding_mode='circular'))

        # output layer
        self.output_layer = nn.Linear(self.C, od, bias=False)

    def forward_module(self, input_data, activationf):
        outputs = input_data
        # cnn layers
        for d in range(self.D_cnn):
            outputs_plus = activationf(self.cnn_layers[d](outputs))
            outputs_minus = activationf(self.cnn_layers[d](torch.flip(outputs, [-1])))
            outputs_minus = torch.flip(outputs_minus, [-1])
            outputs = outputs_plus - outputs_minus
        # sum over space
        outputs = torch.sum(outputs, (-1, -2)) / torch.sqrt(torch.tensor(outputs.size()[-1]*outputs.size()[-2]))
        # outputs = torch.mean(outputs, (-1, -2))
        outputs = outputs.view(-1, self.C)
        # output layer
        outputs = self.output_layer(outputs)

        return outputs
    
    def forward(self, input_data):
        if self.activationf == 'ReLU':
            outputs1 = self.forward_module(input_data, F.relu)
            outputs2 = self.forward_module(torch.flip(input_data, [-1]), F.relu)
            outputs = outputs1 - outputs2
        elif self.activationf == 'LeakyReLU':
            outputs1 = self.forward_module(input_data, F.leaky_relu)
            outputs2 = self.forward_module(torch.flip(input_data, [-1]), F.leaky_relu)
            outputs = outputs1 - outputs2
        elif self.activationf == 'ELU':
            outputs1 = self.forward_module(input_data, F.elu)
            outputs2 = self.forward_module(torch.flip(input_data, [-1]), F.elu)
            outputs = outputs1 - outputs2
        elif self.activationf == 'SoftThresholding':
            outputs1 = self.forward_module(input_data, F.softshrink)
            outputs2 = self.forward_module(torch.flip(input_data, [-1]), F.softshrink)
            outputs = outputs1 - outputs2
        
        return outputs


####### Neural networks #######
class NNLRSymmDense(nn.Module):
    """
      Dense Neural Network models with left-right antisymmetry.
    """
    def __init__(self, W_list, od=1, L=72, T=50, activationf=nn.ReLU()):
        """
        W_list: list of widths, number of neurons in each dense layer.
        od: output dimenstion.
        L: the longer spatial dimension of the input (the other is 1)
        T: length in time dimension of the input (in multiples of 1/100 s)
        """
        super().__init__()
        self.L = L
        self.T = T
        self.activationf = activationf

        self.dense_layers = nn.ModuleList([])
        # first layer
        self.dense_layers.append(nn.Linear(int(L*T), W_list[0]))
        # other layers
        if len(W_list) > 1:
            for d in range(len(W_list)-1):
                self.dense_layers.append(nn.Linear(W_list[d], W_list[d+1]))
        # output layer
        self.dense_layers.append(nn.Linear(W_list[-1], od, bias=False))
    
    def forward_module(self, input_data):
        outputs = input_data.clone()
        for d in range(len(self.dense_layers)):
            outputs = self.dense_layers[d](outputs)
            outputs = self.activationf(outputs)

        return outputs

    def forward(self, input_data):
        input_data_normal = input_data.view(-1, self.T*self.L)
        outputs1 = self.forward_module(input_data_normal)
        input_data_flipped = torch.flip(input_data, [-1]).view(-1, self.T*self.L)
        outputs2 = self.forward_module(input_data_flipped)
        outputs = outputs1 - outputs2
        
        return outputs


####### Constraints #######
class general_constraint_positive(object):
    def __init__(self):
        pass
    
    def __call__(self, module):
        md = module.data
        md = md.clamp(0., 1e10)
        module.data = md


class weight_constraint_positive(object):
    def __init__(self):
        pass
    
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0., 1e10)
            module.weight.data = w


class bias_constraint_negative(object):
    def __init__(self):
        pass
    
    def __call__(self, module):
        if hasattr(module, 'bias'):
            w = module.bias.data
            w = w.clamp(-1e10, 0.)
            module.bias.data = w




    