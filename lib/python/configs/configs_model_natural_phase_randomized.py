#!/usr/bin/env python

######################
# Configurations for models.
######################
import sys
sys.path.append('../configs/')

from ml_collections import config_dict
import numpy as np
import importlib

import configs_data_natural_phase_randomized as confdn
import configs_model_base as confmb
importlib.reload(confdn)
importlib.reload(confmb)

configsdata = confdn.get_config()
configsmodel = confmb.get_config()


def get_config():
  """
  With spacial invariance and left-right symmetry
  """
  config = config_dict.ConfigDict()
  config.training_data_path = configsdata.training_data_path
  config.testing_data_path = configsdata.testing_data_path
  config.scope = configsdata.scope

  config.T = configsmodel.T
  config.D_cnn_list = configsmodel.D_cnn_list
  config.C_list = configsmodel.C_list
  config.k = configsmodel.k
  config.od = configsmodel.od
  config.activationf = configsmodel.activationf
  config.num_epochs = configsmodel.num_epochs
  config.batch_size = configsmodel.batch_size
  config.lr = configsmodel.lr
  config.l1_factor = configsmodel.l1_factor
  config.l2_factor = configsmodel.l2_factor
  config.constraints_weight = configsmodel.constraints_weight
  config.constraints_bias = configsmodel.constraints_bias
  config.constrained_weight_layers = configsmodel.constrained_weight_layers
  config.constrained_bias_layers = configsmodel.constrained_bias_layers
  config.loss_function = configsmodel.loss_function
  config.Repeat = configsmodel.Repeat

  config.contrast_stats = 'NA'
  config.save_folder = '../results/trained_models_natural_phase_randomized_true_left_right_symm/'
  config.figure_folder = '../results/preliminary_figures_natural_phase_randomized_true_left_right_symm/'

  
  
  return config
