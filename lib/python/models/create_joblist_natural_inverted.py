#!/usr/bin/env python


from absl import app
from absl import flags
from ml_collections import config_flags

_FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'configuration file', lock_config=True)
flags.mark_flags_as_required(['config'])

def main(_):
    # get lists of hyper parameters
    _VALUE = _FLAGS.config
    D_cnn_list = _VALUE.D_cnn_list # list of depth
    C_list = _VALUE.C_list # list of number of independent channels
    Repeat = _VALUE.Repeat # repeat for different initializations

    # Save info to a joblist file
    log_file = 'joblist_natural_inverted.txt'
    with open(log_file, 'w') as f:
        f.truncate()
        for R in range(100, Repeat):
            for D_cnn in D_cnn_list:
                for C in C_list:
                    f.write(f'module load miniconda; source activate py3_pytorch; python3 train_validate_run.py --config=../configs/configs_model_natural_inverted.py --D_cnn={D_cnn} --C={C} --R={R+1}\n')
                            
if __name__ == '__main__':
  app.run(main)

























