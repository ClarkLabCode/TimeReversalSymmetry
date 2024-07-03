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
    Repeat = _VALUE.Repeat # repeat for different initializations

    # Save info to a joblist file
    log_file = 'joblist_natural_phase_randomized_test.txt'
    with open(log_file, 'w') as f:
        f.truncate()
        for R in range(Repeat):
            f.write(f'module load miniconda; source activate py3_pytorch; python3 test_run.py --config=../configs/configs_model_natural_phase_randomized.py --R={R+1}\n')
                            
if __name__ == '__main__':
  app.run(main)

























