'''Third experiment using Dictionary learning with LSTM model
This experiment is similar to the first experiment, but uses a more complex LSTM model, with
timepoint positional encoding and decoding
'''

from pathlib import Path

from gusto.lib.data import TRAINING_DATA, VALIDATION_DATA
from gusto.experiments.dict_learning.lib.model import LSTMLearning
from gusto.experiments.dict_learning.experiment_1 import run_step_one, run_step_two, run_step_three


if __name__ == '__main__':
    latent_size = 32
    num_train = 122
    num_val = 15
    test_name = f'{Path(__file__).stem}'
    run_step_one(latent_size, test_name, TRAINING_DATA)
    run_step_two(latent_size, test_name, num_train, VALIDATION_DATA)
    run_step_three(
        latent_size, test_name, LSTMLearning, num_train, num_val, TRAINING_DATA, VALIDATION_DATA
    )
