'''Second experiment using Dictionary learning with LSTM model
This experiment is similar to the first experiment, but uses only complete data,
with no missing values
'''

from pathlib import Path

from gusto.experiments.dict_learning.lib.model import SimpleLSTMLearning
from gusto.lib.data import TRAINING_DATA_NOMISSING, VALIDATION_DATA_NOMISSING
from gusto.experiments.dict_learning.experiment_1 import run_step_one, run_step_two, run_step_three


if __name__ == '__main__':
    latent_size = 32
    num_train = 13
    num_val = 2
    test_name = f'{Path(__file__).stem}'
    run_step_one(latent_size, test_name, TRAINING_DATA_NOMISSING)
    run_step_two(latent_size, test_name, num_train, VALIDATION_DATA_NOMISSING)
    run_step_three(
        latent_size,
        test_name,
        SimpleLSTMLearning,
        num_train,
        num_val,
        TRAINING_DATA_NOMISSING,
        VALIDATION_DATA_NOMISSING,
    )
