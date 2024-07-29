from pathlib import Path

from gusto.lib.data import TRAINING_DATA, VALIDATION_DATA
from gusto.experiments.dict_learning.lib.model import LSTMLearning
from gusto.experiments.dict_learning.experiment_1 import run_step_one, run_step_two, run_step_three


def repeat_experiment(latent_size):
    num_train = 122
    num_val = 15
    test_name = f'{Path(__file__).stem}'
    run_step_one(latent_size, test_name, TRAINING_DATA)
    run_step_two(latent_size, test_name, num_train, VALIDATION_DATA)
    run_step_three(latent_size, test_name, LSTMLearning, num_train, num_val)


if __name__ == '__main__':
    for latent in [32, 64, 128, 256]:
        repeat_experiment(latent)
