'''First version of the model using the GUSTO dataset'''

from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import optuna
from torch.utils.data import DataLoader
from gusto.development.hparam_search_model_v1 import trial_params_to_hparams
from gusto.lib.data import GustoDataset
from gusto.lib.torch_lightning import LITModel
from gusto.experiments.cnn_autoencoder.lib.autoencoder_hparams import AutoEncoderWeightedAverage

torch.set_float32_matmul_precision('high')


class LITAutoEncoder(LITModel):
    '''Lightning model for AutoEncoder using the GUSTO training dataset'''

    def __init__(self, **kwargs):
        super().__init__()
        self.model = AutoEncoderWeightedAverage(**kwargs)

    def forward(self, input_cpg, time_in, time_out):
        # pylint: disable=arguments-differ
        return self.model(input_cpg, time_in, time_out)


def get_logger():
    '''Get the logger for the Lightning model'''
    test_name = Path(__file__).stem
    logdir = Path.home().joinpath('Dev', 'git', 'gusto', 'logs', test_name)
    if not logdir.parent.is_dir():
        logdir.parent.mkdir(parents=True)
    return TensorBoardLogger(
        save_dir=str(logdir.parent.parent),
        name='logs',
        version=test_name,
    )


if __name__ == '__main__':
    trainer = Trainer(max_epochs=400, logger=get_logger(), accelerator='gpu', devices=1)
    train_dataset = GustoDataset(
        Path.home().joinpath(
            'Dev', 'git', 'gusto', 'data', '3-9-48-72months_383CpGs_153indivs_train.pkl'
        )
    )
    val_dataset = GustoDataset(
        Path.home().joinpath(
            'Dev', 'git', 'gusto', 'data', '3-9-48-72months_383CpGs_153indivs_val.pkl'
        )
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True
    )
    study = optuna.create_study(
        storage='sqlite:///model_v1_random_fix.sqlite3',
        sampler=optuna.samplers.RandomSampler(),
        study_name='model_v1_random_fix2',
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    hparams = trial_params_to_hparams(study.best_trial.params)
    model = LITAutoEncoder(**hparams)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
