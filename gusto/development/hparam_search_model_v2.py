'''Model for hyperparameter search using Lightning and Optuna using the interpolation as training dataset'''

from pathlib import Path
from itertools import combinations


import optuna
from optuna.integration import PyTorchLightningPruningCallback

from lightning.pytorch import Trainer

from torch.utils.data import DataLoader
from gusto.lib.data import GustoDataset, GustoInterpDataset
from gusto.lib.torch_lightning import LITModel
from gusto.experiments.cnn_autoencoder.lib.autoencoder_hparams import AutoEncoder
from gusto.development.hparam_search_model_v1 import get_trial_params, trial_params_to_hparams

EPOCHS = 50
PERCENT_VALID_EXAMPLES = 0.5


def get_branch_b_combos():
    '''Get all possible combinations of branches for block B'''
    all_combos = []
    original_list = [0, 1, 2, 3]
    for r in range(1, len(original_list) + 1):
        comb = list(combinations(original_list, r))
        all_combos.extend(comb)
    return all_combos


class LITAutoEncoder(LITModel):
    '''Lightning model for AutoEncoder using the GUSTO training dataset'''

    def __init__(self, **kwargs):
        super().__init__()
        self.model = AutoEncoder(**kwargs)

    def forward(self, input_cpg, time_in, time_out):
        # pylint: disable=arguments-differ
        return self.model(input_cpg, time_in, time_out)


def get_datasets():
    '''Get the training and validation datasets'''
    train_dataset = GustoInterpDataset(
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
    return train_dataloader, val_dataloader


def objective(trial: optuna.trial.Trial) -> float:
    '''Objective function for Optuna'''
    # We optimize the number of layers, hidden units in each layer and dropouts.
    train_dataloader, val_dataloader = get_datasets()

    params = get_trial_params(trial)
    hparams = trial_params_to_hparams(params)

    model = LITAutoEncoder(**hparams)

    trainer = Trainer(
        logger=True,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        accelerator='gpu',
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_loss')],
    )
    trainer.logger.log_hyperparams(hparams)  # type: ignore
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return trainer.callback_metrics['val_loss'].item()


if __name__ == "__main__":

    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        storage="sqlite:///model_v1_random_fix.sqlite3",
        sampler=optuna.samplers.RandomSampler(),
        study_name='model_v2',
        direction='minimize',
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=3000, catch=[RuntimeError])

    print(f'Number of finished trials: {len(study.trials)}')
    print('Best trial:')
    print(f'  Value: {study.best_trial.value}')
    print('  Params: ')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')
