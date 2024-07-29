'''Fourth experiment with the CNN autoencoder and the GUSTO dataset,
    This experiment uses Optuna to perform hyperparameter optimization for the
    previous three experiments
'''

from itertools import combinations
from typing import Any, Dict

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from lightning.pytorch import Trainer

from gusto.lib.torch_lightning import LITModel
from gusto.experiments.cnn_autoencoder.lib.autoencoder_hparams import AutoEncoder
from gusto.experiments.cnn_autoencoder.lib.autoencoder_hparams import AutoEncoderWeightedAverage
from gusto.experiments.cnn_autoencoder.experiment_1 import get_dataset as get_dataset_exp1
from gusto.experiments.cnn_autoencoder.experiment_3 import get_dataset as get_dataset_exp3
from gusto.lib.utils import get_logger
from gusto.experiments.cnn_autoencoder.lib.utils import LOGDIR


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

    def __init__(self, **kwargs):
        super().__init__()
        self.model = AutoEncoder(**kwargs)

    def forward(self, input_cpg, time_in, time_out):
        # pylint: disable=arguments-differ
        return self.model(input_cpg, time_in, time_out)


class LITAutoEncoderWA(LITModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = AutoEncoderWeightedAverage(**kwargs)

    def forward(self, input_cpg, time_in, time_out):
        # pylint: disable=arguments-differ
        return self.model(input_cpg, time_in, time_out)


def trial_params_to_hparams(params: Dict[str, Any]) -> Dict[str, Any]:
    '''Convert Optuna trial parameters to Lightning model hyperparameters'''
    hparams = {}

    branch_b_total_num_channels = 8 * len(params['branch_b'])
    if params['block_a_layers'] == 1:
        hparams['layer_dims_a'] = (branch_b_total_num_channels * params['layer_dims_a_pt2'],)
    else:
        layer_dims_a_pt1 = tuple(
            params[f'block_a_layer_{i}_dim'] for i in range(params['block_a_layers'] - 1)
        )
        layer_dims_a_pt2 = branch_b_total_num_channels * params['layer_dims_a_pt2']
        hparams['layer_dims_a'] = layer_dims_a_pt1 + (layer_dims_a_pt2,)
    hparams['kernels_a'] = tuple(
        params[f'block_a_layer_{i}_kernel'] for i in range(params['block_a_layers'])
    )
    hparams['batch_norm_a'] = params['batch_norm_a']
    hparams['output_dim_b'] = branch_b_total_num_channels * params['output_dim_b']
    hparams['branches_b'] = params['branch_b']
    hparams['layer_dims_c'] = tuple(
        params[f'block_c_layer_{i}_dim'] for i in range(params['block_c_layers'])
    )
    hparams['kernels_c'] = tuple(
        params[f'block_c_layer_{i}_kernel'] for i in range(params['block_c_layers'])
    )
    hparams['time_embedding_layers'] = (1,) + tuple(
        8 * params[f'time_emb_layer_{i}_dim'] for i in range(params['num_time_emb_layers'])
    )
    hparams['latent_size'] = 8 * params['latent_size']

    return hparams


def get_trial_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    params = {}

    params['block_a_layers'] = trial.suggest_int('block_a_layers', 1, 4)
    for i in range(params['block_a_layers'] - 1):
        params[f'block_a_layer_{i}_dim'] = trial.suggest_int(f'block_a_layer_{i}_dim', 8, 128)
    params['layer_dims_a_pt2'] = trial.suggest_int('layer_dims_a_pt2', 1, 12)
    for i in range(params['block_a_layers']):
        params[f'block_a_layer_{i}_kernel'] = trial.suggest_int(f'block_a_layer_{i}_kernel', 1, 7)
    params['batch_norm_a'] = trial.suggest_categorical('batch_norm_a', [True, False])
    params['output_dim_b'] = trial.suggest_int('output_dim_b', 1, 12)
    params['branch_b'] = trial.suggest_categorical('branch_b', get_branch_b_combos())
    params['batch_norm_b'] = trial.suggest_categorical('batch_norm_b', [True, False])
    params['block_c_layers'] = trial.suggest_int('block_c_layers', 1, 4)
    for i in range(params['block_c_layers']):
        params[f'block_c_layer_{i}_dim'] = trial.suggest_int(f'block_c_layer_{i}_dim', 8, 128)
        params[f'block_c_layer_{i}_kernel'] = trial.suggest_int(f'block_c_layer_{i}_kernel', 1, 7)
    params['batch_norm_c'] = trial.suggest_categorical('batch_norm_c', [True, False])
    params['num_time_emb_layers'] = trial.suggest_int('num_time_emb_layers', 1, 3)
    for i in range(params['num_time_emb_layers']):
        params[f'time_emb_layer_{i}_dim'] = trial.suggest_int(f'time_emb_layer_{i}_dim', 1, 4)
    params['latent_size'] = trial.suggest_int('latent_size', 1, 8)

    return params


def objective(trial: optuna.trial.Trial, get_dataset_func, model_class) -> float:
    '''Objective function for Optuna'''
    # We optimize the number of layers, hidden units in each layer and dropouts.
    train_dataloader, val_dataloader = get_dataset_func()

    params = get_trial_params(trial)
    hparams = trial_params_to_hparams(params)

    model = model_class(**hparams)

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


def run_hyperparam_search_experiment(name: str, get_dataset_func, model_class):
    '''Run the hyperparameter search experiment'''
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        storage='sqlite:///cnn_autoencoder.sqlite3',
        sampler=optuna.samplers.RandomSampler(),
        study_name=name,
        direction='minimize',
        pruner=pruner,
        load_if_exists=True,
    )

    def obj(trial):
        return objective(trial, get_dataset_func, model_class)

    study.optimize(obj, n_trials=3000, catch=[RuntimeError])

    print(f'Number of finished trials: {len(study.trials)}')
    print('Best trial:')
    print(f'  Value: {study.best_trial.value}')
    print('  Params: ')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')

    # Train best model
    hparams = trial_params_to_hparams(study.best_trial.params)
    trainer = Trainer(max_epochs=400, logger=get_logger(name, LOGDIR), accelerator='gpu', devices=1)
    train_dataloader, val_dataloader = get_dataset_func()
    model = model_class(**hparams)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    # run_hyperparam_search_experiment('experiment_1_opt', get_dataset_exp1, LITAutoEncoder)
    run_hyperparam_search_experiment('experiment_2_opt', get_dataset_exp1, LITAutoEncoderWA)
    run_hyperparam_search_experiment('experiment_3_opt', get_dataset_exp3, LITAutoEncoder)
