'''First experiment using Dictionary learning with LSTM model'''

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from gusto.lib.data import TRAINING_DATA, VALIDATION_DATA
from gusto.experiments.dict_learning.lib.utils import CHECKPOINT_DIR
from gusto.experiments.dict_learning.lib.utils import get_logger, get_model_checkpoint_callback
from gusto.experiments.dict_learning.lib.model import Alpha, DictLearning, SimpleLSTMLearning
from gusto.experiments.dict_learning.lib.dataset import DictionaryDataset, DictionaryLSTMDataset


def run_step_one(latent_dim, name, dataset_fpath: Path):
    '''Step one learns the dictionary D and training input & output alphas'''
    dataset = DictionaryDataset(dataset_fpath, x_indices=[0, 1, 2, 3])
    train_alpha = Alpha(input_num=dataset.input_data.shape[0], latent_dim=latent_dim)
    model = DictLearning(train_alpha, latent_dim, train_alpha_only=False)
    trainer = L.Trainer(
        max_epochs=1000,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=1,
        logger=get_logger(f'{name}_step1', latent_dim),
        callbacks=[get_model_checkpoint_callback(f'{name}', latent_dim, 'step_one')],
    )
    trainer.fit(model, DataLoader(dataset))


def run_step_two(latent_dim, name, num_subjects, dataset_fpath: Path):
    '''Step two freezes D and learns learns the validation input alpha'''
    dataset = DictionaryDataset(dataset_fpath, x_indices=[0, 1, 2])
    train_alpha = Alpha(input_num=num_subjects * 4, latent_dim=latent_dim)
    val_alpha = Alpha(input_num=dataset.input_data.shape[0], latent_dim=latent_dim)
    model = DictLearning.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path=CHECKPOINT_DIR.joinpath(f'{name}_{latent_dim}', 'step_one.ckpt'),
        alpha=train_alpha,
        latent_dim=latent_dim,
        train_alpha_only=True,
    )
    model.alpha = val_alpha
    trainer = L.Trainer(
        max_epochs=1000,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=1,
        logger=get_logger(f'{name}_step2', latent_dim),
        callbacks=[get_model_checkpoint_callback(name, latent_dim, 'step_two')],
    )
    trainer.fit(model, DataLoader(dataset))


def run_step_three(
    latent_dim,
    name,
    lstm_class,
    train_num_subjects,
    val_num_subjects,
    train_fpath: Path,
    val_fpath: Path,
):
    '''Step three trains an LSTM model with the frozen train dictionary D and alpha coefficients,
    and validates against frozen input validation alpha coefficients
    '''
    train_dict = DictLearning.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path=CHECKPOINT_DIR.joinpath(f'{name}_{latent_dim}', 'step_one.ckpt'),
        alpha=Alpha(input_num=train_num_subjects * 4, latent_dim=latent_dim),
        latent_dim=latent_dim,
        train_alpha_only=False,
    )
    val_dict = DictLearning.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path=CHECKPOINT_DIR.joinpath(f'{name}_{latent_dim}', 'step_two.ckpt'),
        alpha=Alpha(input_num=val_num_subjects * 3, latent_dim=latent_dim),
        latent_dim=latent_dim,
        train_alpha_only=True,
    )
    train_dataset = DictionaryLSTMDataset(
        train_num_subjects, train_dict.alpha.alpha_params.detach(), train_fpath
    )
    val_dataset = DictionaryLSTMDataset(
        val_num_subjects, val_dict.alpha.alpha_params.detach(), val_fpath
    )
    model = lstm_class(train_dict.dict_encoder, latent_dim=latent_dim)
    trainer = L.Trainer(
        max_epochs=1000,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=1,
        logger=get_logger(f'{name}_step3', latent_dim),
        callbacks=[
            get_model_checkpoint_callback(name, latent_dim, 'step_three', monitor='val_loss_lstm')
        ],
    )
    trainer.fit(model, DataLoader(train_dataset), DataLoader(val_dataset))


if __name__ == '__main__':
    latent_size = 32
    num_train = 122
    num_val = 15
    test_name = f'{Path(__file__).stem}'
    run_step_one(latent_size, test_name, TRAINING_DATA)
    run_step_two(latent_size, test_name, num_train, VALIDATION_DATA)
    run_step_three(
        latent_size,
        test_name,
        SimpleLSTMLearning,
        num_train,
        num_val,
        TRAINING_DATA,
        VALIDATION_DATA,
    )
