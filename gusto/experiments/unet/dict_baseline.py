'''Dictionary Learning baseline with 2 timepoints and 383 features.'''

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader

from gusto.lib.data import TRAINING_DATA, VALIDATION_DATA

from gusto.experiments.dict_learning.lib.model import Alpha, DictLearning, LSTMLearning
from gusto.experiments.dict_learning.lib.dataset import DictionaryDataset, DictionaryLSTMDataset

from gusto.experiments.unet.lib.utils import CHECKPOINT_DIR, LOGDIR

from gusto.lib.utils import get_logger


def get_model_checkpoint_callback(name, filename, monitor='train_loss_alpha'):
    '''Get the model checkpoint callback for the Lightning model'''
    chkpoint_dir = CHECKPOINT_DIR.joinpath(f'{name}')
    if not chkpoint_dir.parent.is_dir():
        chkpoint_dir.parent.mkdir(parents=True)
    return ModelCheckpoint(dirpath=chkpoint_dir, filename=filename, save_top_k=1, monitor=monitor)


def run_step_one(latent_dim, name, dataset_fpath: Path):
    '''Step one learns the dictionary D and training input & output alphas'''
    dataset = DictionaryDataset(dataset_fpath, x_indices=[0, 1, 2])
    train_alpha = Alpha(input_num=dataset.input_data.shape[0], latent_dim=latent_dim)
    model = DictLearning(train_alpha, latent_dim, train_alpha_only=False)
    trainer = L.Trainer(
        max_epochs=1000,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=1,
        logger=get_logger(f'{name}_step1', LOGDIR),
        callbacks=[get_model_checkpoint_callback(name, 'step_one')],
    )
    trainer.fit(model, DataLoader(dataset))


def run_step_two(latent_dim, name, num_subjects, dataset_fpath: Path):
    '''Step two freezes D and learns learns the validation input alpha'''
    dataset = DictionaryDataset(dataset_fpath, x_indices=[0, 1])
    train_alpha = Alpha(input_num=num_subjects * 3, latent_dim=latent_dim)
    val_alpha = Alpha(input_num=dataset.input_data.shape[0], latent_dim=latent_dim)
    model = DictLearning.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path=CHECKPOINT_DIR.joinpath(name, 'step_one.ckpt'),
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
        logger=get_logger(f'{name}_step2', LOGDIR),
        callbacks=[get_model_checkpoint_callback(name, 'step_two')],
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
        checkpoint_path=CHECKPOINT_DIR.joinpath(name, 'step_one.ckpt'),
        alpha=Alpha(input_num=train_num_subjects * 3, latent_dim=latent_dim),
        latent_dim=latent_dim,
        train_alpha_only=False,
    )
    val_dict = DictLearning.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path=CHECKPOINT_DIR.joinpath(name, 'step_two.ckpt'),
        alpha=Alpha(input_num=val_num_subjects * 2, latent_dim=latent_dim),
        latent_dim=latent_dim,
        train_alpha_only=True,
    )
    train_dataset = DictionaryLSTMDataset(
        train_num_subjects,
        train_dict.alpha.alpha_params.detach(),
        train_fpath,
        x_indices=[0, 1],
        y_indices=[2],
    )
    val_dataset = DictionaryLSTMDataset(
        val_num_subjects,
        val_dict.alpha.alpha_params.detach(),
        val_fpath,
        x_indices=[0, 1],
        y_indices=[2],
    )
    model = lstm_class(train_dict.dict_encoder, latent_dim=latent_dim)
    trainer = L.Trainer(
        max_epochs=1000,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=1,
        logger=get_logger(f'{name}_step3', LOGDIR),
        callbacks=[get_model_checkpoint_callback(name, 'step_three', monitor='val_loss_lstm')],
    )
    trainer.fit(model, DataLoader(train_dataset), DataLoader(val_dataset))


def run_baseline():
    '''Run the baseline experiment.'''
    latent_size = 128
    num_train = 122
    num_val = 15
    test_name = f'{Path(__file__).stem}'
    run_step_one(latent_size, test_name, TRAINING_DATA)
    run_step_two(latent_size, test_name, num_train, VALIDATION_DATA)
    run_step_three(
        latent_size, test_name, LSTMLearning, num_train, num_val, TRAINING_DATA, VALIDATION_DATA
    )


if __name__ == '__main__':
    run_baseline()
