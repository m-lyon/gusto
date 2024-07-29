from pathlib import Path

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

LOGDIR = Path(__file__).parent.parent.joinpath('logs')
CHECKPOINT_DIR = LOGDIR.parent.joinpath('checkpoints')


def get_logger(test_name, latent_dim):
    '''Get the logger for the Lightning model'''
    if not LOGDIR.is_dir():
        LOGDIR.mkdir(parents=True)
    return TensorBoardLogger(
        save_dir=str(LOGDIR.parent),
        name='logs',
        version=f'{test_name}_{latent_dim}',
    )


def get_model_checkpoint_callback(name, latent_dim, filename, monitor='train_loss_alpha'):
    '''Get the model checkpoint callback for the Lightning model'''
    chkpoint_dir = CHECKPOINT_DIR.joinpath(f'{name}_{latent_dim}')
    if not chkpoint_dir.parent.is_dir():
        chkpoint_dir.parent.mkdir(parents=True)
    return ModelCheckpoint(dirpath=chkpoint_dir, filename=filename, save_top_k=1, monitor=monitor)
