'''Utility functions for CNN autoenconder experiments'''

from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

LOGDIR = Path(__file__).parent.parent.joinpath('logs')
CHECKPOINT_DIR = LOGDIR.parent.joinpath('checkpoints')


def get_logger(test_name: str):
    '''Get the logger for the Lightning model'''
    logdir = LOGDIR.joinpath(test_name)
    if not logdir.parent.is_dir():
        logdir.parent.mkdir(parents=True)
    return TensorBoardLogger(
        save_dir=str(logdir.parent.parent),
        name='logs',
        version=test_name,
    )


def get_model_checkpoint_callback(test_name: str):
    '''Get the model checkpoint callback for the Lightning model'''
    chkpoint_dir = CHECKPOINT_DIR.joinpath(test_name)
    if not chkpoint_dir.parent.is_dir():
        chkpoint_dir.parent.mkdir(parents=True)
    return ModelCheckpoint(
        dirpath=chkpoint_dir,
        filename='{epoch}-{val_loss:.5f}',
        save_top_k=3,
        monitor='val_loss',
    )
