'''Utility functions for CNN autoenconder experiments'''

from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def get_logger(test_name: str, logdir: Path):
    '''Get the logger for the Lightning model'''
    if not logdir.is_dir():
        logdir.mkdir(parents=True)
    return TensorBoardLogger(
        save_dir=str(logdir.parent),
        name='logs',
        version=test_name,
    )


def get_model_checkpoint_callback(test_name: str, chkpoint_dir: Path):
    '''Get the model checkpoint callback for the Lightning model'''
    if not chkpoint_dir.is_dir():
        chkpoint_dir.mkdir(parents=True)
    return ModelCheckpoint(
        dirpath=chkpoint_dir.joinpath(test_name),
        filename='{epoch}-{val_loss:.5f}',
        save_top_k=3,
        monitor='val_loss',
    )
