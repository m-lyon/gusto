'''Gusto model prediction using single CpG site timepoints'''

from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
from gusto.experiments.cnn_autoencoder.lib.autoencoder_hparams import LinearLayer
from gusto.lib.data import GustoSingleCpGDataset
from gusto.lib.torch_lightning import LITModel


torch.set_float32_matmul_precision('high')


class LITAutoEncoder(LITModel):
    '''Lightning model for AutoEncoder using the interpolation as training dataset'''

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            LinearLayer(3, 32), LinearLayer(32, 32), LinearLayer(32, 1)
        )

    def forward(self, input_cpg, time_in, time_out):
        # pylint: disable=arguments-differ
        return self.model(input_cpg)


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


def get_model_checkpoint_callback():
    '''Get the model checkpoint callback for the Lightning model'''
    test_name = Path(__file__).stem
    chkpoint_dir = Path.home().joinpath('Dev', 'git', 'gusto', 'checkpoints', test_name)
    if not chkpoint_dir.parent.is_dir():
        chkpoint_dir.parent.mkdir(parents=True)
    return ModelCheckpoint(
        dirpath=chkpoint_dir,
        filename='{epoch}-{val_loss:.5f}',
        save_top_k=3,
        monitor='val_loss',
    )


if __name__ == '__main__':
    trainer = Trainer(
        max_epochs=400,
        logger=get_logger(),
        accelerator='gpu',
        devices=1,
        callbacks=[get_model_checkpoint_callback()],
    )
    train_dataset = GustoSingleCpGDataset(
        Path.home().joinpath(
            'Dev', 'git', 'gusto', 'data', '3-9-48-72months_383CpGs_153indivs_train.pkl'
        )
    )
    val_dataset = GustoSingleCpGDataset(
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
    model = LITAutoEncoder()
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
