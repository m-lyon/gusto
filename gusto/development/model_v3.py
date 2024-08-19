'''LITAutoEncoder using the interpolation as training dataset, with CPG site shuffling after every epoch'''

from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from gusto.lib.data import GustoInterpDataModule
from gusto.lib.torch_lightning import LITModel
from gusto.experiments.cnn_autoencoder.lib.autoencoder import AutoEncoder

torch.set_float32_matmul_precision('high')


class LITAutoEncoder(LITModel):
    '''Lightning model for AutoEncoder using the interpolation as training dataset'''

    def __init__(self, latent_size):
        super().__init__()
        self.model = AutoEncoder(latent_size)

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
    trainer = L.Trainer(
        max_epochs=400,
        logger=get_logger(),
        accelerator='gpu',
        devices=1,
        reload_dataloaders_every_n_epochs=1,
    )
    datamodule = GustoInterpDataModule(
        train_filepath=Path.home().joinpath(
            'Dev', 'git', 'gusto', 'data', '3-9-48-72months_383CpGs_153indivs_train.pkl'
        ),
        val_filepath=Path.home().joinpath(
            'Dev', 'git', 'gusto', 'data', '3-9-48-72months_383CpGs_153indivs_val.pkl'
        ),
    )
    model = LITAutoEncoder(32)
    trainer.fit(model, datamodule=datamodule)
