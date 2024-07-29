'''Third experiment with the CNN autoencoder and the GUSTO dataset,
    This experiment interpolates the data in the time dimension.
'''

from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from gusto.lib.torch_lightning import LITModel
from gusto.lib.data import GustoInterpDataset, GustoDataset, TRAINING_DATA, VALIDATION_DATA
from gusto.experiments.cnn_autoencoder.lib.autoencoder import AutoEncoder
from gusto.lib.utils import get_logger, get_model_checkpoint_callback
from gusto.experiments.cnn_autoencoder.lib.utils import LOGDIR, CHECKPOINT_DIR


torch.set_float32_matmul_precision('high')


class LITAutoEncoder(LITModel):
    '''Lightning model for AutoEncoder using the interpolation as training dataset'''

    def __init__(self, latent_size):
        super().__init__()
        self.model = AutoEncoder(latent_size)

    def forward(self, input_cpg, time_in, time_out):
        # pylint: disable=arguments-differ
        return self.model(input_cpg, time_in, time_out)


def get_dataset():
    '''Get the training and validation datasets'''
    train_dataset = GustoInterpDataset(TRAINING_DATA)
    val_dataset = GustoDataset(VALIDATION_DATA)
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True
    )
    return train_dataloader, val_dataloader


if __name__ == '__main__':
    test_name = Path(__file__).stem
    trainer = L.Trainer(
        max_epochs=400,
        logger=get_logger(test_name, LOGDIR),
        accelerator='gpu',
        devices=1,
        callbacks=[get_model_checkpoint_callback(test_name, CHECKPOINT_DIR)],
    )
    train_dataloaders, val_dataloaders = get_dataset()
    model = LITAutoEncoder(32)
    trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
