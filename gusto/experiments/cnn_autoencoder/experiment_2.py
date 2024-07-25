'''Second version of the CNN autoencoder using the GUSTO dataset,
using a Weighted Average of the timepoints
'''

from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from gusto.lib.data import TRAINING_DATA, VALIDATION_DATA, GustoDataset
from gusto.experiments.cnn_autoencoder.lib.torch_lightning import LITModel
from gusto.experiments.cnn_autoencoder.lib.autoencoder import AutoEncoderWeightedAverage
from gusto.experiments.cnn_autoencoder.lib.utils import get_logger
from gusto.experiments.cnn_autoencoder.lib.utils import get_model_checkpoint_callback

torch.set_float32_matmul_precision('high')


class LITAutoEncoder(LITModel):
    '''Lightning model for AutoEncoder using the GUSTO training dataset'''

    def __init__(self, latent_size):
        super().__init__()
        self.model = AutoEncoderWeightedAverage(latent_size, 2)

    def forward(self, input_cpg, time_in, time_out):
        # pylint: disable=arguments-differ
        return self.model(input_cpg, time_in, time_out)


if __name__ == '__main__':
    test_name = Path(__file__).stem
    trainer = L.Trainer(
        max_epochs=400,
        logger=get_logger(test_name),
        accelerator='gpu',
        devices=1,
        callbacks=[get_model_checkpoint_callback(test_name)],
    )
    train_dataset = GustoDataset(TRAINING_DATA)
    val_dataset = GustoDataset(VALIDATION_DATA)
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True
    )
    model = LITAutoEncoder(32)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
