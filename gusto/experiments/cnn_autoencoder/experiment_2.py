'''Second experiment with the CNN autoencoder and the GUSTO dataset,
    This experiment uses a weighted average of the timepoints.
'''

from pathlib import Path

import lightning as L
import torch

from gusto.lib.torch_lightning import LITModel
from gusto.experiments.cnn_autoencoder.lib.autoencoder import AutoEncoderWeightedAverage
from gusto.lib.utils import get_logger, get_model_checkpoint_callback
from gusto.experiments.cnn_autoencoder.experiment_1 import get_dataset
from gusto.experiments.cnn_autoencoder.lib.utils import LOGDIR, CHECKPOINT_DIR

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
        logger=get_logger(test_name, LOGDIR),
        accelerator='gpu',
        devices=1,
        callbacks=[get_model_checkpoint_callback(test_name, CHECKPOINT_DIR)],
    )
    train_dataloader, val_dataloader = get_dataset()
    model = LITAutoEncoder(32)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
