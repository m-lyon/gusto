'''First experiment using simple model prediction with independent CpG sites'''

from pathlib import Path

from lightning.pytorch import Trainer

import torch
from torch.utils.data import DataLoader

from gusto.lib.layers import LinearLayer
from gusto.lib.torch_lightning import LITModel
from gusto.lib.utils import get_logger, get_model_checkpoint_callback
from gusto.lib.data import TRAINING_DATA, VALIDATION_DATA, GustoSingleCpGDataset
from gusto.experiments.independent_cpg.lib.utils import LOGDIR, CHECKPOINT_DIR


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


def get_dataset():
    '''Get the training and validation datasets'''
    train_dataset = GustoSingleCpGDataset(TRAINING_DATA)
    val_dataset = GustoSingleCpGDataset(VALIDATION_DATA)
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True
    )
    return train_dataloader, val_dataloader


if __name__ == '__main__':
    test_name = Path(__file__).stem
    trainer = Trainer(
        max_epochs=400,
        logger=get_logger(test_name, LOGDIR),
        accelerator='gpu',
        devices=1,
        callbacks=[get_model_checkpoint_callback(test_name, CHECKPOINT_DIR)],
    )
    train_dataloaders, val_dataloaders = get_dataset()
    model = LITAutoEncoder()
    trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
