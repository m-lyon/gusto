'''Model for hyperparameter search using Lightning and Optuna'''

from pathlib import Path

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from lightning.pytorch import Trainer

from torch.utils.data import DataLoader
from gusto.lib.data import GustoDataset
from gusto.lib.torch_lightning import LITModel
from gusto.experiments.cnn_autoencoder.lib.autoencoder_hparams import AutoEncoder

EPOCHS = 50
PERCENT_VALID_EXAMPLES = 0.5


class LITAutoEncoder(LITModel):
    '''Lightning model for AutoEncoder using the GUSTO training dataset'''

    def __init__(self, **kwargs):
        super().__init__()
        self.model = AutoEncoder(**kwargs)

    def forward(self, input_cpg, time_in, time_out):
        # pylint: disable=arguments-differ
        return self.model(input_cpg, time_in, time_out)


def get_datasets():
    '''Get the training and validation datasets'''
    train_dataset = GustoDataset(
        Path.home().joinpath(
            'Dev', 'git', 'gusto', 'data', '3-9-48-72months_383CpGs_153indivs_train.pkl'
        )
    )
    val_dataset = GustoDataset(
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
    return train_dataloader, val_dataloader


if __name__ == "__main__":

    train, val = get_datasets()

    hyperparameters = {'latent_size': 8}

    model = LITAutoEncoder(**hyperparameters)

    trainer = Trainer(
        logger=False,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        accelerator='gpu',
        devices=1,
    )
    trainer.fit(model, train_dataloaders=train, val_dataloaders=val)
