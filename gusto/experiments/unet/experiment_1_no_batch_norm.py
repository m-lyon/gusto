'''Initial UNet experiment with 2D data.
This experiment uses no batch normalisation within the model.
'''

from pathlib import Path

import torch
import lightning as L

from gusto.lib.utils import get_logger_reruns, get_model_checkpoint_callback

from gusto.experiments.unet.lib.utils import LOGDIR, CHECKPOINT_DIR
from gusto.experiments.unet.experiment_1 import UNet, get_dataset


torch.set_float32_matmul_precision('high')


if __name__ == '__main__':
    test_name = Path(__file__).stem
    model = UNet(batch_norm=False)
    train_dataloaders, val_dataloaders = get_dataset()
    trainer = L.Trainer(
        max_epochs=50,
        logger=get_logger_reruns(test_name, LOGDIR),
        accelerator='gpu',
        devices=1,
        callbacks=[get_model_checkpoint_callback(test_name, CHECKPOINT_DIR)],
    )
    trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
