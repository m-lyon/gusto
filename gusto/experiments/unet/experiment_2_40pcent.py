'''Second UNet experiment with 2D data.
This experiment prunes the trained model and finetunes it.
'''

from pathlib import Path

import torch
from torch.nn.utils.prune import ln_structured
import lightning as L

from gusto.lib.utils import get_logger_reruns, get_model_checkpoint_callback
from gusto.lib.utils import get_best_checkpoint

from gusto.experiments.unet.lib.utils import LOGDIR, CHECKPOINT_DIR
from gusto.experiments.unet.experiment_1 import UNet, get_dataset


torch.set_float32_matmul_precision('high')


def prune_model(model: UNet) -> UNet:
    '''Prune the model.'''
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            ln_structured(module, name='weight', amount=0.4, n=1, dim=0)
    return model


if __name__ == '__main__':
    test_name = Path(__file__).stem
    best_model = get_best_checkpoint(CHECKPOINT_DIR.joinpath('experiment_1'))
    print(f'Loading model from {best_model}.')
    unpruned_model = UNet.load_from_checkpoint(best_model)  # pylint: disable=no-value-for-parameter
    pruned_model = prune_model(unpruned_model)
    train_dataloaders, val_dataloaders = get_dataset()
    trainer = L.Trainer(
        max_epochs=50,
        logger=get_logger_reruns(test_name, LOGDIR),
        accelerator='gpu',
        devices=1,
        callbacks=[get_model_checkpoint_callback(test_name, CHECKPOINT_DIR)],
    )
    trainer.fit(pruned_model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
