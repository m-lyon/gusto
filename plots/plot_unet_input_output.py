from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import torch

from gusto.lib.utils import get_best_checkpoint
from gusto.experiments.unet.lib.dataset import UNetDataset

from gusto.experiments.unet.lib.utils import CHECKPOINT_DIR
from gusto.experiments.unet.experiment_1 import UNet


best_model = get_best_checkpoint(CHECKPOINT_DIR.joinpath('experiment_1'))
model = UNet.load_from_checkpoint(best_model)  # pylint: disable=no-value-for-parameter
model.to(device='cpu')
dataset = UNetDataset(start=0.9, end=1.0, dims=(256, 256), strategy='wrap')

example_input = dataset[0][0]
example_output = dataset[0][1]
pred_output = model(
    torch.Tensor(example_input[0][None, ...]), torch.Tensor(example_input[1][None, ...])
)
pred_output = pred_output.detach().numpy()

fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
axs[0].imshow(np.squeeze(example_output), cmap='viridis', vmin=0, vmax=1)
axs[0].set_title('Ground Truth')
im = axs[1].imshow(np.squeeze(pred_output), cmap='viridis', vmin=0, vmax=1)
axs[1].set_title('Predicted')
fig.colorbar(im, ax=axs[1], fraction=0.05, pad=0.04)

plt.savefig(Path(__file__).parent.parent.joinpath('docs', 'unet_pred.png'))
