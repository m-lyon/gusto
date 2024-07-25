from functools import partial
import lightning as L

import torchmetrics
import torch
import torch.nn.functional as F

from lightning.pytorch.loggers import TensorBoardLogger


class LITModel(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.train_metric = self.metric_func()
        self.val_metric = self.metric_func()
        self.test_metric = self.metric_func()

    @property
    def metric_func(self):
        return partial(torchmetrics.MeanSquaredError, squared=False)

    @property
    def loss_func(self):
        return F.l1_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def training_step(self, train_batch, *args):
        inputs, outputs = train_batch
        out_pred = self(*inputs)

        self.train_metric.update(out_pred, outputs)
        loss = self.loss_func(outputs, out_pred)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, *args):
        inputs, outputs = val_batch
        out_pred = self(*inputs)
        loss = self.loss_func(outputs, out_pred)

        if not self.trainer.sanity_checking:
            self.val_metric.update(out_pred, outputs)
        return loss

    def predict_step(self, batch):
        inputs = tuple(batch)
        return self(*inputs)

    def test_step(self, batch, *args):
        inputs, dmri_out = batch
        dmri_out_inf = self(*inputs)

        self.test_metric.update(dmri_out, dmri_out_inf)
        return self.loss_func(dmri_out, dmri_out_inf)

    def on_train_epoch_end(self):
        self.logger: TensorBoardLogger
        self.logger.experiment.add_scalars(
            'epoch_loss', {'train': self.train_metric.compute()}, self.current_epoch
        )
        self.train_metric.reset()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.logger.experiment.add_scalars(
                'epoch_loss', {'validation': self.val_metric.compute()}, self.current_epoch
            )
            self.log('val_loss', self.val_metric.compute(), on_step=False)
            self.val_metric.reset()

    def on_test_epoch_end(self) -> None:
        print(f'Test loss: {self.test_metric.compute()}')
        self.test_metric.reset()
