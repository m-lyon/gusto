'''TRMF model but with separate optimizers for encoder and decoder.'''

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer

from gusto.development.trmf import LITAutoRegressiveModel, get_logger, get_model_checkpoint_callback
from gusto.lib.data import GustoFactorisedDataset


class LITAutoRegressive2StepModel(LITAutoRegressiveModel):
    '''LightningModule for training the AutoRegressiveModel with 2-step training.'''

    def __init__(self, input_dim=383, latent_dim=64):
        super().__init__(input_dim=input_dim, latent_dim=latent_dim)
        self.automatic_optimization = False

    def training_step(self, train_batch, *args):
        enc_opt: torch.optim.Adam
        dec_opt: torch.optim.Adam
        enc_opt, dec_opt = self.optimizers()  # type: ignore # pylint: disable=unpacking-non-sequence
        inputs, outputs = train_batch
        out, pos, mask = outputs
        out_pred_enc = self.model(*inputs)  # encoder + time_encoder
        out_gt_enc = self.model.encoder(*outputs)  # encoder

        # Encoder + time encoder update
        loss_enc = self.loss_func(out_pred_enc, out_gt_enc)
        enc_opt.zero_grad()
        self.manual_backward(loss_enc)
        enc_opt.step()

        # Decoder update
        out_pred = self.model.decoder(out_pred_enc, pos, mask)
        loss_dec = self.loss_func(out_pred, out)
        dec_opt.zero_grad()
        self.manual_backward(loss_dec)
        dec_opt.step()

        self.log_dict({'train_loss_enc': loss_enc, 'train_loss_dec': loss_dec}, prog_bar=True)

    def configure_optimizers(self):
        optimizer_encoder = torch.optim.Adam(self.model.encoder.parameters())
        optimizer_decoder = torch.optim.Adam(self.model.decoder.parameters())
        return optimizer_encoder, optimizer_decoder

    def predict_step(self, batch):
        raise NotImplementedError


if __name__ == '__main__':
    tname = Path(__file__).stem
    latent_size = 64
    trainer = Trainer(
        max_epochs=300,
        logger=get_logger(tname, latent_size),
        accelerator='gpu',
        devices=1,
        callbacks=[get_model_checkpoint_callback(tname, latent_size)],
        log_every_n_steps=4,
    )
    train_dataset = GustoFactorisedDataset(
        Path.home().joinpath(
            'Dev', 'git', 'gusto', 'data', '3-9-48-72months_383CpGs_153indivs_train.pkl'
        )
    )
    val_dataset = GustoFactorisedDataset(
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
    model = LITAutoRegressiveModel(latent_dim=latent_size)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
