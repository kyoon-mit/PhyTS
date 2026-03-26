import torch
import torch.nn as nn
from torch import optim
import lightning as L

from models.s4d import S4Model

class S4DMSELoss(L.LightningModule):
    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.lr_decay = lr_decay
        self.criterion = nn.MSELoss(reduction='mean')
        self.model = torch.compile(S4Model(
            d_input=d_input,
            d_output=d_output,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            dt_min=dt_min,
            dt_max=dt_max,
        ))

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        X, y_target, _ = batch
        X = X.transpose(2, 1)          # (B, n_ifos, L) -> (B, L, n_ifos)
        outputs = self(X)               # (B, d_output)
        loss = self.criterion(outputs, y_target)
        per_var_mse = nn.functional.mse_loss(outputs, y_target, reduction='none').mean(0)  # (d_output,)
        return loss, per_var_mse

    def training_step(self, batch, batch_idx):
        loss, per_var_mse = self._step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for i, v in enumerate(per_var_mse):
            self.log(f'train/mse/var_{i}', v, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, per_var_mse = self._step(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for i, v in enumerate(per_var_mse):
            self.log(f'val/mse/var_{i}', v, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, per_var_mse = self._step(batch)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for i, v in enumerate(per_var_mse):
            self.log(f'test/mse/var_{i}', v, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }
