from typing import Any, Dict, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch.nn import functional as F
import torch.nn as nn
from src.common.utils import PROJECT_ROOT
from src.pl_modules.transf import Transformer
from src.pl_modules.time import Time2Vector

class MyModel(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters() # populate self.hparams with args and kwargs automagically!
        
        time = self.hparams.data.datamodule.time
        n_feature = self.hparams.n_feature # 5
        
        if self.hparams.time_vec:
            self.time_vec = Time2Vector(time, n_feature)
            n_feature += 2
            
        self.transformer = Transformer(self.hparams.nhead, self.hparams.layers, time, n_feature)

        self.fc1 = nn.Linear(time * (n_feature), (n_feature))
        self.drop = nn.Dropout(p=0.15)
        self.fc2 = nn.Linear(n_feature, 1)

    def forward(self, seq: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """

        if self.hparams.time_vec:
            x = self.time_vec(seq)
            x = torch.concat([seq, x], dim=-1)
        x = self.transformer(x)
        x = x.view(x.shape[0], -1)
        #x = self.drop(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch['x'], batch['y']
        x = self(x)
        loss = F.mse_loss(x, y)
        return {
                "logits": x,
                "loss": loss,
            }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step = self.step(batch, batch_idx)

        self.log_dict(
            {
                "train_loss": step["loss"], 
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return step["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step = self.step(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": step["loss"], 
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return step["loss"]

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step = self.step(batch, batch_idx)
        self.log_dict(
            {
                "test_loss": step["loss"], 
            },
        )
        return step["loss"]

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]

class LSTM_std(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters() # populate self.hparams with args and kwargs automagically!
        
        time = self.hparams.data.datamodule.time

        self.n_feature = self.hparams.n_feature
        self.num_layers = self.hparams.num_layers
        if self.hparams.time_vec:
            self.time_vec = Time2Vector(time, self.n_feature)
            self.n_feature += 2
        
        self.lstm = nn.LSTM(input_size=self.n_feature,num_layers= self.num_layers, batch_first=True, dropout=0.2, hidden_size=self.n_feature)
        
        self.fc1 = nn.Linear(self.n_feature * time , self.n_feature)
        self.fc2 = nn.Linear(self.n_feature, 1)
        self.drop = nn.Dropout(p=0.15)


    def forward(self, seq: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """

        if self.hparams.time_vec:
            t = self.time_vec(seq)
            seq = torch.concat([seq, t], dim=-1)

        # Propagate input through LSTM
        out, (_, _) = self.lstm(seq) #lstm with input, hidden, and internal state
        #print(f"out.shape: {out.shape}")
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out) #first Dense
        out = F.relu(out)
        #out = self.drop(out)
        out = self.fc2(out) #second Dense
        return out

    def step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch['x'], batch['y']
        x = self(x)
        loss = F.mse_loss(x, y)
        return {
                #"logits": x,
                "loss": loss,
            }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step = self.step(batch, batch_idx)

        self.log_dict(
            {
                "train_loss": step["loss"], 
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return step["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step = self.step(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": step["loss"], 
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return step["loss"]

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step = self.step(batch, batch_idx)
        self.log_dict(
            {
                "test_loss": step["loss"], 
            },
        )
        return step["loss"]

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
