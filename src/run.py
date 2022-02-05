from pathlib import Path
from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
import torch
import pandas as pd
import wandb

from src.common.utils import log_hyperparameters, PROJECT_ROOT


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "tqdm_progress_bar" in cfg.logging:
        hydra.utils.log.info(f"Adding callback <TQDMProgressBar>")
        callbacks.append(
            TQDMProgressBar(
                refresh_rate=cfg.logging.tqdm_progress_bar.refresh_rate,
            )
        )

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info(f"Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info(f"Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info(f"Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info(f"Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            tags=cfg.core.tags,
        )
        hydra.utils.log.info(f"W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (Path(wandb_logger.experiment.dir) / "hparams.yaml").write_text(yaml_conf)

    hydra.utils.log.info(f"Instantiating the Trainer")

    # The Lightning core, the Trainer
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        val_check_interval=cfg.logging.val_check_interval,
        **cfg.train.pl_trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info(f"Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    hydra.utils.log.info(f"Starting testing!")
    trainer.test(datamodule=datamodule, ckpt_path='best')
        

    # Logger closing to release resources/avoid multi-run conflicts
    if "wandb" in cfg.logging:
        #(#Dataset, #Steps, #Batch)
        hydra.utils.log.info(f"Wandb Plotting Predictions for: {cfg.data.datamodule.time} days, {cfg.model.layers} layers, {cfg.model._target_} model, {cfg.model.time_vec} time_vec")
        wandb_logger.experiment.tags += [f"{cfg.data.datamodule.time} days", f"{cfg.model.layers} layers", f"{cfg.model._target_} model", f"{cfg.model.time_vec} time_vec"]
        predictions = trainer.predict(datamodule=datamodule, ckpt_path='best', return_predictions=True)
        for i, ds in enumerate(datamodule.pred_datasets):
            xs = []
            ys = []
            pred = torch.cat(predictions[i])
            pred = pred.reshape(-1).numpy()
            comparison = pd.DataFrame(
                ds.data,
                index=pd.RangeIndex(start=0, stop=len(ds.data), step=1),
                columns=["Open", "High", "Low", "Close"]#, "Volume"
            )
            xs.append(list(range(0, len(ds))))
            ys.append(list(comparison["Close"]))
            xs.append(list(range(cfg.data.datamodule.time, len(ds.data))))
            ys.append(list(pred))
            wandb.log({
                f"{ds.name}_pred": wandb.plot.line_series(xs,ys,title= f"{ds.name} Predictions", xname="Day", keys=["Close", "Pred"])
            })
        hydra.utils.log.info(f"Closing WandbLogger")
        wandb.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
