import random
from typing import Optional, Sequence

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ValueNode 
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split, Subset

from src.common.utils import PROJECT_ROOT


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        days: ValueNode,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.days = days

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets : Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None
        self.pred_datasets: Optional[Sequence[Dataset]] = None
        self.train_subset : Optional[Sequence[int]] = None
        self.val_subset : Optional[Sequence[int]] = None

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train, days = self.days)
            train_size = int(0.85 * len(self.train_dataset))
            self.train_subset, self.val_subset = range(0,train_size), range(train_size,len(self.train_dataset))
        
        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg, days = self.days)
                for dataset_cfg in self.datasets.test
            ]

        if stage is None or stage == "predict":
            self.pred_datasets = [
                hydra.utils.instantiate(dataset_cfg, days = self.days)
                for dataset_cfg in self.datasets.predict
            ]
            
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            Subset(self.train_dataset, self.train_subset),
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return DataLoader(
            Subset(self.train_dataset, self.val_subset),
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.test_datasets
        ]
    
    def predict_dataloader(self) -> Sequence[DataLoader]:
        return  [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.pred,
                num_workers=self.num_workers.pred,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.pred_datasets
        ]
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    print(datamodule)
    datamodule.setup()



if __name__ == "__main__":
    main()
