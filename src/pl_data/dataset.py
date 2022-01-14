from typing import Dict, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

from src.common.utils import PROJECT_ROOT


class MyDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode, time : ValueNode, predict = False,**kwargs):
        super().__init__()
        self.path = str(PROJECT_ROOT / path)
        self.name = name
        self.time = time
        self.scaler= MinMaxScaler(feature_range=(0, 1))
        self.data = read_csv(self.path,usecols=["Open", "High", "Low", "Close", "Volume"]).values
        self.data = self.scaler.fit_transform(self.data)
        self.data = self.data.reshape(-1, 5)
        self.data = torch.from_numpy(self.data)
        self.data = self.data.float()
        

    def __len__(self) -> int:
        return self.data.shape[0] - self.time

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return {
            "x" : self.data[index:index+self.time, :],
            "y" : self.data[index+self.time, 3].reshape(1),
        } if not self.predict else self.data[index:index+self.time, :]

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=}, {self.time=})\n\tData: {self.data.shape=}"


class PredictDataset(Dataset):
    def __init__(self, path: str, time : int, **kwargs):
        super().__init__()
        self.path = str(PROJECT_ROOT / path)
        self.time = time
        self.scaler= MinMaxScaler(feature_range=(0, 1))
        self.data = read_csv(self.path,usecols=["Open", "High", "Low", "Close", "Volume"]).values
        self.data = self.scaler.fit_transform(self.data)
        self.data = self.data.reshape(-1, 5)
        self.data = torch.from_numpy(self.data)
        self.data = self.data.float()
        
    def __len__(self) -> int:
        return self.data.shape[0] - self.time

    def __getitem__(
        self, index
    ) -> torch.Tensor:
        return self.data[index:index+self.time, :]

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=}, {self.time=})\n\tData: {self.data.shape=}"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: MyDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train,
        time = cfg.data.datamodule.time,
        _recursive_=False
    )
    print(dataset)


if __name__ == "__main__":
    main()
