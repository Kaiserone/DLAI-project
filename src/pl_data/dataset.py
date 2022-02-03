from typing import Dict, Tuple, Union

import hydra
import omegaconf
import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset
import pandas as pd

from src.common.utils import PROJECT_ROOT

class MyDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode, time : ValueNode,**kwargs):
        super().__init__()
        self.name = name
        self.path = str(PROJECT_ROOT / path)
        self.time = time
        self.data : pd.DataFrame = pd.read_csv(self.path,usecols=["Open", "High", "Low", "Close", "Volume"])
        #self.data = self.data.rolling(window=10,min_periods=1).mean()
        self.data = self.data.pct_change() 
        self.data.dropna(inplace=True)
        min_return = self.data[['Open', 'High', 'Low', 'Close']].min()
        max_return = self.data[['Open', 'High', 'Low', 'Close']].max()

        print(f"{min_return=}, {max_return=}")
        self.data[['Open', 'High', 'Low', 'Close']] = (self.data[['Open','High', 'Low', 'Close']] - min_return) / (max_return - min_return)
        ###############################################################################
        '''Normalize volume column'''

        min_volume = self.data['Volume'].min()
        max_volume = self.data['Volume'].max()

        # Min-max normalize volume columns (0-1 range)
        self.data['Volume'] = (self.data['Volume'] - min_volume) / (max_volume - min_volume)
        
        self.data = self.data[["Open", "High", "Low", "Close", "Volume"]].values.reshape(-1, 5)
        self.data = torch.from_numpy(self.data)
        self.data = self.data.float()
        

    def __len__(self) -> int:
        return self.data.shape[0] - self.time

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return {
            "x" : self.data[index:index+self.time],
            "y" : self.data[index+self.time, 3].reshape(1),
        }

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=}, {self.time=})\n\tData: {self.data.shape=}"


class PredictDataset(MyDataset):
    def __init__(self, path: str, time : int, **kwargs):
        super().__init__(name="prediction",path=path, time=time, **kwargs)

    def __getitem__(
        self, index
    ) -> torch.Tensor:
        return self.data[index:index+self.time]


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
