from typing import Dict, Tuple, Union

import hydra
import omegaconf
import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset
import pandas as pd
from src.common.utils import PROJECT_ROOT

class MyDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode, days : ValueNode,**kwargs):
        super(MyDataset,self).__init__()
        self.name = name
        self.path = str(PROJECT_ROOT / path)
        self.days = days

        # Read csv file
        self.data : pd.DataFrame = pd.read_csv(self.path, parse_dates=['Date'], index_col="Date",usecols=["Date","Open", "High", "Low", "Close"]).sort_index()
        
        # Mean rolling with window 10
        self.data = self.data.rolling(window=10,min_periods=1).mean()
        self.data["Original"] = self.data["Close"]
        
        # Percentage change
        self.data[['Open', 'High', 'Low', 'Close']] = self.data[['Open', 'High', 'Low', 'Close']].pct_change()
        
        # Drop first row
        self.data.dropna(inplace=True)
        
        #Normalization
        min_return = min(self.data[['Open', 'High', 'Low', 'Close']].min())
        max_return = max(self.data[['Open', 'High', 'Low', 'Close']].max())
        self.min_pct = min_return
        self.max_pct = max_return
        self.data[['Open', 'High', 'Low', 'Close']] = (self.data[['Open','High', 'Low', 'Close']] - min_return) / (max_return - min_return)

        #to Tensor
        self.data = self.data[["Open", "High", "Low", "Close", "Original"]].values
        self.data = torch.from_numpy(self.data)
        self.data = self.data.float()

        

    def __len__(self) -> int:
        return self.data.shape[0] - self.days

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return {
            #sequence of length days with 4 features
            "x" : self.data[index:index+self.days, :4],
            "y" : self.data[index+self.days, 3].reshape(1),
         }

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=}, {self.days=})\n\tData: {self.data.shape=}"


class PredictDataset(MyDataset):
    def __init__(self,name:ValueNode, path: ValueNode, days : ValueNode, **kwargs):
        super(PredictDataset,self).__init__(name=name,path=path, days=days, **kwargs)

    def __getitem__(
        self, index
    ) -> torch.Tensor:
        return self.data[index:index+self.days, :4]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: MyDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train,
        days = cfg.data.datamodule.days,
        _recursive_=False
    )
    print(dataset)


if __name__ == "__main__":
    main()
