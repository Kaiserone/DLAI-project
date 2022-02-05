from pathlib import Path
import pandas as pd
from pytorch_lightning.core import datamodule
from pytorch_lightning.trainer.trainer import Trainer

import streamlit as st
import torch
from torch.utils.data.dataloader import DataLoader
import wandb

from omegaconf import OmegaConf
from src.pl_data.dataset import PredictDataset
from src.pl_modules.model import MyModel, LSTM_std
import pytorch_lightning as pl
from src.ui.ui_utils import select_checkpoint, get_hydra_cfg
from sklearn.preprocessing import MinMaxScaler
import altair as alt
import sys
import inspect

@st.cache(allow_output_mutation=True)
def get_model(model_type : str, checkpoint_path: Path):
    m = getattr(sys.modules["src.pl_modules.model"], model_type)
    return m.load_from_checkpoint(checkpoint_path=str(checkpoint_path))


if wandb.api.api_key is None:
    st.error("You are not logged in on `Weights and Biases`: https://docs.wandb.ai/ref/cli/wandb-login")
    st.stop()

st.sidebar.subheader(f"Logged in W&B as: {wandb.api.viewer()['entity']}")

models_types = map(
    lambda classes: classes[0], 
    filter(
        lambda classes: classes[1].__module__ == "src.pl_modules.model",
        inspect.getmembers(sys.modules["src.pl_modules.model"], inspect.isclass)
    )
)

model_type = st.sidebar.selectbox(
    label="Select a Model",
    options=list(models_types),
)

checkpoint_path = select_checkpoint()
model = get_model(model_type,checkpoint_path=checkpoint_path)

#show a choise of datasets from folder /data
dataset_path = st.sidebar.selectbox(
    label="Select a dataset",
    options=list(Path("data").glob("./*.csv")),
)

time = model.time

dataset: PredictDataset = PredictDataset(path=dataset_path, time=time)

trainer = Trainer()#gpus=1'

pred = trainer.predict(model, dataloaders=DataLoader(dataset, shuffle=False, batch_size=32, num_workers=12), return_predictions=True)

pred = torch.cat(pred)
#pred = (pred.reshape(-1) * dataset.std) + dataset.mean
pred = pred.reshape(-1)
#truth = torch.tensor(dataset.data['Close']).reshape(-1)
whole = pd.DataFrame(
    dataset.data,
    index=pd.RangeIndex(start=0, stop=len(dataset.data), step=1),
    columns=["Open", "High", "Low", "Close"]#, "Volume"
)
whole["Pred"] = pd.DataFrame(pred,index=pd.RangeIndex(start=time, stop=len(dataset.data), step=1))
whole.dropna(inplace=True)

stock = alt.Chart(whole.reset_index()).transform_fold(fold=["Pred", "Close"], as_=["Stock", "price"]).mark_line().encode(
    x='index:Q',
    y='price:Q',
    color='Stock:N',
)

st.altair_chart(stock, use_container_width=True)
