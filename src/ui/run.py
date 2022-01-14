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
from src.pl_modules.model import MyModel
import pytorch_lightning as pl
from src.ui.ui_utils import select_checkpoint, get_hydra_cfg
from sklearn.preprocessing import MinMaxScaler
import altair as alt

@st.cache(allow_output_mutation=True)
def get_model(checkpoint_path: Path):
    return MyModel.load_from_checkpoint(checkpoint_path=str(checkpoint_path))


if wandb.api.api_key is None:
    st.error("You are not logged in on `Weights and Biases`: https://docs.wandb.ai/ref/cli/wandb-login")
    st.stop()

st.sidebar.subheader(f"Logged in W&B as: {wandb.api.viewer()['entity']}")

checkpoint_path = select_checkpoint()
model: MyModel = get_model(checkpoint_path=checkpoint_path)

#show a choise of datasets from folder /data
dataset_path = st.sidebar.selectbox(
    label="Select a dataset",
    options=list(Path("data").glob("./*.csv")),
)
name, time = "aapl", 20
#time = st.sidebar.slider("Days", 0, 128, value=time)
dataset: PredictDataset = PredictDataset(path=dataset_path, time=time, predict=True)

trainer = Trainer()#gpus=1'

pred = trainer.predict(model, dataloaders=DataLoader(dataset, shuffle=False, batch_size=32, num_workers=12), return_predictions=True)

pred = torch.cat(pred)
pred = pred.reshape(-1,1)
truth = dataset.data[:,3].reshape(-1,1)

scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = dataset.scaler.min_[3], dataset.scaler.scale_[3]

df = pd.DataFrame({"pred": scaler.inverse_transform(pred).reshape(-1), "truth": scaler.inverse_transform(truth).reshape(-1)[time:]}, index=pd.RangeIndex(start=0, stop=len(pred), step=1))

stock = alt.Chart(df.reset_index()).transform_fold(fold=["pred", "truth"], as_=["Stock", "price"]).mark_line().encode(
    x='index:Q',
    y='price:Q',
    color='Stock:N'
)

st.altair_chart(stock, use_container_width=True)

#st.line_chart(scaler.inverse_transform(truth))
#st.line_chart(scaler.inverse_transform(pred))
#st.line_chart(scaler.inverse_transform(dataset.data[:,3]),scaler.inverse_transform(pred))
