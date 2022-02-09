# Stock Prediction via Transformers and Time Vectors 
<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/code-Lightning-blueviolet"></a>
    <a href="https://hydra.cc/"><img alt="Conf: hydra" src="https://img.shields.io/badge/conf-hydra-blue"></a>
    <a href="https://wandb.ai/site"><img alt="Logging: wandb" src="https://img.shields.io/badge/logging-wandb-yellow"></a>
    <a href="https://streamlit.io/"><img alt="UI: streamlit" src="https://img.shields.io/badge/ui-streamlit-orange"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## Here I include the additional images obtained from training/prediction.

---
Train and validation loss for the best performing models
<p float="left">
<img width="40%" src="images/train_loss_all.svg">
<img width="40%" src="images/val_loss_all.svg">
</p>

---
## My model 32 days input sequence with some of the best predictions with 1 layer
<p float="left">
<img width="40%" src="images/StockModel_Amazon_32d.svg">
<img width="40%" src="images/StockModel_Apple_32d.svg">
<img width="40%" src="images/StockModel_Google_32d.svg">
<img width="40%" src="images/StockModel_Tesla_32d.svg">
</p>

---
## My model 64 days input sequence with some of the best predictions with 2 layer
We can actually see that at 4 and 6 layers the predictions collapses to a constant value.
<p float="left">
<img width="40%" src="images/StockModel_Amazon_64d.svg">
<img width="40%" src="images/StockModel_Apple_64d.svg">
<img width="40%" src="images/StockModel_Google_64d.svg">
<img width="40%" src="images/StockModel_Tesla_64d.svg">
</p>

---
## My model 128 days input sequence.
We can actually see that at 4 and 6 layers the predictions collapses to a constant value.
<p float="left">
<img width="40%" src="images/StockModel_Amazon_128d.svg">
<img width="40%" src="images/StockModel_Apple_128d.svg">
<img width="40%" src="images/StockModel_Google_128d.svg">
<img width="40%" src="images/StockModel_Tesla_128d.svg">
</p>

---
## Naive LSTM 32 days input sequence
<p float="left">
<img width="40%" src="images/NaiveLSTM_Amazon_32d.svg">
<img width="40%" src="images/NaiveLSTM_Apple_32d.svg">
<img width="40%" src="images/NaiveLSTM_Google_32d.svg">
<img width="40%" src="images/NaiveLSTM_Tesla_32d.svg">
</p>

---
## Naive LSTM 32 days input sequence
<p float="left">
<img width="40%" src="images/NaiveLSTM_Amazon_64d.svg">
<img width="40%" src="images/NaiveLSTM_Apple_64d.svg">
<img width="40%" src="images/NaiveLSTM_Google_64d.svg">
<img width="40%" src="images/NaiveLSTM_Tesla_64d.svg">
</p>
