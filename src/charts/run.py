from typing import List, Tuple
import pandas as pd
import altair as alt
import glob
import itertools as it
import hydra
from src.common.utils import log_hyperparameters, PROJECT_ROOT
from hydra.core.hydra_config import HydraConfig
import omegaconf
from omegaconf import DictConfig, OmegaConf
#Amazon 64d 1/2l


def run(cfg: DictConfig):
    for m,ds,day in it.product(cfg.charts.models, cfg.charts.dataset, cfg.charts.days):
        file = f"{m}_{ds}_{day}d_*.csv"
        whole_df = pd.DataFrame()
        layers = []
        files = glob.glob(f"{PROJECT_ROOT}/images/{file}")
        for r in files:
            print(f"Reading {r}")
            f = r.split("/")[-1]
            (_, _, d, l, t) = f.split(".")[0].split("_")
            if not cfg.charts.no_time and t == "no-time":
                continue
            if l not in [f"{cl}l" for cl in cfg.charts.layers]:
                continue
            pred_col = f"{d}-{l}-{t}"
            layers += [(l,t)]
            df = pd.read_csv(r, usecols=["Close", "Original", pred_col])
            whole_df[["Close", "Original", f"{l}-{t}"]] = df[["Close", "Original", pred_col]]
            diff = whole_df[f"{l}-{t}"] - whole_df["Close"]
            diff = diff.abs().mean()
            print(f"\t{l} diff: {diff}")

        if len(files) > 0:
            stock = alt.Chart(whole_df.rename_axis('day').reset_index()).transform_fold(fold=["Close"] + sort_layers(cfg, layers), as_=["Stock", "price"]).mark_line().encode(
                x='day:Q',
                y='price:Q',
                color='Stock:N',
            )
            print(f"Saving images/{m}_{ds}_{day}d.svg")
            stock.save(PROJECT_ROOT/f"images/{m}_{ds}_{day}d.svg")

def sort_layers(cfg: DictConfig, found: List[Tuple[str,str]]) -> List[str]: 
    tp = sorted(found, key = lambda x : x[1], reverse=not cfg.charts.time_priority)
    f = sorted(tp, key = lambda x : x[0], reverse = not cfg.charts.descending)
    return [f"{s[0]}-{s[1]}" for s in f] 

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()
