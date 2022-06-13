from omegaconf import DictConfig, OmegaConf
import hydra
import mlflow

import torch

from mf_writer import MlflowWriter
from model import extractor


def make_train_loader():
    return


def make_test_loader():
    return


def train_one_epoch():
    return


@hydra.main(config_name="config")
def main(cfg):
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # mlflowを使ったデータの管理
    experiment_name = cfg.train.experiment_name
    writer = MlflowWriter(experiment_name)
    writer.log_params_from_omegaconf_dict(cfg)

    # ネットワーク
    model = extractor()

    # 最適化手法

    # 損失関数

    # dataloader作成

    # training