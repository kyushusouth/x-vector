
from omegaconf import DictConfig, OmegaConf
import hydra
import mlflow

import os
import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader

from mf_writer import MlflowWriter
from model import extractor
from dataset import x_vec_Dataset, x_vec_trans
from loss import GE2ELoss


def make_train_loader(cfg):
    trans = x_vec_trans(
        cfg=cfg
    )
    dataset = x_vec_Dataset(
        data_root=cfg.model.train_path,
        train=True,
        transform=trans,
        cfg=cfg
    )
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    return train_loader


def make_test_loader(cfg):
    trans = x_vec_trans(
        cfg=cfg
    )
    dataset = x_vec_Dataset(
        data_root=cfg.model.test_path,
        train=False,
        transform=trans,
        cfg=cfg
    )
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    return test_loader


def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    epoch_loss = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    for batch in train_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')

        mel, label = batch
        mel = mel.to(device)

        batch_size = mel.shape[0]
        data_cnt += batch_size

        # 出力
        x_vec, out = model(mel)
        breakpoint()

        loss = loss_fn(x_vec)

        # 最適化
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # 平均
    epoch_loss /= data_cnt
    return epoch_loss


@hydra.main(config_name="config", config_path="conf")
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
    model = extractor(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels
    )

    # 最適化手法
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, betas=(cfg.train.beta_1, cfg.train.beta_2)
    )

    # dataloader作成
    train_loader = make_train_loader(cfg)
    # test_loader = make_test_loader(cfg)

    # 損失関数
    loss_fn = GE2ELoss(device)

    # training
    with mlflow.start_run():
        model.train()
        for epoch in range(cfg.train.max_epoch):
            print(f"##### {epoch} #####")
            epoch_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            print(f"epoch_loss = {epoch_loss}")
            writer.log_metric("loss", epoch_loss)

    writer.log_torch_model(model)
    writer.log_torch_state_dict(model.state_dict())
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    writer.set_terminated()


if __name__ == "__main__":
    main()