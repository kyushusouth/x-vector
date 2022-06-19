
from omegaconf import DictConfig, OmegaConf
import hydra
import mlflow
from datetime import datetime
import wandb

import os
from tqdm import tqdm
import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from mf_writer import MlflowWriter
from model import extractor
from dataset import x_vec_Dataset, x_vec_trans
# from loss import GE2ELoss
from loss_another import GE2ELoss


# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def make_train_val_loader(cfg):
    trans = x_vec_trans(
        cfg=cfg
    )
    dataset = x_vec_Dataset(
        data_root=cfg.train.train_path,
        train=True,
        transform=trans,
        cfg=cfg
    )

    # 学習用と検証用にデータセットを分割
    n_samples = len(dataset)
    train_size = int(n_samples * 0.95)
    val_size = n_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_dataset.train = False

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    return train_loader, val_loader


def make_test_loader(cfg):
    trans = x_vec_trans(
        cfg=cfg
    )
    dataset = x_vec_Dataset(
        data_root=cfg.train.test_path,
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


def train_one_epoch(model, train_loader, optimizer, loss_f_soft, loss_f_cont, device, cfg, writer):
    epoch_loss = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    for batch in tqdm(train_loader):
        iter_cnt += 1

        mel, speaker_label = batch
        mel = mel.to(device)

        batch_size = mel.shape[0]
        data_cnt += batch_size

        # 出力
        x_vec, out = model(mel)

        # 損失
        loss = loss_f_soft(x_vec) + loss_f_cont(x_vec)
        loss.backward()

        # gradient clipping
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)

        optimizer.step()

        epoch_loss += loss.item()

        wandb.log({"train_iter_loss": loss.item()})
        # writer.log_metric("train_iter_loss", loss.item())
        

        if cfg.train.debug:
            if iter_cnt == 10:
                break

    # 平均
    epoch_loss /= iter_cnt
    return epoch_loss


def calc_val_loss(model, val_loader, loss_f_soft, loss_f_cont, device, cfg, writer):
    val_loss = 0
    data_cnt = 0
    iter_cnt = 0
    print("--- validation ---")
    for batch in tqdm(val_loader):
        iter_cnt += 1

        mel, speaker_label = batch
        mel = mel.to(device)

        batch_size = mel.shape[0]
        data_cnt += batch_size

        with torch.no_grad():
            x_vec, out = model(mel)
        
        loss = loss_f_soft(x_vec) + loss_f_cont(x_vec)

        val_loss += loss.item()

        wandb.log({"val_iter_loss": loss.item()})
        # writer.log_metric("val_iter_loss", loss.item())

    val_loss /= iter_cnt
    return val_loss


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')

    # wandbでhydraを使うための設定
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )

    if cfg.train.debug:
        print('--- debug ---')
        cfg.train.max_epoch = 5
        cfg.train.batch_size = 5
        cfg.train.n_utterance = 2

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # mlflowを使ったデータの管理
    experiment_name = cfg.train.experiment_name
    writer = MlflowWriter(experiment_name)
    writer.log_params_from_omegaconf_dict(cfg)

    # パラメータの保存先
    save_path = os.path.join(cfg.train.train_save_path, current_time)
    os.makedirs(save_path, exist_ok=True)

    # dataloader作成
    train_loader, val_loader = make_train_val_loader(cfg)
    # test_loader = make_test_loader(cfg)

    # 損失関数
    loss_f_soft = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='softmax') # for softmax loss
    loss_f_cont = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='contrast') # for contrast loss

    # training
    with wandb.init(**cfg.wandb.setup, config=wandb_cfg) as run:
        # ネットワーク
        model = extractor(
            in_channels=cfg.model.in_channels,
            n_dim=cfg.model.n_dim,
            out_channels=cfg.model.out_channels,
        ).to(device)

        # 最適化手法
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay
        )

        # schedular
        schedular = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=cfg.train.max_epoch // 2,
            gamma=cfg.train.lr_decay_rate
        )

        wandb.watch(model, **cfg.wandb.watch)

        model.train()
        for epoch in range(cfg.train.max_epoch):
            print(f"##### {epoch} #####")
            epoch_loss = train_one_epoch(model, train_loader, optimizer, loss_f_soft, loss_f_cont, device, cfg, writer)
            print(f"epoch_loss = {epoch_loss}")
            # writer.log_metric("epoch_loss", epoch_loss)

            if epoch % cfg.train.val_step == 0:
                val_loss = calc_val_loss(model, val_loader, loss_f_soft, loss_f_cont, device, cfg, writer)
                print(f"val_loss = {val_loss}")
                # writer.log_metric("val_loss", val_loss)
            
            schedular.step()
        
        # モデルパラメータの保存
        torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(os.path.join(save_path, "model.pth"))
        wandb.log_artifact(artifact)
        
    wandb.finish()

    
    # with mlflow.start_run():
    #     model.train()
    #     for epoch in range(cfg.train.max_epoch):
    #         print(f"##### {epoch} #####")
    #         epoch_loss = train_one_epoch(model, train_loader, optimizer, loss_f_soft, loss_f_cont, device, cfg, writer)
    #         print(f"epoch_loss = {epoch_loss}")
    #         writer.log_metric("epoch_loss", epoch_loss)

    #         if epoch % cfg.train.val_step == 0:
    #             val_loss = calc_val_loss(model, val_loader, loss_f_soft, loss_f_cont, device, cfg, writer)
    #             print(f"val_loss = {val_loss}")
    #             writer.log_metric("val_loss", val_loss)
            
    #         schedular.step()

    # writer.log_torch_model(model)
    # writer.log_torch_state_dict(model.state_dict())
    # writer.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    # writer.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    # writer.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    # writer.set_terminated()


@hydra.main(config_name="config", config_path="conf")
def test(cfg):
    wandb.config = OmegaConf.to_container(
        cfg, 
        resolve=True, 
        throw_on_missing=True,
    )
    wandb.init(
        entity=cfg.wandb.entity, 
        project=cfg.wandb.setup.project,
        name="test_multirun",
    )
    wandb.finish()


if __name__ == "__main__":
    main()
    # test()