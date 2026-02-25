import os
from dataclasses import dataclass, asdict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from dataset import DeepDoFDataModule
from lit import DeepDoFLit


@dataclass
class Config:
    seed: int = 42

    data_dir: str = "./multi-channel-edof_dataset"
    raw_imsize: int = 512
    batch_size: int = 1
    num_workers: int = 0

    in_channels: int = 4
    out_channels: int = 4
    n_phi: int = 5
    noise_sigma: float = 0.01

    optic_blackbox_path: str = "optic_module.pt"

    lr_digital: float = 1e-4
    weight_decay: float = 1e-3

    max_epochs: int = 50
    accelerator: str = "gpu"
    devices: int = 1

    checkpoint_dir: str = "checkpoints"
    every_n_epochs: int = 2


def build_argparser():
    import argparse
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default=Config.data_dir)
    p.add_argument("--optic_blackbox_path", type=str, default=Config.optic_blackbox_path)

    p.add_argument("--raw_imsize", type=int, default=Config.raw_imsize)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)

    p.add_argument("--in_channels", type=int, default=Config.in_channels)
    p.add_argument("--out_channels", type=int, default=Config.out_channels)
    p.add_argument("--n_phi", type=int, default=Config.n_phi)
    p.add_argument("--noise_sigma", type=float, default=Config.noise_sigma)

    p.add_argument("--lr_digital", type=float, default=Config.lr_digital)
    p.add_argument("--weight_decay", type=float, default=Config.weight_decay)

    p.add_argument("--max_epochs", type=int, default=Config.max_epochs)
    p.add_argument("--accelerator", type=str, default=Config.accelerator)
    p.add_argument("--devices", type=int, default=Config.devices)

    p.add_argument("--checkpoint_dir", type=str, default=Config.checkpoint_dir)
    p.add_argument("--every_n_epochs", type=int, default=Config.every_n_epochs)

    p.add_argument("--seed", type=int, default=Config.seed)

    return p


def main():
    args = build_argparser().parse_args()
    cfg = Config(**vars(args))

    pl.seed_everything(cfg.seed, workers=True)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    datamodule = DeepDoFDataModule(
        data_dir=cfg.data_dir,
        raw_imsize=cfg.raw_imsize,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    step_1 = {
        "lr_digital": cfg.lr_digital
    }

    model = DeepDoFLit(
        optic_blackbox_path=cfg.optic_blackbox_path,
        n_phi=cfg.n_phi,
        noise_sigma=cfg.noise_sigma,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        step_1=step_1,
    )

    logger = CSVLogger(save_dir="logs", name="train")

    ckpt = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        save_top_k=-1,
        every_n_epochs=cfg.every_n_epochs,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        logger=logger,
        callbacks=[ckpt],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()