import os
import sys
import yaml
import warnings
from argparse import Namespace
from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

import torch

from lprnet.small_lpr_lightning import SmallLPRLightning
from lprnet.small_lpr_datamodule import SmallLPRDataModule

warnings.filterwarnings("ignore")

RESUME_CKPT = None  # đặt path checkpoint ở đây để resume, hoặc truyền qua CLI
torch.serialization.add_safe_globals([Namespace])


def train(resume_ckpt: str = None):
    config_path = "config/small_lpr_config.yaml"
    with open(config_path) as f:
        args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    if resume_ckpt:
        # Tiếp tục lưu checkpoint vào cùng thư mục với checkpoint cũ
        args.saving_ckpt = os.path.dirname(resume_ckpt)
        print(f"Resuming from: {resume_ckpt}")
        print(f"Checkpoints will continue in: {args.saving_ckpt}")
    else:
        args.saving_ckpt = os.path.join(args.saving_ckpt, datetime.now().strftime("%Y-%m-%d_%H-%M"))
        os.makedirs(args.saving_ckpt, exist_ok=True)

    model = SmallLPRLightning(args)
    data_module = SmallLPRDataModule(args)

    logger = WandbLogger(
        project="SmallLPR-Vietnamese", name=f"run_{datetime.now().strftime('%m%d_%H%M')}"
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision="32",
        gradient_clip_val=args.gradient_clip_val,
        logger=logger,
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(
                dirpath=args.saving_ckpt,
                monitor="val_acc",
                mode="max",
                filename="small_lpr-{epoch:02d}-{val_acc:.3f}",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(monitor="val_acc", mode="max", patience=50, verbose=True),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    print(f"Starting training SmallLPR...")
    print(f"Config: LR={args.lr}, Epochs={args.max_epochs}, ImageSize={args.img_size}")
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else RESUME_CKPT
    train(resume_ckpt=ckpt)
