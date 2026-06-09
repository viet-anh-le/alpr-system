import os
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

from lprnet.trans_lightning import TransLightningModule
from lprnet.trans_datamodule import DataModule

warnings.filterwarnings("ignore")


def train():
    # Load configuration
    config_path = "config/trans_vietnam_config.yaml"
    with open(config_path) as f:
        args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    # Create checkpoint directory with timestamp
    args.saving_ckpt = os.path.join(args.saving_ckpt, datetime.now().strftime("%Y-%m-%d_%H-%M"))
    os.makedirs(args.saving_ckpt, exist_ok=True)

    # Initialize Model and DataModule
    model = TransLightningModule(args)
    data_module = DataModule(args)

    # Setup Logger (optional, can be disabled if not needed)
    logger = WandbLogger(
        project="TransLPRNet-Vietnames", name=f"run_{datetime.now().strftime('%m%d_%H%M')}"
    )

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision="32",
        gradient_clip_val=args.gradient_clip_val,
        logger=logger,
        # overfit_batches=1,
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(
                dirpath=args.saving_ckpt,
                monitor="val_acc",
                mode="max",
                filename="trans_lprnet-{epoch:02d}-{val_acc:.3f}",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor="val_acc",
                mode="max",
                patience=50,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    # Start Training
    print(f"Starting training TransLPRNet...")
    print(f"Config: LR={args.lr}, Epochs={args.max_epochs}, ImageSize={args.img_size}")
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
