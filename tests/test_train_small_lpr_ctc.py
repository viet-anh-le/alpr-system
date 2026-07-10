from __future__ import annotations

import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest


def _install_training_script_stubs() -> None:
    sys.modules.setdefault("torch", types.SimpleNamespace())
    sys.modules.setdefault("lightning", types.SimpleNamespace())

    callbacks = types.ModuleType("lightning.pytorch.callbacks")
    callbacks.EarlyStopping = object
    callbacks.LearningRateMonitor = object
    callbacks.ModelCheckpoint = object
    callbacks.RichProgressBar = object
    sys.modules.setdefault("lightning.pytorch.callbacks", callbacks)

    loggers = types.ModuleType("lightning.pytorch.loggers")
    loggers.CSVLogger = object
    sys.modules.setdefault("lightning.pytorch.loggers", loggers)

    lprnet = types.ModuleType("lprnet")
    datamodule = types.ModuleType("lprnet.small_lpr_ctc_datamodule")
    datamodule.SmallLPRCTCDataModule = object
    lightning_module = types.ModuleType("lprnet.small_lpr_ctc_lightning")
    lightning_module.SmallLPRCTCLightning = object
    sys.modules.setdefault("lprnet", lprnet)
    sys.modules.setdefault("lprnet.small_lpr_ctc_datamodule", datamodule)
    sys.modules.setdefault("lprnet.small_lpr_ctc_lightning", lightning_module)


_install_training_script_stubs()

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "train_small_lpr_ctc.py"
SPEC = importlib.util.spec_from_file_location("train_small_lpr_ctc", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_validate_checkpoint_mode_rejects_resume_and_finetune_together() -> None:
    cli = Namespace(resume="last.ckpt", finetune_from="best.ckpt")

    with pytest.raises(ValueError, match="--resume hoặc --finetune-from"):
        MODULE._validate_checkpoint_mode(cli)


@pytest.mark.unit
def test_resolve_checkpoint_returns_absolute_repo_path() -> None:
    resolved = MODULE._resolve_checkpoint("weights/ocr/small_lpr_ctc/best.ckpt")

    assert resolved == str(MODULE.ROOT / "weights/ocr/small_lpr_ctc/best.ckpt")


@pytest.mark.unit
def test_resolve_checkpoint_keeps_none() -> None:
    assert MODULE._resolve_checkpoint(None) is None
