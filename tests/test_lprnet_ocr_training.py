from __future__ import annotations

import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
LPRNET_ROOT = ROOT / "LPRNet"
if str(LPRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(LPRNET_ROOT))


def _install_train_script_stubs() -> None:
    sys.modules.setdefault("torch", types.SimpleNamespace())

    lightning = types.SimpleNamespace(seed_everything=lambda *args, **kwargs: None)
    sys.modules.setdefault("lightning", lightning)

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
    lprnet.__path__ = []
    sys.modules.setdefault("lprnet", lprnet)
    datamodule = types.ModuleType("lprnet.datamodule")
    datamodule.DataModule = object
    sys.modules.setdefault("lprnet.datamodule", datamodule)
    model_module = types.ModuleType("lprnet.lprnet")
    model_module.LPRNet = object
    sys.modules.setdefault("lprnet.lprnet", model_module)


def _load_train_script():
    _install_train_script_stubs()
    script_path = ROOT / "scripts" / "train_lprnet_ocr.py"
    spec = importlib.util.spec_from_file_location("train_lprnet_ocr", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_lprnet_collate_returns_integer_ctc_targets() -> None:
    from lprnet.datamodule import collate_fn

    image = torch.zeros((3, 50, 100), dtype=torch.float32).numpy()
    _, labels, lengths = collate_fn([(image, [1, 2], 2), (image, [3], 1)])

    assert labels.dtype == torch.long
    assert labels.tolist() == [1, 2, 3]
    assert lengths == [2, 1]


@pytest.mark.unit
def test_lprnet_dataset_strips_non_alnum_for_ocr_label_mode() -> None:
    from lprnet.datamodule import LPRNetDataset

    chars = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "A",
        "B",
        "C",
        "D",
        "E",
        "_",
    ]
    ids = {char: idx for idx, char in enumerate(chars)}
    dataset = object.__new__(LPRNetDataset)
    dataset.args = Namespace(chars=chars, label_mode="alnum")

    assert dataset._normalize_label_text("68E[SEP]009.58") == "68E00958"
    assert dataset._normalize_label_text("80A-026.51") == "80A02651"
    assert dataset.check([ids["6"], ids["8"], ids["E"], ids["0"], ids["0"], ids["9"]])
    assert not dataset.check([ids["_"]])


@pytest.mark.unit
def test_lprnet_ocr_config_targets_new_ocr_dataset() -> None:
    config_path = ROOT / "LPRNet" / "config" / "lprnet_ocr_config.yaml"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    assert config["train_dir"] == "data/datasets/ocr/train"
    assert config["valid_dir"] == "data/datasets/ocr/valid"
    assert config["img_size"] == (100, 50)
    assert config["t_length"] == 19
    assert config["label_mode"] == "alnum"
    assert config["chars"][-1] == "_"
    assert "." not in config["chars"]
    assert "[SEP]" not in config["chars"]


@pytest.mark.unit
def test_train_lprnet_ocr_load_config_overrides_data_root(tmp_path: Path) -> None:
    module = _load_train_script()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "train_dir": "unused/train",
                "valid_dir": "unused/valid",
                "test_dir": "unused/valid",
                "saving_ckpt": "unused",
                "img_size": (100, 50),
                "dropout_rate": 0.5,
                "weight_decay": 0.00002,
                "lr": 0.001,
                "batch_size": 32,
                "max_epochs": 50,
                "t_length": 19,
                "label_mode": "alnum",
                "gradient_clip_val": 1.0,
                "chars": ["0", "1", "_"],
            }
        ),
        encoding="utf-8",
    )
    data_root = tmp_path / "ocr"
    cli = Namespace(
        config=str(config_path),
        data_root=str(data_root),
        train_split="train",
        valid_split="valid",
        out_dir=str(tmp_path / "weights"),
        run_name="lprnet_smoke",
        epochs=1,
        batch_size=8,
        lr=0.0003,
    )

    args = module.load_config(cli)

    assert args.train_dir == str(data_root / "train")
    assert args.valid_dir == str(data_root / "valid")
    assert args.test_dir == str(data_root / "valid")
    assert args.saving_ckpt == str(tmp_path / "weights" / "lprnet_smoke")
    assert args.max_epochs == 1
    assert args.batch_size == 8
    assert args.lr == pytest.approx(0.0003)
