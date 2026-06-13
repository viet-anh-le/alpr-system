from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def test_small_lpr_configs_support_dot_and_sep_token() -> None:
    from ocr.models.utils import encode

    label = "99-MDD3-000.46"
    config_paths = [
        ROOT / "configs" / "ocr" / "mobilevit.yaml",
        ROOT / "LPRNet" / "config" / "small_lpr_config.yaml",
    ]

    for path in config_paths:
        config = _load_yaml(path)
        chars = config["chars"]
        tokens = encode(label, chars)

        assert "." in chars
        assert "[SEP]" in chars
        assert len(tokens) + 2 <= config["max_seq_len"]


def test_small_lpr_collate_preserves_long_sep_labels() -> None:
    sys.path.insert(0, str(ROOT / "LPRNet"))
    from lprnet.trans_datamodule import collate_fn

    image = np.zeros((3, 48, 96), dtype=np.float32)
    label = list(range(15))

    _, labels, lengths = collate_fn([(image, label, len(label))])

    assert labels.shape == (1, 15)
    assert lengths == [15]
    assert torch.equal(labels[0], torch.tensor(label, dtype=torch.long))
