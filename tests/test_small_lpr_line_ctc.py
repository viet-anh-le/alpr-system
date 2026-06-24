from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LPRNet"))


@pytest.fixture(autouse=True)
def _use_real_lprnet_package() -> None:
    module = sys.modules.get("lprnet")
    if module is not None and not hasattr(module, "__path__"):
        for name in list(sys.modules):
            if name == "lprnet" or name.startswith("lprnet."):
                del sys.modules[name]
    lightning_module = sys.modules.get("lightning")
    if lightning_module is not None and not hasattr(lightning_module, "LightningModule"):
        for name in list(sys.modules):
            if name == "lightning" or name.startswith("lightning."):
                del sys.modules[name]
    if str(ROOT / "LPRNet") not in sys.path:
        sys.path.insert(0, str(ROOT / "LPRNet"))


CHARS = [
    "<blank>",
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
    "F",
    "G",
    "H",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "X",
    "Y",
    "Z",
    "Đ",
    "-",
    ".",
    "[SEP]",
    "_",
]


def _ctc_logits(sequence: list[int], vocab_size: int) -> torch.Tensor:
    logits = torch.full((1, len(sequence), vocab_size), -10.0)
    for timestep, token_id in enumerate(sequence):
        logits[0, timestep, token_id] = 10.0
    return logits


@pytest.mark.unit
def test_parse_line_ctc_label_treats_ddl_dash_dot_as_one_line() -> None:
    from lprnet.small_lpr_line_ctc_datamodule import (
        LAYOUT_ONE_LINE,
        parse_line_ctc_label,
    )

    parsed = parse_line_ctc_label("51G-123.45", CHARS)

    assert parsed.layout_label == LAYOUT_ONE_LINE
    assert parsed.layout_loss_mask is True
    assert parsed.one_line_loss_mask is True
    assert parsed.top_loss_mask is False
    assert parsed.bottom_loss_mask is False
    assert parsed.one_line_text == "51G-123.45"
    assert parsed.top_text == ""
    assert parsed.bottom_text == ""
    assert parsed.global_text == "51G-123.45"
    assert parsed.global_label
    assert parsed.one_line_label
    assert parsed.top_label == []
    assert parsed.bottom_label == []
    assert parsed.is_ambiguous_layout is False


@pytest.mark.unit
def test_parse_line_ctc_label_splits_two_line_sep_label() -> None:
    from lprnet.small_lpr_line_ctc_datamodule import (
        LAYOUT_TWO_LINE,
        parse_line_ctc_label,
    )

    parsed = parse_line_ctc_label("51G[SEP]123.45", CHARS)

    assert parsed.layout_label == LAYOUT_TWO_LINE
    assert parsed.layout_loss_mask is True
    assert parsed.one_line_loss_mask is False
    assert parsed.top_loss_mask is True
    assert parsed.bottom_loss_mask is True
    assert parsed.one_line_text == ""
    assert parsed.top_text == "51G"
    assert parsed.bottom_text == "123.45"
    assert parsed.global_text == "51G[SEP]123.45"


@pytest.mark.unit
def test_parse_line_ctc_label_keeps_other_no_sep_labels_as_one_line() -> None:
    from lprnet.small_lpr_line_ctc_datamodule import (
        LAYOUT_ONE_LINE,
        parse_line_ctc_label,
    )

    parsed = parse_line_ctc_label("60LD-4300", CHARS)

    assert parsed.layout_label == LAYOUT_ONE_LINE
    assert parsed.layout_loss_mask is True
    assert parsed.one_line_loss_mask is True
    assert parsed.top_loss_mask is False
    assert parsed.bottom_loss_mask is False
    assert parsed.one_line_text == "60LD-4300"
    assert parsed.top_text == ""
    assert parsed.bottom_text == ""


@pytest.mark.unit
def test_collate_line_ctc_trains_ddl_dash_dot_as_one_line() -> None:
    from lprnet.small_lpr_line_ctc_datamodule import (
        LAYOUT_ONE_LINE,
        collate_fn_line_ctc,
        parse_line_ctc_label,
    )

    image = np.zeros((3, 48, 96), dtype=np.float32)
    batch = [
        (image, parse_line_ctc_label("51G-123.45", CHARS)),
        (image, parse_line_ctc_label("51G[SEP]123.45", CHARS)),
    ]

    collated = collate_fn_line_ctc(batch)

    assert tuple(collated["images"].shape) == (2, 3, 48, 96)
    assert collated["layout_labels"].tolist()[0] == LAYOUT_ONE_LINE
    assert collated["layout_loss_mask"].tolist() == [True, True]
    assert collated["one_line_loss_mask"].tolist() == [True, False]
    assert collated["top_loss_mask"].tolist() == [False, True]
    assert collated["bottom_loss_mask"].tolist() == [False, True]
    assert collated["global_lengths"].tolist()[0] > 0
    assert collated["one_line_lengths"].tolist()[0] > 0
    assert collated["top_lengths"].tolist()[0] == 0
    assert collated["bottom_lengths"].tolist()[0] == 0
    assert collated["is_ambiguous_layout"].tolist() == [False, False]


@pytest.mark.unit
def test_line_ctc_dataset_excludes_reviewed_bad_paths(tmp_path: Path) -> None:
    from lprnet.small_lpr_line_ctc_datamodule import _load_excluded_paths

    dataset_root = tmp_path / "ocr"
    excluded_file = dataset_root / "valid" / "bad.jpg"
    absolute_file = dataset_root / "train" / "bad_abs.jpg"
    exclude_path = tmp_path / "exclude_paths.txt"
    exclude_path.write_text(
        "\n".join(
            [
                "# review exclusions",
                "valid/bad.jpg",
                str(absolute_file),
                "",
            ]
        ),
        encoding="utf-8",
    )

    excluded = _load_excluded_paths(exclude_path, dataset_root=dataset_root)

    assert excluded_file.resolve() in excluded
    assert absolute_file.resolve() in excluded


@pytest.mark.unit
def test_masked_layout_cross_entropy_ignores_explicit_ignore_index() -> None:
    from lprnet.small_lpr_line_ctc_datamodule import LAYOUT_IGNORE_INDEX
    from lprnet.small_lpr_line_ctc_lightning import masked_layout_cross_entropy

    logits = torch.tensor([[0.0, 5.0], [5.0, 0.0], [0.0, 5.0]])
    labels = torch.tensor([LAYOUT_IGNORE_INDEX, 0, LAYOUT_IGNORE_INDEX])

    loss = masked_layout_cross_entropy(logits, labels)

    assert loss == pytest.approx(F.cross_entropy(logits[1:2], labels[1:2]))


@pytest.mark.unit
def test_lightning_losses_include_one_line_ctc_branch() -> None:
    from lprnet.small_lpr_line_ctc_lightning import SmallLPRLineCTCLightning

    args = SimpleNamespace(
        chars=CHARS,
        d_model=16,
        backbone_ch=16,
        line_prior_strength=1.0,
        global_loss_weight=7.0,
        one_line_loss_weight=2.0,
        top_loss_weight=3.0,
        bottom_loss_weight=5.0,
        layout_loss_weight=11.0,
        two_line_threshold=0.5,
        lr=1e-3,
        weight_decay=1e-4,
        scheduler="constant",
        max_epochs=1,
    )
    module = SmallLPRLineCTCLightning(args)
    vocab_size = len(CHARS)
    sep_id = CHARS.index("[SEP]")
    outputs = {
        "global_logits": torch.randn((2, 6, vocab_size), dtype=torch.float32),
        "one_line_logits": torch.randn((2, 4, vocab_size), dtype=torch.float32),
        "top_logits": torch.randn((2, 4, vocab_size), dtype=torch.float32),
        "bottom_logits": torch.randn((2, 4, vocab_size), dtype=torch.float32),
        "layout_logits": torch.randn((2, 2), dtype=torch.float32),
    }
    batch = {
        "global_targets": torch.tensor([1, 2, 3, 1, sep_id, 2], dtype=torch.long),
        "global_input_lengths": torch.tensor([6, 6], dtype=torch.long),
        "global_lengths": torch.tensor([3, 3], dtype=torch.long),
        "one_line_targets": torch.tensor([1, 2, 3], dtype=torch.long),
        "one_line_lengths": torch.tensor([3, 0], dtype=torch.long),
        "one_line_loss_mask": torch.tensor([True, False], dtype=torch.bool),
        "top_targets": torch.tensor([1], dtype=torch.long),
        "top_lengths": torch.tensor([0, 1], dtype=torch.long),
        "top_loss_mask": torch.tensor([False, True], dtype=torch.bool),
        "bottom_targets": torch.tensor([2], dtype=torch.long),
        "bottom_lengths": torch.tensor([0, 1], dtype=torch.long),
        "bottom_loss_mask": torch.tensor([False, True], dtype=torch.bool),
        "layout_labels": torch.tensor([0, 1], dtype=torch.long),
    }

    losses = module._losses(outputs, batch)

    expected = (
        args.global_loss_weight * losses["global_ctc_loss"]
        + args.one_line_loss_weight * losses["one_line_ctc_loss"]
        + args.top_loss_weight * losses["top_ctc_loss"]
        + args.bottom_loss_weight * losses["bottom_ctc_loss"]
        + args.layout_loss_weight * losses["layout_loss"]
    )
    assert losses["one_line_ctc_loss"].isfinite()
    assert torch.allclose(losses["loss"], expected)


@pytest.mark.unit
def test_lightning_decode_mode_can_use_global_branch() -> None:
    from lprnet.small_lpr_line_ctc_lightning import SmallLPRLineCTCLightning

    ids = {char: idx for idx, char in enumerate(CHARS)}
    args = SimpleNamespace(
        chars=CHARS,
        d_model=16,
        backbone_ch=16,
        line_prior_strength=1.0,
        global_loss_weight=1.0,
        one_line_loss_weight=1.0,
        top_loss_weight=1.0,
        bottom_loss_weight=1.0,
        layout_loss_weight=0.2,
        two_line_threshold=0.5,
        decode_mode="global",
        lr=1e-3,
        weight_decay=1e-4,
        scheduler="constant",
        max_epochs=1,
    )
    module = SmallLPRLineCTCLightning(args)
    outputs = {
        "global_logits": _ctc_logits([ids["5"], ids["1"], ids["G"]], len(CHARS)),
        "one_line_logits": _ctc_logits([ids["6"], ids["0"], ids["L"]], len(CHARS)),
        "top_logits": _ctc_logits([ids["6"], ids["0"], ids["L"]], len(CHARS)),
        "bottom_logits": _ctc_logits([ids["1"], ids["2"], ids["3"]], len(CHARS)),
        "layout_logits": torch.tensor([[5.0, 0.0]]),
    }
    batch = {"texts": ["51G"]}

    assert float(module._line_accuracy(outputs, batch)) == pytest.approx(1.0)

    module.decode_mode = "layout"

    assert float(module._line_accuracy(outputs, batch)) == pytest.approx(0.0)


@pytest.mark.unit
def test_line_ctc_decode_uses_one_line_branch_and_top_bottom_for_two_line() -> None:
    from lprnet.small_lpr_line_ctc import line_ctc_greedy_decode

    ids = {char: idx for idx, char in enumerate(CHARS)}
    outputs = {
        "global_logits": _ctc_logits(
            [
                ids["5"],
                ids["1"],
                ids["G"],
                ids["-"],
                ids["9"],
                ids["9"],
                ids["9"],
            ],
            len(CHARS),
        ),
        "one_line_logits": _ctc_logits(
            [
                ids["6"],
                ids["0"],
                ids["L"],
                ids["D"],
                ids["-"],
                ids["4"],
                ids["3"],
                ids["0"],
                0,
                ids["0"],
            ],
            len(CHARS),
        ),
        "top_logits": _ctc_logits([ids["5"], ids["1"], ids["G"]], len(CHARS)),
        "bottom_logits": _ctc_logits(
            [ids["1"], ids["2"], ids["3"], ids["."], ids["4"], ids["5"]],
            len(CHARS),
        ),
        "layout_logits": torch.tensor([[5.0, 0.0]]),
    }

    assert line_ctc_greedy_decode(outputs, CHARS) == ["60LD-4300"]

    outputs["layout_logits"] = torch.tensor([[0.0, 5.0]])

    assert line_ctc_greedy_decode(outputs, CHARS) == ["51G[SEP]123.45"]


@pytest.mark.unit
def test_small_lpr_line_ctc_forward_shapes() -> None:
    from lprnet.small_lpr_line_ctc import SmallLPRLineCTC

    model = SmallLPRLineCTC(vocab_size=len(CHARS), d_model=32, backbone_ch=32)
    model.eval()

    with torch.no_grad():
        outputs = model(torch.zeros((2, 3, 48, 96), dtype=torch.float32))

    assert tuple(outputs["global_logits"].shape) == (2, 72, len(CHARS))
    assert tuple(outputs["one_line_logits"].shape) == (2, 16, len(CHARS))
    assert tuple(outputs["top_logits"].shape) == (2, 12, len(CHARS))
    assert tuple(outputs["bottom_logits"].shape) == (2, 12, len(CHARS))
    assert tuple(outputs["layout_logits"].shape) == (2, 2)
    assert tuple(outputs["one_line_attention"].shape) == (2, 6, 12)
    assert tuple(outputs["top_attention"].shape) == (2, 6, 12)
    assert tuple(outputs["bottom_attention"].shape) == (2, 6, 12)


@pytest.mark.unit
def test_small_lpr_line_ctc_forward_shapes_without_stn_or_positional_encoding() -> None:
    from lprnet.small_lpr_line_ctc import SmallLPRLineCTC

    model = SmallLPRLineCTC(
        vocab_size=len(CHARS),
        d_model=32,
        backbone_ch=32,
        use_stn=False,
        use_pos_enc=False,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(torch.zeros((2, 3, 48, 96), dtype=torch.float32))

    assert tuple(outputs["global_logits"].shape) == (2, 72, len(CHARS))
    assert tuple(outputs["layout_logits"].shape) == (2, 2)


@pytest.mark.unit
def test_small_lpr_line_ctc_zero_line_prior_keeps_initial_line_attention_symmetric() -> None:
    from lprnet.small_lpr_line_ctc import SmallLPRLineCTC

    model = SmallLPRLineCTC(
        vocab_size=len(CHARS),
        d_model=32,
        backbone_ch=32,
        line_prior_strength=0.0,
        use_stn=False,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(torch.zeros((1, 3, 48, 96), dtype=torch.float32))

    assert torch.allclose(outputs["top_attention"], outputs["bottom_attention"])


@pytest.mark.unit
def test_line_ctc_dataset_can_disable_train_augmentation(tmp_path: Path) -> None:
    from lprnet.small_lpr_line_ctc_datamodule import SmallLPRLineCTCDataset

    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    train_dir.mkdir()
    valid_dir.mkdir()

    args = SimpleNamespace(
        train_dir=str(train_dir),
        valid_dir=str(valid_dir),
        test_dir=str(valid_dir),
        img_size=(96, 48),
        min_img_width=20,
        min_img_height=8,
        exclude_paths_file="",
        chars=CHARS,
        augment=False,
    )

    dataset = SmallLPRLineCTCDataset(args, "train")

    assert dataset.transform is None


@pytest.mark.unit
def test_line_ctc_dataset_uses_train_augmentation_by_default(tmp_path: Path) -> None:
    from lprnet.small_lpr_line_ctc_datamodule import SmallLPRLineCTCDataset

    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    train_dir.mkdir()
    valid_dir.mkdir()

    args = SimpleNamespace(
        train_dir=str(train_dir),
        valid_dir=str(valid_dir),
        test_dir=str(valid_dir),
        img_size=(96, 48),
        min_img_width=20,
        min_img_height=8,
        exclude_paths_file="",
        chars=CHARS,
    )

    dataset = SmallLPRLineCTCDataset(args, "train")

    assert dataset.transform is not None
