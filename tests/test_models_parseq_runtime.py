from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.unit
def test_preprocess_plate_parseq_backend_uses_parseq_image_contract() -> None:
    from api.core.models import preprocess_plate_parseq

    crop = np.full((24, 96, 3), 127, dtype=np.uint8)
    tensor = preprocess_plate_parseq(crop, image_width=128, image_height=32)

    assert tuple(tensor.shape) == (3, 32, 128)
    assert tensor.dtype == torch.float32


@pytest.mark.unit
def test_preprocess_plate_for_model_uses_parseq_checkpoint_dimensions() -> None:
    from api.core.models import ParseqOcrModel, preprocess_plate_for_model

    crop = np.full((24, 96, 3), 127, dtype=np.uint8)
    wrapper = ParseqOcrModel(model=torch.nn.Identity(), image_width=160, image_height=40)
    tensor = preprocess_plate_for_model(wrapper, crop)

    assert tuple(tensor.shape) == (3, 40, 160)


@pytest.mark.unit
def test_parseq_label_probs_collapse_sep_to_single_ocr_token() -> None:
    from api.core.models import parseq_label_to_char_probs

    chars = parseq_label_to_char_probs(
        "59-U1[SEP]027.95",
        torch.full((len("59-U1[SEP]027.95"),), 0.81),
    )

    assert [char for char, _ in chars] == [
        "5",
        "9",
        "-",
        "U",
        "1",
        "[SEP]",
        "0",
        "2",
        "7",
        ".",
        "9",
        "5",
    ]
    assert chars[5] == ("[SEP]", pytest.approx(0.81))


@pytest.mark.unit
def test_ocr_batch_dispatches_parseq_wrapper_and_returns_pipeline_contract() -> None:
    from api.core.models import ParseqOcrModel, ocr_batch

    class FakeTokenizer:
        def decode(self, _probs):
            labels = ["59-U1[SEP]027.95", "30G-51827"]
            batch_probs = [
                torch.full((len(labels[0]),), 0.88),
                torch.full((len(labels[1]),), 0.94),
            ]
            return labels, batch_probs

    class FakeParseq(torch.nn.Module):
        tokenizer = FakeTokenizer()

        def forward(self, images):
            return torch.zeros((images.shape[0], 25, 40), dtype=torch.float32)

    wrapper = ParseqOcrModel(model=FakeParseq(), image_width=128, image_height=32)
    results = ocr_batch(wrapper, torch.zeros((2, 3, 32, 128)), torch.device("cpu"))

    assert [[char for char, _ in chars] for chars, _ in results] == [
        ["5", "9", "-", "U", "1", "[SEP]", "0", "2", "7", ".", "9", "5"],
        list("30G-51827"),
    ]
    assert [all_confident for _, all_confident in results] == [False, True]


@pytest.mark.unit
def test_normalize_ocr_backend_rejects_removed_small_lpr_nar_aliases() -> None:
    from api.core.models import normalize_ocr_backend

    for backend in ("small_lpr_nar", "smalllpr_nar", "nar"):
        with pytest.raises(ValueError):
            normalize_ocr_backend(backend)


@pytest.mark.unit
def test_normalize_ocr_backend_accepts_small_lpr_line_ctc_aliases() -> None:
    from api.core.models import normalize_ocr_backend

    assert normalize_ocr_backend("default") == "smalllpr_line_ctc"
    assert normalize_ocr_backend("small-lpr-line-ctc") == "smalllpr_line_ctc"
    assert normalize_ocr_backend("small_lpr_line_ctc") == "smalllpr_line_ctc"
    assert normalize_ocr_backend("smalllpr_line_ctc") == "smalllpr_line_ctc"
    assert normalize_ocr_backend("line_ctc") == "smalllpr_line_ctc"


@pytest.mark.unit
def test_normalize_ocr_backend_accepts_vietnamese_yolov5_only_as_yolov5_option() -> None:
    from api.core.models import normalize_ocr_backend

    assert normalize_ocr_backend("vietnamese_yolov5") == "vietnamese_yolov5"


@pytest.mark.unit
def test_normalize_ocr_backend_rejects_removed_ocr_backends() -> None:
    from api.core.models import normalize_ocr_backend

    for backend in ("smalllpr", "small_lpr", "smalllpr_ctc", "small_lpr_ctc", "ctc", "parseq", "yolov5_char"):
        with pytest.raises(ValueError, match="smalllpr_line_ctc"):
            normalize_ocr_backend(backend)


@pytest.mark.unit
def test_normalize_ocr_backend_rejects_small_lpr_line_ctc_alnum_backend() -> None:
    from api.core.models import normalize_ocr_backend

    with pytest.raises(ValueError, match="smalllpr_line_ctc"):
        normalize_ocr_backend("small_lpr_line_ctc_alnum")


@pytest.mark.unit
def test_ocr_batch_dispatches_small_lpr_line_ctc_wrapper_with_layout() -> None:
    from api.core.models import SmallLprLineCtcOcrModel, ocr_batch

    chars = ["<pad>", "3", "0", "G", "-", "5", "1", "8", "2", "7", "[SEP]"]

    def logits_from_sequences(sequences: list[list[int]]) -> torch.Tensor:
        width = max(len(sequence) for sequence in sequences)
        logits = torch.full((len(sequences), width, len(chars)), -10.0)
        for batch_idx, sequence in enumerate(sequences):
            for pos, token_id in enumerate(sequence):
                logits[batch_idx, pos, token_id] = 10.0
        return logits

    class FakeLineCtc(torch.nn.Module):
        def forward(self, images):
            return {
                "one_line_logits": logits_from_sequences(
                    [
                        [0, 1, 1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                        [9, 9, 0],
                    ]
                ),
                "top_logits": logits_from_sequences(
                    [
                        [1, 2, 0],
                        [1, 2, 3, 0],
                    ]
                ),
                "bottom_logits": logits_from_sequences(
                    [
                        [5, 6, 0],
                        [5, 6, 7, 8, 9],
                    ]
                ),
                "layout_logits": torch.tensor([[10.0, -10.0], [-10.0, 10.0]]),
            }

    wrapper = SmallLprLineCtcOcrModel(model=FakeLineCtc(), chars=chars)
    results = ocr_batch(wrapper, torch.zeros((2, 3, 48, 96)), torch.device("cpu"))

    assert [[char for char, _ in char_probs] for char_probs, _ in results] == [
        list("30G-51827"),
        ["3", "0", "G", "[SEP]", "5", "1", "8", "2", "7"],
    ]
    assert [all_confident for _, all_confident in results] == [True, True]


@pytest.mark.unit
def test_load_small_lpr_line_ctc_checkpoint_without_global_head(tmp_path) -> None:
    import api.core.models as models

    chars = ["<blank>", "3", "0", "G"]
    source = models.SmallLPRLineCTC(
        vocab_size=len(chars),
        d_model=16,
        backbone_ch=16,
        use_global_head=False,
    )
    checkpoint = tmp_path / "no_global.ckpt"
    torch.save(
        {
            "state_dict": {
                f"model.{name}": tensor
                for name, tensor in source.state_dict().items()
            },
            "hyper_parameters": {
                "args": {
                    "chars": chars,
                    "d_model": 16,
                    "backbone_ch": 16,
                    "use_global_head": False,
                    "line_prior_strength": 1.0,
                    "use_stn": True,
                    "use_pos_enc": True,
                    "two_line_threshold": 0.5,
                    "line_separator": "[SEP]",
                }
            },
        },
        checkpoint,
    )

    wrapper = models.load_small_lpr_line_ctc_model(
        checkpoint,
        device=torch.device("cpu"),
    )

    assert wrapper.model.global_head is None
    assert "global_logits" not in wrapper.model(torch.zeros((1, 3, 48, 96)))


@pytest.mark.unit
def test_ocr_batch_dispatches_small_lpr_ctc_wrapper_and_collapses_repeats() -> None:
    from api.core.models import SmallLprCtcOcrModel, ocr_batch

    chars = ["<blank>", "3", "0", "G", "-", "5", "1", "8", "2", "7", "[SEP]"]

    class FakeCtc(torch.nn.Module):
        def forward(self, images):
            logits = torch.full((images.shape[0], 14, len(chars)), -10.0)
            sequences = [
                [0, 1, 1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0],
                [1, 2, 0, 3, 10, 10, 0, 5, 6, 7, 8, 9, 0, 0],
            ]
            for batch_idx, sequence in enumerate(sequences):
                for pos, token_id in enumerate(sequence):
                    logits[batch_idx, pos, token_id] = 10.0
            return logits

    wrapper = SmallLprCtcOcrModel(model=FakeCtc(), chars=chars)
    results = ocr_batch(wrapper, torch.zeros((2, 3, 48, 96)), torch.device("cpu"))

    assert [[char for char, _ in char_probs] for char_probs, _ in results] == [
        list("30G-51827"),
        ["3", "0", "G", "[SEP]", "5", "1", "8", "2", "7"],
    ]
    assert [all_confident for _, all_confident in results] == [True, True]


@pytest.mark.unit
def test_ocr_batch_handles_ctc_wrapper_loaded_from_core_namespace(monkeypatch) -> None:
    """Monitor events must not fall back to autoregressive SmallLPR OCR.

    The FastAPI entrypoint used to load models through ``core.models`` while
    event analysis imported OCR helpers through ``api.core.models``. That
    made the CTC wrapper fail the ``isinstance`` dispatch and call ``.encode``.
    """
    import importlib

    import api.core.models as api_models

    api_dir = str(api_models.ROOT / "api")
    monkeypatch.syspath_prepend(api_dir)
    core_models = importlib.import_module("core.models")

    chars = ["<blank>", "3", "0"]

    class FakeCtc(torch.nn.Module):
        def forward(self, images):
            logits = torch.full((images.shape[0], 4, len(chars)), -10.0)
            for pos, token_id in enumerate([1, 1, 0, 2]):
                logits[:, pos, token_id] = 10.0
            return logits

    wrapper = core_models.SmallLprCtcOcrModel(model=FakeCtc(), chars=chars)

    results = api_models.ocr_batch(wrapper, torch.zeros((1, 3, 48, 96)), torch.device("cpu"))

    assert [[char for char, _ in char_probs] for char_probs, _ in results] == [["3", "0"]]


@pytest.mark.unit
def test_load_models_uses_small_lpr_line_ctc_backend_by_default(monkeypatch) -> None:
    import api.core.models as models

    monkeypatch.setattr(models.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(models, "YOLO", lambda path: SimpleNamespace(path=str(path)))
    monkeypatch.setattr(
        models,
        "VehicleTracker",
        lambda **kwargs: SimpleNamespace(kind="tracker", kwargs=kwargs),
    )
    monkeypatch.setattr(
        models.PlateQualityRouter,
        "from_env",
        classmethod(lambda cls, device=None: SimpleNamespace(kind="router", device=device)),
    )

    loaded_paths: list[object] = []

    def fake_load_line_ctc(path, *, device):
        loaded_paths.append(path)
        return models.SmallLprLineCtcOcrModel(model=torch.nn.Identity(), chars=["<blank>", "A"])

    monkeypatch.setattr(models, "load_small_lpr_line_ctc_model", fake_load_line_ctc)

    bundle = models.load_models()

    assert isinstance(bundle.ocr, models.SmallLprLineCtcOcrModel)
    assert bundle.ocr_backend == "smalllpr_line_ctc"
    assert loaded_paths[0] == models.SMALL_LPR_LINE_CTC_CKPT_PATH


@pytest.mark.unit
def test_load_models_can_force_reid_tracker_to_cpu(monkeypatch) -> None:
    import api.core.models as models

    monkeypatch.setattr(models, "REID_DEVICE", "cpu")
    monkeypatch.setattr(models.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(models, "YOLOV5_CHAR_CKPT_PATH", Path("__missing_char.pt"))
    monkeypatch.setattr(models, "YOLO", lambda path: SimpleNamespace(path=str(path)))
    monkeypatch.setattr(
        models,
        "load_yolov5_vehicle_detector",
        lambda path, *, device: SimpleNamespace(kind="vehicle", path=str(path), device=device),
    )
    monkeypatch.setattr(
        models.PlateQualityRouter,
        "from_env",
        classmethod(lambda cls, device=None: SimpleNamespace(kind="router", device=device)),
    )
    monkeypatch.setattr(
        models,
        "load_small_lpr_line_ctc_model",
        lambda path, *, device: models.SmallLprLineCtcOcrModel(
            model=torch.nn.Identity(),
            chars=["<blank>", "A"],
        ),
    )

    bundle = models.load_models()

    assert str(bundle.device) == "cuda"
    assert bundle.tracker_device == "cpu"


@pytest.mark.unit
def test_load_models_rejects_removed_ocr_backend_before_loading_models(monkeypatch) -> None:
    import api.core.models as models

    monkeypatch.setattr(models, "OCR_BACKEND", "parseq")
    monkeypatch.setattr(
        models,
        "YOLO",
        lambda path: pytest.fail(f"YOLO should not load for invalid OCR backend: {path}"),
    )

    with pytest.raises(ValueError, match="smalllpr_line_ctc"):
        models.load_models()
