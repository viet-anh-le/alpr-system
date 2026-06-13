from __future__ import annotations

import json
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from synthetic_vn_lplc.audit import audit_vietnamese_sources, reconstruct_label_from_yolo_chars, split_stratified_records
from synthetic_vn_lplc.diffusion import (
    DiffusionRunConfig,
    build_controlnet_lora_command,
    load_controlnet_lora_adapter,
    resolve_controlnet_lora_artifact,
)
from synthetic_vn_lplc.diffusion_dataset import (
    build_plate_style_conditions,
    effective_mask_expand_px,
    letterbox_frame_and_polygon,
    letterbox_plate_image,
    polygon_mask,
    prepare_lplc_plate_style_pair,
)
from synthetic_vn_lplc.diffusion_training import ControlNetLoRATrainConfig, build_controlnet_lora_train_command
from synthetic_vn_lplc.filtering import FilterDecision, filter_manifest_records
from synthetic_vn_lplc.geometry import (
    LAYOUTS,
    fit_layout_polygon,
    polygon_aspect,
    repair_polygon,
)
from synthetic_vn_lplc.grammar import (
    generate_vn_plate_label,
    normalize_vn_label,
    validate_vn_plate_label,
)
from synthetic_vn_lplc.lplc import iter_lplc_records
from synthetic_vn_lplc.pipeline import generate_ocr_sample
from synthetic_vn_lplc.render import render_vn_plate
from synthetic_vn_lplc.style import measure_plate_style


def test_vn_grammar_accepts_required_formats() -> None:
    labels = [
        "51G-12345",
        "50LD-12345",
        "29-A1-12345",
        "31H-9999",
        "29-F4-8888",
        "51A-1234",
        "51-A1-1234",
        "59-MĐ2-01265",
    ]

    for label in labels:
        assert validate_vn_plate_label(label), label


def test_vn_grammar_generator_avoids_existing_labels() -> None:
    existing = {"51G-12345"}
    label = generate_vn_plate_label(seed=1, existing_labels=existing, layout_name="long")

    assert validate_vn_plate_label(label)
    assert label not in existing
    assert normalize_vn_label(" 51 g 12345 ") == "51G-12345"


def test_renderer_preserves_physical_aspect_with_real_font() -> None:
    font_path = Path("font-chu-bien-so-xe/Soxe2banh.TTF")

    long_plate = render_vn_plate("51A-1234", "long", font_path)
    short_plate = render_vn_plate("51-A1-1234", "short", font_path)
    motor_plate = render_vn_plate("29-A1-12345", "motor", font_path)

    assert long_plate.image.size == (1040, 220)
    assert short_plate.image.size == (660, 330)
    assert motor_plate.image.size == (380, 280)
    assert abs(long_plate.aspect_ratio - LAYOUTS["long"].aspect_ratio) < 1e-6
    assert abs(short_plate.aspect_ratio - LAYOUTS["short"].aspect_ratio) < 1e-6
    assert abs(motor_plate.aspect_ratio - 19 / 14) < 1e-6
    assert abs(motor_plate.aspect_ratio - LAYOUTS["motor"].aspect_ratio) < 1e-6
    assert long_plate.glyph_mask.getbbox() is not None
    assert short_plate.glyph_mask.getbbox() is not None
    assert motor_plate.glyph_mask.getbbox() is not None


def test_motorcycle_generator_and_geometry_use_19_to_14_layout() -> None:
    labels = [generate_vn_plate_label(seed=seed, layout_name="motor") for seed in range(20)]
    source = np.asarray([[100, 100], [160, 105], [158, 145], [98, 140]], dtype=np.float32)

    fitted = fit_layout_polygon(source, "motor", image_size=(400, 300))

    assert all(validate_vn_plate_label(label) for label in labels)
    assert all(label.count("-") == 2 for label in labels)
    assert LAYOUTS["motor"].width_mm == 190
    assert LAYOUTS["motor"].height_mm == 140
    assert abs(polygon_aspect(fitted) - 19 / 14) < 0.005


def test_renderer_draws_separator_even_when_font_lacks_dash_glyph() -> None:
    font_path = Path("font-chu-bien-so-xe/Soxe2banh.TTF")

    rendered = render_vn_plate("51A-1234", "long", font_path)
    dash_boxes = [box for char, box in rendered.char_boxes if char == "-"]
    assert dash_boxes
    x1, y1, x2, y2 = dash_boxes[0]
    dash_crop = np.asarray(rendered.glyph_mask.crop((x1, y1, x2, y2)))

    assert dash_crop.max() == 255


def test_geometry_fits_vietnamese_aspect_instead_of_lplc_aspect() -> None:
    lplc_like_poly = np.array(
        [[100, 100], [190, 105], [188, 132], [98, 127]],
        dtype=np.float32,
    )

    fitted = fit_layout_polygon(lplc_like_poly, "long", image_size=(400, 300))

    assert abs(polygon_aspect(fitted) - LAYOUTS["long"].aspect_ratio) < 0.005
    assert abs(polygon_aspect(fitted) - polygon_aspect(lplc_like_poly)) > 1.0


def test_repair_polygon_handles_degenerate_lplc_annotation() -> None:
    degenerate = [281, 394, 346, 394, 346, 420, 346, 420]

    repaired = repair_polygon(degenerate)

    assert repaired.shape == (4, 2)
    assert polygon_aspect(repaired) > 0


def test_repair_polygon_uses_bbox_for_three_point_lplc_annotation() -> None:
    degenerate = [281, 394, 346, 394, 346, 420, 346, 420]

    repaired = repair_polygon(degenerate)

    assert np.allclose(
        repaired,
        np.asarray([[281, 394], [346, 394], [346, 420], [281, 420]], dtype=np.float32),
    )


def test_polygon_mask_caps_large_expansion_for_small_resized_plate() -> None:
    polygon = np.asarray([[46.0, 142.2], [63.3, 142.2], [63.3, 147.8], [46.0, 147.8]], dtype=np.float32)

    requested_expand = 10
    effective_expand = effective_mask_expand_px(polygon, requested_expand)
    mask = polygon_mask((256, 256), polygon, expand_px=requested_expand)
    ys, xs = np.where(mask > 0)

    assert effective_expand == 2
    assert xs.min() >= int(np.floor(polygon[:, 0].min())) - effective_expand - 1
    assert xs.max() <= int(np.ceil(polygon[:, 0].max())) + effective_expand + 1
    assert ys.min() >= int(np.floor(polygon[:, 1].min())) - effective_expand - 1
    assert ys.max() <= int(np.ceil(polygon[:, 1].max())) + effective_expand + 1


def test_lplc_parser_and_style_measurement(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[30:45, 40:90] = (220, 220, 230)
    cv2.imwrite(str(image_dir / "sample.jpg"), image)
    annotations = {
        "sample.jpg": {
            "cam": 1,
            "time": "night",
            "faulty": False,
            "rain": True,
            "day": 2,
            "anns": [
                {
                    "ocr": "ABC1234",
                    "leg": 2,
                    "xy": [40, 30, 90, 30, 90, 45, 40, 45],
                    "occ": False,
                }
            ],
        }
    }
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(annotations), encoding="utf-8")

    records = list(iter_lplc_records(ann_path, image_dir))
    style = measure_plate_style(image, records[0].polygon)

    assert len(records) == 1
    assert records[0].time == "night"
    assert records[0].rain is True
    assert style.brightness_mean > 0
    assert style.blur_laplacian >= 0


def test_audit_vietnamese_sources_keeps_valid_and_rejects_foreign(tmp_path: Path) -> None:
    ocr = tmp_path / "ocr"
    (ocr / "train").mkdir(parents=True)
    (ocr / "train" / "29-A1-12345&1#abc.jpg").write_bytes(b"")
    (ocr / "train" / "BEW2I56&1#foreign.jpg").write_bytes(b"")

    report = audit_vietnamese_sources(ocr_root=ocr)

    assert "29-A1-12345" in report.valid_labels
    assert any(item.label == "BEW2I56" for item in report.rejected_labels)


def test_raw_ocr_char_yolo_audit_reconstructs_lines_and_rejects_baza(tmp_path: Path) -> None:
    raw_ocr = tmp_path / "raw_ocr"
    (raw_ocr / "labels" / "train").mkdir(parents=True)
    (raw_ocr / "data.yaml").write_text(
        'names: ["1", "2", "3", "4", "5", "A"]\n',
        encoding="utf-8",
    )
    yolo_label = raw_ocr / "labels" / "train" / "vn_short.txt"
    yolo_label.write_text(
        "\n".join(
            [
                "4 0.10 0.25 0.05 0.20",
                "0 0.20 0.25 0.05 0.20",
                "5 0.30 0.25 0.05 0.20",
                "0 0.40 0.25 0.05 0.20",
                "1 0.10 0.70 0.05 0.20",
                "2 0.20 0.70 0.05 0.20",
                "3 0.30 0.70 0.05 0.20",
                "4 0.40 0.70 0.05 0.20",
            ]
        ),
        encoding="utf-8",
    )
    baza_label = raw_ocr / "labels" / "train" / "1PlateBaza485.txt"
    baza_label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    report = audit_vietnamese_sources(char_yolo_root=raw_ocr)

    assert reconstruct_label_from_yolo_chars(yolo_label, {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "A"}) == "51-A1-2345"
    assert "51-A1-2345" in report.valid_labels
    assert any(item.reason == "foreign_baza_source" for item in report.rejected_labels)


def test_split_stratified_records_preserves_metadata_buckets(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"")
    records = []
    for index in range(6):
        records.append(
            next(
                iter_lplc_records(
                    _write_lplc_annotation(
                        tmp_path / f"ann_{index}.json",
                        image_path.name,
                        cam=index % 2,
                        time="night" if index % 2 else "morning",
                        leg=index % 3,
                    ),
                    tmp_path,
                )
            )
        )

    splits = split_stratified_records(records, seed=3, train_ratio=0.5, valid_ratio=0.25)

    assert set(splits) == {"train", "valid", "test"}
    assert sum(len(items) for items in splits.values()) == 6
    assert all(record.time in {"morning", "night"} for items in splits.values() for record in items)


def test_diffusion_backend_builds_controlnet_lora_command_without_training() -> None:
    config = DiffusionRunConfig(
        base_model="stable-diffusion-v1-5/stable-diffusion-inpainting",
        controlnet_model="lllyasviel/control_v11p_sd15_inpaint",
        lora_weights=Path("weights/synthetic/vn_lplc_lora"),
        input_manifest=Path("manifest.jsonl"),
        output_dir=Path("out"),
        resolution=512,
        width=512,
        height=256,
        steps=12,
        guidance_scale=6.5,
    )

    command = build_controlnet_lora_command(config)

    assert "StableDiffusionControlNetInpaintPipeline" in command.description
    assert "--controlnet-model lllyasviel/control_v11p_sd15_inpaint" in command.cli
    assert "--lora-weights weights/synthetic/vn_lplc_lora" in command.cli


def test_prepare_diffusion_plate_style_pair_outputs_plate_only_training_contract(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    frame[:, :] = (25, 35, 45)
    frame[10:30, 10:60] = (210, 215, 220)
    frame[16:24, 22:48] = (15, 15, 15)
    cv2.imwrite(str(image_dir / "frame.jpg"), frame)
    ann_path = _write_lplc_annotation(
        tmp_path / "ann.json",
        "frame.jpg",
        cam=5,
        time="night",
        leg=2,
    )
    record = next(iter_lplc_records(ann_path, image_dir))

    pair = prepare_lplc_plate_style_pair(
        record,
        output_dir=tmp_path / "diffusion",
        canvas_size=(512, 256),
        detail_protect_dilate_px=3,
        index=7,
    )

    assert pair.target_path.exists()
    assert pair.masked_image_path.exists()
    assert pair.mask_path.exists()
    assert pair.control_path.exists()
    assert pair.detail_mask_path.exists()
    assert pair.target_path.parent.name == "target"
    assert Image.open(pair.target_path).size == (512, 256)
    mask = np.asarray(Image.open(pair.mask_path).convert("L"))
    detail = np.asarray(Image.open(pair.detail_mask_path).convert("L"))
    control = np.asarray(Image.open(pair.control_path).convert("L"))
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask.sum() > 0
    assert detail.sum() > 0
    assert np.all(mask[detail > 0] == 0)
    assert np.mean(control == 255) < 0.25
    assert pair.to_manifest_record()["task"] == "plate_style_transfer"
    assert pair.to_manifest_record()["canvas_size"] == [512, 256]
    assert pair.to_manifest_record()["prompt"]
    assert pair.to_manifest_record()["metadata"]["cam"] == 5


def test_plate_style_conditions_protect_details_and_do_not_fill_control() -> None:
    plate = np.full((128, 512, 3), 220, dtype=np.uint8)
    cv2.rectangle(plate, (8, 8), (503, 119), (10, 10, 10), 4)
    cv2.putText(plate, "ABC1234", (55, 88), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (15, 15, 15), 5)
    canvas, content_box = letterbox_plate_image(plate, canvas_size=(512, 256))

    masked, surface_mask, control, detail_mask = build_plate_style_conditions(
        canvas,
        content_box=content_box,
        detail_protect_dilate_px=3,
    )

    assert masked.shape == canvas.shape
    assert surface_mask.shape == canvas.shape[:2]
    assert detail_mask.shape == canvas.shape[:2]
    assert np.all(surface_mask[detail_mask > 0] == 0)
    assert np.mean(control[:, :, 0] == 255) < 0.25
    assert np.mean(masked[surface_mask > 0]) == 0


@pytest.mark.parametrize(
    ("label", "layout", "output_shape"),
    [
        ("51A-1234", "long", (220, 1040)),
        ("29-A1-12345", "motor", (280, 380)),
    ],
)
def test_diffusion_infer_styles_only_ocr_crop_and_preserves_vn_glyphs(
    tmp_path: Path,
    monkeypatch,
    label: str,
    layout: str,
    output_shape: tuple[int, int],
) -> None:
    module_path = Path("scripts/synthetic_vn_lplc_diffusion_infer.py")
    spec = importlib.util.spec_from_file_location("synthetic_diffusion_infer", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rendered = render_vn_plate(label, layout, Path("font-chu-bien-so-xe/Soxe2banh.TTF"))
    crop_path = tmp_path / f"{label}&input.png"
    rendered.image.save(crop_path)
    calls: dict[str, Image.Image] = {}

    def fake_run_controlnet_lora_inpaint(**kwargs):
        calls["image"] = kwargs["image"]
        calls["mask_image"] = kwargs["mask_image"]
        calls["control_image"] = kwargs["control_image"]
        return Image.new("RGB", kwargs["image"].size, (255, 255, 255))

    monkeypatch.setattr(module, "run_controlnet_lora_inpaint", fake_run_controlnet_lora_inpaint)

    result = module.harmonize_sample(
        {
            "label": label,
            "layout": layout,
            "ocr_crop_path": str(crop_path),
        },
        pipe=object(),
        run_config=DiffusionRunConfig(
            base_model="base",
            controlnet_model="controlnet",
            lora_weights=None,
            input_manifest=tmp_path / "manifest.jsonl",
            output_dir=tmp_path / "out",
            resolution=512,
            steps=1,
            guidance_scale=1.0,
            width=512,
            height=256,
        ),
        ocr_output_dir=tmp_path / "ocr",
        font_path=Path("font-chu-bien-so-xe/Soxe2banh.TTF"),
        detail_protect_dilate_px=3,
        seed=1,
    )

    mask = np.asarray(calls["mask_image"].convert("L"))
    output = np.asarray(Image.open(result["ocr_crop_path"]).convert("RGB"))
    glyph_mask = np.asarray(rendered.glyph_mask)

    assert calls["image"].size == (512, 256)
    assert mask.mean() > 0
    assert output.shape[:2] == output_shape
    assert output[glyph_mask > 0].mean() < 80
    assert result["generator_backend"] == "controlnet_lora_plate_style"
    assert result["task"] == "ocr_plate_style_output"
    assert "full_frame_path" not in result
    assert "source_full_frame_path" not in result
    assert "yolo_obb_path" not in result
    assert Path(result["ocr_crop_path"]).exists()


def test_controlnet_lora_train_command_is_server_ready() -> None:
    config = ControlNetLoRATrainConfig(
        train_manifest=Path("data/synthetic/diffusion_train/manifest.jsonl"),
        output_dir=Path("weights/synthetic/vn_lplc_lora"),
        pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-inpainting",
        controlnet_model_name_or_path="lllyasviel/control_v11p_sd15_inpaint",
        width=512,
        height=256,
        train_batch_size=1,
        gradient_accumulation_steps=4,
        max_train_steps=1000,
        learning_rate=1e-4,
        lora_rank=16,
        mixed_precision="fp16",
    )

    command = build_controlnet_lora_train_command(config)

    assert command.startswith("accelerate launch scripts/synthetic_vn_lplc_train_controlnet_lora.py")
    assert "--train-manifest data/synthetic/diffusion_train/manifest.jsonl" in command
    assert "--controlnet-model-name-or-path lllyasviel/control_v11p_sd15_inpaint" in command
    assert "--width 512" in command
    assert "--height 256" in command
    assert "--lora-rank 16" in command
    assert "--mixed-precision fp16" in command


def test_controlnet_lora_dataset_keeps_rectangular_plate_canvas(tmp_path: Path) -> None:
    module_path = Path("scripts/synthetic_vn_lplc_train_controlnet_lora.py")
    spec = importlib.util.spec_from_file_location("synthetic_train_lora_dataset", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    paths: dict[str, Path] = {}
    for key, mode, color in [
        ("target_path", "RGB", (220, 220, 220)),
        ("masked_image_path", "RGB", (0, 0, 0)),
        ("control_path", "RGB", (10, 10, 10)),
        ("mask_path", "L", 255),
    ]:
        path = tmp_path / f"{key}.png"
        Image.new(mode, (512, 256), color).save(path)
        paths[key] = path
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task": "plate_style_transfer",
                "prompt": "plate style",
                **{key: str(path) for key, path in paths.items()},
            }
        ),
        encoding="utf-8",
    )

    class Tokenizer:
        model_max_length = 4

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_ids=torch.ones(1, 4, dtype=torch.long))

    dataset = module.PlateStyleManifestDataset(manifest, tokenizer=Tokenizer(), width=512, height=256)
    item = dataset[0]

    assert item["pixel_values"].shape == (3, 256, 512)
    assert item["masked_image_values"].shape == (3, 256, 512)
    assert item["mask_values"].shape == (1, 256, 512)
    assert item["conditioning_pixel_values"].shape == (3, 256, 512)


def test_attach_lora_to_controlnet_falls_back_to_peft_wrapper() -> None:
    module_path = Path("scripts/synthetic_vn_lplc_train_controlnet_lora.py")
    spec = importlib.util.spec_from_file_location("synthetic_train_lora", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Model:
        pass

    wrapped = object()

    def fake_get_peft_model(model, config):
        assert isinstance(model, Model)
        assert config.kwargs["target_modules"] == ["to_q", "to_k", "to_v", "to_out.0"]
        return wrapped

    result = module.attach_lora_to_controlnet(Model(), Config, fake_get_peft_model, rank=4)

    assert result is wrapped


def test_controlnet_lora_training_step_routes_inpaint_channels_correctly() -> None:
    module_path = Path("scripts/synthetic_vn_lplc_train_controlnet_lora.py")
    spec = importlib.util.spec_from_file_location("synthetic_train_lora_step", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    batch = {
        "pixel_values": torch.randn(1, 3, 64, 64),
        "masked_image_values": torch.randn(1, 3, 64, 64),
        "mask_values": torch.ones(1, 1, 64, 64),
        "conditioning_pixel_values": torch.randn(1, 3, 64, 64),
        "input_ids": torch.ones(1, 4, dtype=torch.long),
    }

    unet = _ShapeCheckingUNet()
    loss = module.training_step(
        batch=batch,
        vae=_FakeVAE(),
        unet=unet,
        controlnet=_ShapeCheckingControlNet(),
        text_encoder=_FakeTextEncoder(),
        noise_scheduler=_FakeNoiseScheduler(),
        weight_dtype=torch.float32,
    )

    assert loss.ndim == 0
    loss.backward()
    assert unet.scale.grad is not None


def test_resolve_controlnet_lora_artifact_reads_training_config(tmp_path: Path) -> None:
    lora_dir = tmp_path / "lora"
    lora_dir.mkdir()
    weights = lora_dir / "controlnet_lora.safetensors"
    weights.write_bytes(b"not-real-safetensors")
    (lora_dir / "training_config.json").write_text('{"lora_rank": 8}', encoding="utf-8")

    resolved_weights, training_config = resolve_controlnet_lora_artifact(lora_dir)

    assert resolved_weights == weights
    assert training_config["lora_rank"] == 8


def test_load_controlnet_lora_adapter_merges_peft_wrapper_for_pipeline_compat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lora_dir = tmp_path / "lora"
    lora_dir.mkdir()
    (lora_dir / "controlnet_lora.safetensors").write_bytes(b"fake")
    (lora_dir / "training_config.json").write_text('{"lora_rank": 4}', encoding="utf-8")
    base_controlnet = object()
    captured: dict[str, object] = {}

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakePeftWrapper:
        def __init__(self, base):
            self.base = base
            self.loaded = False
            self.merged = False

        def merge_and_unload(self):
            self.merged = True
            return self.base

    wrapper = FakePeftWrapper(base_controlnet)

    def fake_get_peft_model(model, config):
        captured["model"] = model
        captured["config"] = config
        return wrapper

    def fake_set_peft_model_state_dict(model, state_dict):
        assert model is wrapper
        assert state_dict["adapter"].item() == 1
        model.loaded = True

    fake_peft = SimpleNamespace(
        LoraConfig=FakeLoraConfig,
        get_peft_model=fake_get_peft_model,
        set_peft_model_state_dict=fake_set_peft_model_state_dict,
    )
    fake_safetensors_torch = SimpleNamespace(load_file=lambda path: {"adapter": torch.ones(1)})
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)

    result = load_controlnet_lora_adapter(base_controlnet, lora_dir)

    assert result is base_controlnet
    assert wrapper.loaded
    assert wrapper.merged
    assert captured["model"] is base_controlnet
    assert captured["config"].kwargs["r"] == 4


def test_filter_manifest_records_uses_only_ocr_plate_contract(tmp_path: Path) -> None:
    crop = tmp_path / "51A-1234&1#ok.png"
    crop_pixels = np.full((220, 1040, 3), 230, dtype=np.uint8)
    crop_pixels[60:170, 220:820] = 20
    Image.fromarray(crop_pixels).save(crop)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "label": "51A-1234",
                        "layout": "long",
                        "ocr_crop_path": str(crop),
                        "aspect_ratio": LAYOUTS["long"].aspect_ratio,
                    }
                ),
                json.dumps(
                    {
                        "label": "51A-1234",
                        "layout": "long",
                        "ocr_crop_path": str(crop),
                        "aspect_ratio": LAYOUTS["long"].aspect_ratio,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    decisions = filter_manifest_records(
        manifest,
        ocr_predict=lambda sample: ("51A-1234", 0.91) if sample.index == 0 else ("51A-1235", 0.95),
        ocr_threshold=0.85,
    )

    assert decisions[0].status == FilterDecision.ACCEPTED
    assert decisions[1].status == FilterDecision.REJECTED
    assert "ocr_mismatch" in decisions[1].reasons
    assert "detector_iou" not in decisions[0].to_manifest_record()


def test_filter_rejects_blank_or_wrong_size_ocr_crops(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    wrong_aspect = tmp_path / "wrong.png"
    Image.new("RGB", (1040, 220), (255, 255, 255)).save(blank)
    wrong_pixels = np.full((300, 300, 3), 230, dtype=np.uint8)
    wrong_pixels[80:220, 80:220] = 15
    Image.fromarray(wrong_pixels).save(wrong_aspect)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "label": "51A-1234",
                        "layout": "long",
                        "ocr_crop_path": str(blank),
                        "aspect_ratio": LAYOUTS["long"].aspect_ratio,
                    }
                ),
                json.dumps(
                    {
                        "label": "51A-1234",
                        "layout": "long",
                        "ocr_crop_path": str(wrong_aspect),
                        "aspect_ratio": LAYOUTS["long"].aspect_ratio,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    decisions = filter_manifest_records(manifest)

    assert "blank_or_low_contrast_crop" in decisions[0].reasons
    assert "crop_aspect_mismatch" in decisions[1].reasons


def test_generate_ocr_sample_writes_plate_crop_without_detection_artifacts(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    frame = np.full((96, 128, 3), 35, dtype=np.uint8)
    frame[10:30, 10:60] = (210, 215, 220)
    frame[16:24, 22:48] = (15, 15, 15)
    cv2.imwrite(str(image_dir / "frame.jpg"), frame)
    record = next(
        iter_lplc_records(
            _write_lplc_annotation(tmp_path / "ann.json", "frame.jpg", cam=5, time="night", leg=2),
            image_dir,
        )
    )

    sample = generate_ocr_sample(
        lplc_record=record,
        layout_name="long",
        font_path=Path("font-chu-bien-so-xe/Soxe2banh.TTF"),
        ocr_crop_path=tmp_path / "ocr" / "51A-1234&1#sample.png",
        seed=7,
        label="51A-1234",
    )
    payload = json.loads(sample.to_json())

    assert Image.open(sample.ocr_crop_path).size == LAYOUTS["long"].canvas_size
    assert payload["task"] == "ocr_plate_style_seed"
    assert payload["source_lplc_image"].endswith("frame.jpg")
    assert "full_frame_path" not in payload
    assert "yolo_obb_path" not in payload
    assert "polygon" not in payload


def test_generated_plate_can_be_saved_as_parseq_filename(tmp_path: Path) -> None:
    font_path = Path("font-chu-bien-so-xe/Soxe2banh.TTF")
    rendered = render_vn_plate("51-A1-1234", "short", font_path)
    out_path = tmp_path / "51-A1-1234&short#sample.png"
    rendered.image.save(out_path)

    opened = Image.open(out_path)

    assert opened.size == (660, 330)
    assert out_path.exists()


def _write_lplc_annotation(path: Path, image_name: str, *, cam: int, time: str, leg: int) -> Path:
    payload = {
        image_name: {
            "cam": cam,
            "time": time,
            "faulty": False,
            "rain": False,
            "day": 1,
            "anns": [
                {
                    "ocr": "ABC1234",
                    "leg": leg,
                    "xy": [10, 10, 60, 10, 60, 30, 10, 30],
                    "occ": False,
                }
            ],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _LatentDistribution:
    def __init__(self, sample: torch.Tensor) -> None:
        self._sample = sample

    def sample(self) -> torch.Tensor:
        return self._sample


class _FakeVAE:
    config = SimpleNamespace(scaling_factor=1.0)

    def encode(self, values: torch.Tensor) -> SimpleNamespace:
        pooled = torch.nn.functional.avg_pool2d(values, kernel_size=8)
        first = pooled[:, :1]
        sample = torch.cat([pooled, first], dim=1)
        return SimpleNamespace(latent_dist=_LatentDistribution(sample))


class _FakeTextEncoder:
    def __call__(self, input_ids: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.zeros(input_ids.shape[0], input_ids.shape[1], 8),)


class _FakeNoiseScheduler:
    config = SimpleNamespace(num_train_timesteps=10, prediction_type="epsilon")

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        return latents + noise


class _ShapeCheckingControlNet:
    config = SimpleNamespace(in_channels=4)

    def __call__(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        return_dict: bool,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        del timesteps, encoder_hidden_states, controlnet_cond, return_dict
        assert sample.shape[1] == 4
        residual = torch.zeros_like(sample)
        return [residual], residual


class _ShapeCheckingUNet(torch.nn.Module):
    config = SimpleNamespace(in_channels=9)

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        down_block_additional_residuals: list[torch.Tensor],
        mid_block_additional_residual: torch.Tensor,
    ) -> SimpleNamespace:
        del timesteps, encoder_hidden_states, down_block_additional_residuals, mid_block_additional_residual
        assert sample.shape[1] == 9
        return SimpleNamespace(sample=sample[:, :4] * self.scale)
