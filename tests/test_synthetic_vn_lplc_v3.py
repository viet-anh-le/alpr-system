from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from synthetic_vn_lplc.blank_bank import iter_vn_blank_candidates
from synthetic_vn_lplc.geometry import LAYOUTS, order_points, repair_polygon
from synthetic_vn_lplc.inpainting import TeleaInpainter, inpaint_masked_region
from synthetic_vn_lplc.lplc import iter_lplc_records
from synthetic_vn_lplc.reference_filtering import ReferenceFilterDecision, filter_reference_manifest, gray_blob_fraction
from synthetic_vn_lplc.reference_geometry import (
    crop_with_context,
    homography_round_trip_error,
    letterbox_image,
    make_letterbox_transform,
    project_layout_quad,
)
from synthetic_vn_lplc.reference_pipeline import (
    PIPELINE_VERSION,
    ReferenceDiffusionConfig,
    restore_only_glyphs,
    run_reference_inpaint,
)
from synthetic_vn_lplc.reference_schema import require_v3_record
from synthetic_vn_lplc.reference_synthesis import (
    composite_degraded_glyphs,
    estimate_donor_degradation,
    generate_reference_sample,
    replay_donor_camera_residual,
)


def test_repair_polygon_preserves_valid_projective_quad() -> None:
    raw = np.asarray([[10, 10], [90, 18], [82, 48], [16, 42]], dtype=np.float32)

    repaired = repair_polygon(raw)

    assert np.allclose(repaired, order_points(raw))
    assert not np.allclose(repaired, cv2.boxPoints(cv2.minAreaRect(raw)))


def test_lplc_parser_preserves_raw_quad_and_car_type(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "sample.jpg"), np.zeros((80, 160, 3), dtype=np.uint8))
    raw = [[20, 20], [130, 24], [126, 50], [24, 47]]
    annotations = {
        "sample.jpg": {
            "faulty": False,
            "anns": [{"ocr": "ABC1234", "xy": raw, "car": {"type": "MOTOCICLETA"}}],
        }
    }
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")

    record = next(iter_lplc_records(annotations_path, images))

    assert np.array_equal(record.raw_polygon, np.asarray(raw, dtype=np.float32))
    assert np.array_equal(record.polygon, order_points(np.asarray(raw, dtype=np.float32)))
    assert record.polygon_method == "annotation_quad"
    assert record.car_type == "MOTOCICLETA"


def test_vn_blank_candidates_never_read_filename_valid_split(tmp_path: Path) -> None:
    ocr_root = tmp_path / "ocr"
    (ocr_root / "train").mkdir(parents=True)
    (ocr_root / "valid").mkdir()
    image = np.full((80, 240, 3), 230, dtype=np.uint8)
    cv2.putText(image, "51A-1234", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (10, 10, 10), 3)
    cv2.imwrite(str(ocr_root / "train" / "51A-1234&1#train.png"), image)
    cv2.imwrite(str(ocr_root / "valid" / "29A-9999&1#valid.png"), image)

    candidates = list(
        iter_vn_blank_candidates(
            raw_ocr_root=None,
            filename_ocr_root=ocr_root,
        )
    )

    assert len(candidates) == 1
    assert candidates[0].image_path.parent.name == "train"


@pytest.mark.parametrize("layout", ["long", "short", "motor"])
def test_project_layout_quad_keeps_projective_pose_and_layout_height(layout: str) -> None:
    source = np.asarray([[60, 50], [190, 62], [178, 100], [68, 91]], dtype=np.float32)

    target, matrix, inverse = project_layout_quad(source, layout)

    assert target.shape == (4, 2)
    assert homography_round_trip_error(matrix, inverse, target) <= 0.5
    source_left = np.linalg.norm(source[3] - source[0])
    source_right = np.linalg.norm(source[2] - source[1])
    target_left = np.linalg.norm(target[3] - target[0])
    target_right = np.linalg.norm(target[2] - target[1])
    assert abs((target_left + target_right) - (source_left + source_right)) < 1.0
    assert LAYOUTS[layout].aspect_ratio > 0


def test_crop_with_context_translates_quad_without_resizing() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    quad = np.asarray([[50, 30], [150, 30], [150, 60], [50, 60]], dtype=np.float32)

    crop = crop_with_context(image, quad, context_fraction=0.2)

    assert crop.image.shape[:2] == (42, 140)
    assert np.allclose(crop.local_quad, quad - np.asarray([30, 24], dtype=np.float32))
    assert crop.offset == (30, 24)


def test_inpaint_masked_region_changes_only_masked_pixels() -> None:
    image = np.full((64, 96, 3), 180, dtype=np.uint8)
    image[20:44, 30:66] = 10
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[18:46, 28:68] = 255

    result = inpaint_masked_region(image, mask, TeleaInpainter(radius=3))

    assert np.array_equal(result[mask == 0], image[mask == 0])
    assert not np.array_equal(result[mask > 0], image[mask > 0])


def test_inpaint_masked_region_normalizes_backend_size() -> None:
    class ResizingInpainter:
        name = "fake_resizing"

        def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
            enlarged = cv2.resize(image_bgr, (image_bgr.shape[1] + 17, image_bgr.shape[0] + 9))
            enlarged[:, :] = (20, 40, 60)
            return enlarged

    image = np.full((35, 79, 3), 180, dtype=np.uint8)
    mask = np.zeros((35, 79), dtype=np.uint8)
    mask[8:25, 20:55] = 255

    result = inpaint_masked_region(image, mask, ResizingInpainter())

    assert result.shape == image.shape
    assert np.array_equal(result[mask == 0], image[mask == 0])
    assert np.all(result[mask > 0] == np.asarray([20, 40, 60], dtype=np.uint8))


def test_restore_only_glyphs_does_not_restore_border_or_surface() -> None:
    pre_composite = np.full((48, 96, 3), 20, dtype=np.uint8)
    generated = np.full((48, 96, 3), 230, dtype=np.uint8)
    glyph_mask = np.zeros((48, 96), dtype=np.uint8)
    glyph_mask[15:34, 34:62] = 255

    restored = restore_only_glyphs(pre_composite, generated, glyph_mask)

    assert np.all(restored[glyph_mask > 0] == 20)
    assert np.all(restored[glyph_mask == 0] == 230)


def test_run_reference_inpaint_passes_style_blank_to_ip_adapter() -> None:
    calls: dict[str, object] = {}

    class FakePipe:
        def set_ip_adapter_scale(self, scale: float) -> None:
            calls["scale"] = scale

        def __call__(self, **kwargs):
            calls.update(kwargs)
            return type("Result", (), {"images": [kwargs["image"]]})()

    image = Image.new("RGB", (64, 64), "white")
    mask = Image.new("L", (64, 64), "black")
    control = Image.new("RGB", (64, 64), "black")
    reference = Image.new("RGB", (64, 64), "gray")
    config = ReferenceDiffusionConfig(ip_adapter_scale=0.65, strength=0.35, steps=4, guidance_scale=3.0)

    result = run_reference_inpaint(
        pipe=FakePipe(),
        image=image,
        mask_image=mask,
        control_image=control,
        style_reference=reference,
        config=config,
        seed=7,
    )

    assert result.size == image.size
    assert calls["ip_adapter_image"] is reference
    assert calls["scale"] == 0.65
    assert calls["strength"] == 0.35
    assert "lora_weights" not in calls


def test_require_v3_record_rejects_legacy_manifest() -> None:
    with pytest.raises(ValueError, match="vn_lplc_reference_v3"):
        require_v3_record({"pipeline_version": "procedural-v2"})

    record = require_v3_record({"pipeline_version": PIPELINE_VERSION, "layout": "long"})
    assert record["layout"] == "long"


def test_reference_filter_allows_camera_view_apparent_aspect(tmp_path: Path) -> None:
    camera = tmp_path / "camera.png"
    rectified = tmp_path / "rectified.png"
    style = tmp_path / "style.png"
    camera_pixels = np.full((80, 180, 3), 210, dtype=np.uint8)
    camera_pixels[25:60, 35:155] = 20
    rectified_pixels = np.full((220, 1040, 3), 220, dtype=np.uint8)
    rectified_pixels[60:170, 180:860] = 20
    Image.fromarray(camera_pixels).save(camera)
    Image.fromarray(rectified_pixels).save(rectified)
    Image.fromarray(camera_pixels).save(style)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "pipeline_version": PIPELINE_VERSION,
                "label": "51A-1234",
                "layout": "long",
                "physical_aspect_ratio": 52 / 11,
                "camera_view_crop_path": str(camera),
                "rectified_ocr_crop_path": str(rectified),
                "lplc_style_blank_path": str(style),
                "homography_round_trip_error_px": 0.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    decisions = filter_reference_manifest(
        manifest,
        camera_ocr_predict=lambda _sample: ("51A-1234", 0.72),
        rectified_ocr_predict=lambda _sample: ("51A-1234", 0.95),
    )

    assert decisions[0].status == ReferenceFilterDecision.ACCEPTED
    assert "camera_aspect_mismatch" not in decisions[0].reasons


def test_reference_filter_routes_medium_camera_confidence_to_hard_pool(tmp_path: Path) -> None:
    camera = tmp_path / "camera.png"
    rectified = tmp_path / "rectified.png"
    style = tmp_path / "style.png"
    pixels = np.full((220, 1040, 3), 210, dtype=np.uint8)
    pixels[60:170, 180:860] = 20
    Image.fromarray(cv2.resize(pixels, (180, 80))).save(camera)
    Image.fromarray(pixels).save(rectified)
    Image.fromarray(cv2.resize(pixels, (180, 80))).save(style)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "pipeline_version": PIPELINE_VERSION,
                "label": "51A-1234",
                "layout": "long",
                "physical_aspect_ratio": 52 / 11,
                "camera_view_crop_path": str(camera),
                "rectified_ocr_crop_path": str(rectified),
                "lplc_style_blank_path": str(style),
                "homography_round_trip_error_px": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    decision = filter_reference_manifest(
        manifest,
        camera_ocr_predict=lambda _sample: ("51A-1234", 0.55),
        rectified_ocr_predict=lambda _sample: ("51A-1234", 0.95),
    )[0]

    assert decision.status == ReferenceFilterDecision.HARD_POOL


def test_reference_filter_applies_style_and_ghost_text_gates(tmp_path: Path) -> None:
    camera = tmp_path / "camera.png"
    rectified = tmp_path / "rectified.png"
    style = tmp_path / "style.png"
    pixels = np.full((220, 1040, 3), 210, dtype=np.uint8)
    pixels[60:170, 180:860] = 20
    Image.fromarray(cv2.resize(pixels, (180, 80))).save(camera)
    Image.fromarray(pixels).save(rectified)
    Image.fromarray(cv2.resize(pixels, (180, 80))).save(style)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "pipeline_version": PIPELINE_VERSION,
                "label": "51A-1234",
                "layout": "long",
                "physical_aspect_ratio": 52 / 11,
                "camera_view_crop_path": str(camera),
                "rectified_ocr_crop_path": str(rectified),
                "lplc_style_blank_path": str(style),
                "homography_round_trip_error_px": 0.1,
                "lplc_blank_quality": {"style_residual_fraction": 0.6},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    decision = filter_reference_manifest(
        manifest,
        style_metric_predict=lambda _sample: {"ip_adapter_embedding_cosine": 0.2},
        min_style_metrics={"ip_adapter_embedding_cosine": 0.5},
        camera_long_secondary_predict=lambda _sample: ("29A-9999", 0.9),
    )[0]

    assert decision.status == ReferenceFilterDecision.REJECTED
    assert "ghost_brazil_text_residual" in decision.reasons
    assert "ip_adapter_embedding_cosine_low" in decision.reasons
    assert "camera_long_secondary_mismatch" in decision.reasons


@pytest.mark.parametrize(
    ("layout", "label", "layout_token"),
    [
        ("long", "51A-1234", "&1#"),
        ("short", "51A-1234", "&2#"),
        ("motor", "51-A1-1234", "&2#"),
    ],
)
def test_generate_reference_sample_is_ocr_only_and_rectifies_physical_aspect(
    tmp_path: Path,
    layout: str,
    label: str,
    layout_token: str,
) -> None:
    canonical_size = LAYOUTS[layout].canvas_size
    vn_blank_path = tmp_path / f"{layout}_blank.png"
    cv2.imwrite(str(vn_blank_path), np.full((canonical_size[1], canonical_size[0], 3), 225, dtype=np.uint8))

    context = np.full((100, 220, 3), 65, dtype=np.uint8)
    context[28:72, 20:200] = 195
    scene_blank_path = tmp_path / "scene_blank.png"
    style_blank_path = tmp_path / "style_blank.png"
    rectified_style_blank_path = tmp_path / "rectified_style_blank.png"
    cv2.imwrite(str(scene_blank_path), context)
    cv2.imwrite(str(style_blank_path), context)
    cv2.imwrite(str(rectified_style_blank_path), cv2.resize(context[28:72, 20:200], (512, 128)))

    vn_blank = {
        "pipeline_version": PIPELINE_VERSION,
        "record_id": f"vn-{layout}",
        "layout": layout,
        "blank_path": str(vn_blank_path),
    }
    reference = {
        "pipeline_version": PIPELINE_VERSION,
        "record_id": "lplc-1",
        "donor_group": "motor" if layout == "motor" else "nonmotor",
        "local_source_quad": [[20, 28], [200, 28], [200, 72], [20, 72]],
        "source_quad": [[100, 80], [280, 80], [280, 124], [100, 124]],
        "context_path": str(style_blank_path),
        "lplc_style_blank_path": str(style_blank_path),
        "lplc_scene_blank_path": str(scene_blank_path),
        "rectified_style_blank_path": str(rectified_style_blank_path),
        "metadata": {"time": "night", "rain": False},
    }

    result = generate_reference_sample(
        vn_blank_record=vn_blank,
        reference_record=reference,
        label=label,
        font_path=Path("font-chu-bien-so-xe/Soxe2banh.TTF"),
        output_root=tmp_path / "generated",
        split="train",
        seed=13,
        backend="deterministic",
        resolution=384,
    )

    camera = cv2.imread(result["camera_view_crop_path"], cv2.IMREAD_COLOR)
    rectified = cv2.imread(result["rectified_ocr_crop_path"], cv2.IMREAD_COLOR)
    assert camera is not None
    assert rectified is not None
    assert rectified.shape[:2] == (canonical_size[1], canonical_size[0])
    assert result["pipeline_version"] == PIPELINE_VERSION
    assert result["physical_aspect_ratio"] == pytest.approx(LAYOUTS[layout].aspect_ratio)
    assert layout_token in Path(result["camera_view_crop_path"]).name
    assert result["backend"] == "deterministic"
    assert result["homography_round_trip_error_px"] <= 0.5
    assert result["mask_alignment_error_px"] <= 1.0
    assert result["homography_condition_number"] < 1e8
    assert "full_frame_path" not in result
    assert "detection_label_path" not in result


def test_letterbox_uses_identical_mapping_for_image_and_mask() -> None:
    image = np.zeros((41, 123, 3), dtype=np.uint8)
    mask = np.zeros((41, 123), dtype=np.uint8)
    image[10:31, 30:91] = 255
    mask[10:31, 30:91] = 255
    transform = make_letterbox_transform((123, 41), (384, 384))

    image_box = letterbox_image(image, transform, interpolation=cv2.INTER_NEAREST)
    mask_box = letterbox_image(mask, transform, interpolation=cv2.INTER_NEAREST)

    image_binary = cv2.cvtColor(image_box, cv2.COLOR_BGR2GRAY) > 0
    assert np.array_equal(image_binary, mask_box > 0)


def test_donor_derived_glyph_degradation_is_deterministic() -> None:
    surface = np.full((80, 240, 3), 220, dtype=np.uint8)
    glyph = np.full_like(surface, 15)
    mask = np.zeros(surface.shape[:2], dtype=np.uint8)
    mask[20:60, 70:170] = 255
    donor = np.tile(np.arange(240, dtype=np.uint8), (80, 1))
    donor = cv2.cvtColor(donor, cv2.COLOR_GRAY2BGR)
    degradation = estimate_donor_degradation(donor)

    first = composite_degraded_glyphs(
        surface,
        glyph,
        mask,
        donor,
        blur_sigma=degradation["glyph_blur_sigma"],
    )
    second = composite_degraded_glyphs(
        surface,
        glyph,
        mask,
        donor,
        blur_sigma=degradation["glyph_blur_sigma"],
    )

    assert np.array_equal(first, second)
    assert not np.array_equal(first, surface)


def test_camera_residual_replay_changes_only_plate_region() -> None:
    generated = np.full((40, 100, 3), 128, dtype=np.uint8)
    donor = generated.copy()
    donor[:, ::2] = 150
    mask = np.zeros((40, 100), dtype=np.uint8)
    mask[10:30, 25:75] = 255

    replayed = replay_donor_camera_residual(generated, donor, mask)

    assert np.array_equal(replayed[mask == 0], generated[mask == 0])
    assert not np.array_equal(replayed[mask > 0], generated[mask > 0])


def test_gray_blob_gate_detects_new_smooth_blob_not_existing_gray_surface() -> None:
    baseline = np.full((80, 180, 3), 220, dtype=np.uint8)
    generated = baseline.copy()
    generated[20:60, 45:135] = 120
    plate_mask = np.full((80, 180), 255, dtype=np.uint8)

    unchanged_fraction = gray_blob_fraction(baseline, plate_mask, reference_bgr=baseline)
    artifact_fraction = gray_blob_fraction(generated, plate_mask, reference_bgr=baseline)

    assert unchanged_fraction == 0.0
    assert artifact_fraction > 0.1
