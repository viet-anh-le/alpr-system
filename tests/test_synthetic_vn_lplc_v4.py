from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from synthetic_vn_lplc.appearance import (
    APPEARANCES,
    classify_plate_appearance,
    get_plate_appearance,
    infer_plate_appearance_class,
)
from synthetic_vn_lplc.blank_bank import iter_vn_blank_candidates
from synthetic_vn_lplc.config import load_config
from synthetic_vn_lplc.legacy_guard import ensure_v3_baseline_allowed
from synthetic_vn_lplc.inpainting import TeleaInpainter
from synthetic_vn_lplc.lplc import LplcRecord
from synthetic_vn_lplc.grammar import generate_vn_plate_label_for_appearance, validate_vn_plate_label_for_appearance
from synthetic_vn_lplc.quality_gates import ModelIdentity, assert_independent_validator
from synthetic_vn_lplc.render import render_vn_plate
from synthetic_vn_lplc.surface_residual_bank import build_surface_residual_candidate
from synthetic_vn_lplc.surface_reconstruction import reconstruct_plate_surface
from synthetic_vn_lplc.template_bank import build_template_bank
from synthetic_vn_lplc.text_removal_dataset import prepare_text_removal_dataset
from synthetic_vn_lplc.text_eraser_training import build_tmim_training_command, validate_tmim_dataset_root
from synthetic_vn_lplc.text_mask import build_text_mask_ensemble
from synthetic_vn_lplc.validators import local_contrast_text_probability
from synthetic_vn_lplc.v4_schema import V4_PIPELINE_VERSION, iter_v4_manifest, require_v4_record
from synthetic_vn_lplc.v4_reference_bank import build_lplc_reference_candidate
from synthetic_vn_lplc.v4_filtering import audit_v4_record
from synthetic_vn_lplc.v4_pilot import V4PromotionMetrics, evaluate_v4_promotion_gate, summarize_v4_accepted_slices
from synthetic_vn_lplc.v4_synthesis import generate_template_dataset, generate_template_sample


FONT_PATH = Path("font-chu-bien-so-xe/Soxe2banh.TTF")
GENERATION_MODELS = (ModelIdentity(family="v4_template_renderer", checkpoint="opencv-pillow-v1"),)
TARGET_OCR_MODEL = ModelIdentity(family="parseq", checkpoint="target-ocr-v1")
SECONDARY_OCR_MODEL = ModelIdentity(family="slot_lpr", checkpoint="secondary-ocr-v1")
TEXT_VALIDATOR_MODEL = ModelIdentity(family="local_contrast_validation", checkpoint="opencv-v1")


def _zero_text_probability(image_bgr: np.ndarray) -> np.ndarray:
    return np.zeros(image_bgr.shape[:2], dtype=np.float32)


@pytest.mark.parametrize(
    ("appearance_class", "expected_polarity"),
    [
        ("civil_white_black", "dark_on_light"),
        ("commercial_yellow_black", "dark_on_light"),
        ("state_blue_white", "light_on_dark"),
        ("military_red_light", "light_on_dark"),
    ],
)
def test_v4_renderer_supports_verified_appearance_polarities(
    appearance_class: str,
    expected_polarity: str,
) -> None:
    rendered = render_vn_plate(
        "51A-1234",
        "long",
        FONT_PATH,
        appearance_class=appearance_class,
    )
    appearance = get_plate_appearance(appearance_class)
    image = np.asarray(rendered.image)
    glyph = np.asarray(rendered.glyph_mask) > 0

    assert appearance.glyph_polarity == expected_polarity
    assert rendered.appearance_class == appearance_class
    assert tuple(np.median(image[glyph], axis=0).astype(int)) == appearance.foreground_rgb
    assert np.any(np.asarray(rendered.border_protect_mask) > 0)
    assert infer_plate_appearance_class(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) == appearance_class


def test_v4_renderer_exposes_mixed_token_styles() -> None:
    rendered = render_vn_plate(
        "80NG-1234",
        "long",
        FONT_PATH,
        appearance_class="diplomatic_white_red_black",
    )

    assert len(rendered.token_masks) == len(rendered.expected_token_styles)
    assert {"red", "black"} <= set(rendered.expected_token_styles)
    assert all(np.any(np.asarray(mask) > 0) for mask in rendered.token_masks)


@pytest.mark.parametrize(
    "appearance_class",
    [
        "civil_white_black",
        "commercial_yellow_black",
        "state_blue_white",
        "military_red_light",
    ],
)
def test_multiplolarity_mask_recovers_rendered_glyphs_without_border(
    appearance_class: str,
) -> None:
    rendered = render_vn_plate(
        "51A-1234",
        "long",
        FONT_PATH,
        appearance_class=appearance_class,
    )
    image_bgr = cv2.cvtColor(np.asarray(rendered.image), cv2.COLOR_RGB2BGR)
    glyph = np.asarray(rendered.glyph_mask) > 0
    border = np.asarray(rendered.border_protect_mask) > 0

    result = build_text_mask_ensemble(
        image_bgr,
        border_protect_mask=np.asarray(rendered.border_protect_mask),
    )

    recall = np.count_nonzero((result.erase_mask > 0) & glyph) / np.count_nonzero(glyph)
    assert recall >= 0.995
    assert not np.any((result.erase_mask > 0) & border)
    assert np.any(result.uncertainty_mask > 0)


def test_deterministic_surface_reconstruction_removes_glyph_without_touching_outside() -> None:
    rendered = render_vn_plate(
        "51A-1234",
        "long",
        FONT_PATH,
        appearance_class="state_blue_white",
    )
    source = cv2.cvtColor(np.asarray(rendered.image), cv2.COLOR_RGB2BGR)
    glyph_mask = np.asarray(rendered.glyph_mask)
    erase_mask = cv2.dilate(glyph_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    result = reconstruct_plate_surface(
        source,
        erase_mask,
        border_protect_mask=np.asarray(rendered.border_protect_mask),
    )
    appearance = get_plate_appearance("state_blue_white")
    expected_background_bgr = np.asarray(appearance.background_rgb[::-1], dtype=np.float32)
    reconstructed_pixels = result.image_bgr[glyph_mask > 0].astype(np.float32)

    assert np.mean(np.abs(reconstructed_pixels - expected_background_bgr)) <= 3.0
    assert np.array_equal(result.image_bgr[erase_mask == 0], source[erase_mask == 0])


def test_mask_ensemble_does_not_erase_a_text_free_low_frequency_surface() -> None:
    height, width = 220, 1040
    gradient = np.tile(np.linspace(90, 145, width, dtype=np.uint8), (height, 1))
    image = cv2.merge((gradient, gradient, gradient))

    result = build_text_mask_ensemble(image)

    assert np.mean(result.erase_mask > 0) < 0.10


def test_mask_ensemble_erases_enclosed_glyph_holes_to_prevent_silhouettes() -> None:
    rendered = render_vn_plate(
        "80A-0699",
        "long",
        FONT_PATH,
        appearance_class="state_blue_white",
    )
    glyph = np.asarray(rendered.glyph_mask)
    flood = glyph.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    enclosed_holes = (cv2.bitwise_not(flood) > 0) & (glyph == 0)
    result = build_text_mask_ensemble(
        cv2.cvtColor(np.asarray(rendered.image), cv2.COLOR_RGB2BGR),
        border_protect_mask=np.asarray(rendered.border_protect_mask),
        glyph_polarity="light_on_dark",
    )

    assert np.any(enclosed_holes)
    assert np.all(result.erase_mask[enclosed_holes] > 0)


def test_v4_filename_candidate_does_not_depend_on_legacy_dark_mask(tmp_path: Path) -> None:
    ocr_root = tmp_path / "ocr"
    (ocr_root / "train").mkdir(parents=True)
    rendered = render_vn_plate(
        "51A-1234",
        "long",
        FONT_PATH,
        appearance_class="state_blue_white",
    )
    rendered.image.save(ocr_root / "train" / "51A-1234&1#blue.png")

    candidates = list(
        iter_vn_blank_candidates(
            raw_ocr_root=None,
            filename_ocr_root=ocr_root,
            use_legacy_filename_mask=False,
        )
    )

    assert len(candidates) == 1
    assert not np.any(candidates[0].glyph_mask)


def test_appearance_classifier_exposes_low_confidence_instead_of_forcing_red() -> None:
    non_plate = np.full((300, 200, 3), (57, 58, 65), dtype=np.uint8)

    estimate = classify_plate_appearance(non_plate)

    assert estimate.appearance_class == "military_red_light"
    assert estimate.distance_rgb > 60.0


def test_template_bank_is_text_free_and_only_emits_accepted_v4_records(tmp_path: Path) -> None:
    accepted_manifest = tmp_path / "accepted.jsonl"

    written = build_template_bank(
        output_dir=tmp_path / "bank",
        accepted_manifest=accepted_manifest,
        layouts=("long",),
        appearance_classes=("state_blue_white", "military_red_light"),
    )
    records = list(iter_v4_manifest(accepted_manifest))

    assert written == 2
    assert len(records) == 2
    assert all(record["status"] == "accepted" for record in records)
    assert all(record["quality_metrics"]["glyph_pixel_fraction"] == 0.0 for record in records)
    for record in records:
        blank = cv2.imread(record["template_blank_path"], cv2.IMREAD_COLOR)
        glyph = cv2.imread(record["glyph_mask_path"], cv2.IMREAD_GRAYSCALE)
        assert blank is not None
        assert glyph is not None
        assert not np.any(glyph)


def test_v4_manifest_rejects_v3_and_missing_quality_metrics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=V4_PIPELINE_VERSION):
        require_v4_record({"pipeline_version": "vn_lplc_reference_v3"})

    with pytest.raises(ValueError, match="quality_metrics"):
        require_v4_record(
            {
                "pipeline_version": V4_PIPELINE_VERSION,
                "status": "accepted",
                "appearance_class": "state_blue_white",
                "layout": "long",
            }
        )

    manifest = tmp_path / "records.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "pipeline_version": V4_PIPELINE_VERSION,
                "status": "manual_review",
                "appearance_class": "state_blue_white",
                "layout": "long",
                "quality_metrics": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert list(iter_v4_manifest(manifest)) == []
    assert len(list(iter_v4_manifest(manifest, accepted_only=False))) == 1


def test_v3_kill_switch_only_allows_explicit_ablation_outputs(tmp_path: Path) -> None:
    ablation_root = tmp_path / "ablation"

    with pytest.raises(RuntimeError, match="blocked"):
        ensure_v3_baseline_allowed(
            production_enabled=False,
            allow_legacy_baseline=False,
            output_paths=(ablation_root / "run",),
            ablation_root=ablation_root,
        )

    with pytest.raises(ValueError, match="ablation"):
        ensure_v3_baseline_allowed(
            production_enabled=False,
            allow_legacy_baseline=True,
            output_paths=(tmp_path / "production",),
            ablation_root=ablation_root,
        )

    ensure_v3_baseline_allowed(
        production_enabled=False,
        allow_legacy_baseline=True,
        output_paths=(ablation_root / "run",),
        ablation_root=ablation_root,
    )


def test_v4_config_disables_v3_and_defines_isolated_outputs() -> None:
    config = load_config("configs/synthetic/vn_lplc.yaml")

    assert not config.reference_v3.production_enabled
    assert config.reference_v3.ablation_root.name == "vn_lplc_reference_v3_ablation"
    assert config.reference_v4.pipeline_version == V4_PIPELINE_VERSION
    assert config.reference_v4.template_manifest.name == "template_bank_accepted.jsonl"


def test_template_only_v4_generation_preserves_appearance_contract(tmp_path: Path) -> None:
    accepted_manifest = tmp_path / "template_accepted.jsonl"
    build_template_bank(
        output_dir=tmp_path / "bank",
        accepted_manifest=accepted_manifest,
        layouts=("long",),
        appearance_classes=("state_blue_white",),
    )
    template_record = next(iter_v4_manifest(accepted_manifest))

    record = generate_template_sample(
        template_record=template_record,
        label="51A-1234",
        font_path=FONT_PATH,
        output_root=tmp_path / "generated",
        split="train",
        seed=7,
        target_ocr_predict=lambda _image: ("51A-1234", 0.99),
        secondary_ocr_predict=lambda _image: ("20D-8100", 0.98),
        text_probability_predict=_zero_text_probability,
        generation_models=GENERATION_MODELS,
        target_ocr_model=TARGET_OCR_MODEL,
        secondary_ocr_model=SECONDARY_OCR_MODEL,
        validation_text_model=TEXT_VALIDATOR_MODEL,
    )

    assert record["pipeline_version"] == V4_PIPELINE_VERSION
    assert record["status"] == "accepted"
    assert record["appearance_class"] == "state_blue_white"
    assert record["source_vn_label"] is None
    assert record["source_lplc_label"] is None
    assert Path(record["ocr_crop_path"]).exists()
    assert Path(record["target_glyph_mask_path"]).exists()
    assert set(record["quality_metrics"]) >= {"glyph_pixel_fraction", "canonical_aspect_error"}
    assert record["quality_metrics"]["token_color_mismatch_fraction"] == 0.0


def test_generated_v4_sample_without_all_independent_validators_is_manual_review(tmp_path: Path) -> None:
    accepted_manifest = tmp_path / "template_accepted.jsonl"
    build_template_bank(
        output_dir=tmp_path / "bank",
        accepted_manifest=accepted_manifest,
        layouts=("long",),
        appearance_classes=("state_blue_white",),
    )
    template_record = next(iter_v4_manifest(accepted_manifest))

    record = generate_template_sample(
        template_record=template_record,
        label="51A-1234",
        font_path=FONT_PATH,
        output_root=tmp_path / "generated",
        split="train",
        seed=8,
        target_ocr_predict=lambda _image: ("51A-1234", 0.99),
        target_ocr_model=TARGET_OCR_MODEL,
    )

    assert record["status"] == "manual_review"
    assert "missing_generation_model_identity" in record["reject_reasons"]
    assert "missing_independent_secondary_ocr_validator" in record["reject_reasons"]
    assert "missing_independent_text_validator" in record["reject_reasons"]


def test_generated_v4_sample_rejects_text_probability_outside_target(tmp_path: Path) -> None:
    accepted_manifest = tmp_path / "template_accepted.jsonl"
    build_template_bank(
        output_dir=tmp_path / "bank",
        accepted_manifest=accepted_manifest,
        layouts=("long",),
        appearance_classes=("state_blue_white",),
    )
    template_record = next(iter_v4_manifest(accepted_manifest))

    def outside_text_probability(image_bgr: np.ndarray) -> np.ndarray:
        probability = np.zeros(image_bgr.shape[:2], dtype=np.float32)
        probability[image_bgr.shape[0] // 2 :, : image_bgr.shape[1] // 5] = 1.0
        return probability

    record = generate_template_sample(
        template_record=template_record,
        label="51A-1234",
        font_path=FONT_PATH,
        output_root=tmp_path / "generated",
        split="train",
        seed=10,
        target_ocr_predict=lambda _image: ("51A-1234", 0.99),
        secondary_ocr_predict=lambda _image: ("51A-1234", 0.98),
        text_probability_predict=outside_text_probability,
        generation_models=GENERATION_MODELS,
        target_ocr_model=TARGET_OCR_MODEL,
        secondary_ocr_model=SECONDARY_OCR_MODEL,
        validation_text_model=TEXT_VALIDATOR_MODEL,
        max_outside_target_text_probability=0.01,
    )

    assert record["status"] == "rejected"
    assert "text_probability_outside_target_high" in record["reject_reasons"]


def test_generated_v4_sample_rejects_secondary_ocr_source_text(tmp_path: Path) -> None:
    accepted_manifest = tmp_path / "template_accepted.jsonl"
    build_template_bank(
        output_dir=tmp_path / "bank",
        accepted_manifest=accepted_manifest,
        layouts=("long",),
        appearance_classes=("state_blue_white",),
    )
    template_record = next(iter_v4_manifest(accepted_manifest))
    reference_record = {
        "pipeline_version": V4_PIPELINE_VERSION,
        "record_id": "reference-ghost",
        "task": "lplc_reference_bank",
        "status": "accepted",
        "reject_reasons": [],
        "appearance_class": "civil_white_black",
        "layout": "long",
        "source_ocr": "BRA9876",
        "donor_group": "nonmotor",
        "style_residual_path": str(tmp_path / "residual.png"),
        "quality_metrics": {"residual_stroke_energy_ratio": 0.0},
    }
    cv2.imwrite(str(tmp_path / "residual.png"), np.full((220, 1040, 3), 128, dtype=np.uint8))

    record = generate_template_sample(
        template_record=template_record,
        reference_record=reference_record,
        label="51A-1234",
        font_path=FONT_PATH,
        output_root=tmp_path / "generated",
        split="train",
        seed=11,
        target_ocr_predict=lambda _image: ("51A-1234", 0.99),
        secondary_ocr_predict=lambda _image: ("BRA9876", 0.98),
        text_probability_predict=_zero_text_probability,
        generation_models=GENERATION_MODELS,
        target_ocr_model=TARGET_OCR_MODEL,
        secondary_ocr_model=SECONDARY_OCR_MODEL,
        validation_text_model=TEXT_VALIDATOR_MODEL,
    )

    assert record["status"] == "rejected"
    assert "source_lplc_label_or_substring_detected" in record["reject_reasons"]


def test_generated_v4_sample_rejects_vn_surface_source_text(tmp_path: Path) -> None:
    accepted_manifest = tmp_path / "template_accepted.jsonl"
    build_template_bank(
        output_dir=tmp_path / "bank",
        accepted_manifest=accepted_manifest,
        layouts=("long",),
        appearance_classes=("state_blue_white",),
    )
    template_record = next(iter_v4_manifest(accepted_manifest))
    residual_path = tmp_path / "surface_residual.png"
    cv2.imwrite(str(residual_path), np.full((220, 1040, 3), 128, dtype=np.uint8))
    surface_record = {
        "pipeline_version": V4_PIPELINE_VERSION,
        "record_id": "surface-source",
        "task": "vn_surface_residual",
        "status": "accepted",
        "reject_reasons": [],
        "appearance_class": "state_blue_white",
        "layout": "long",
        "source_label": "30A-98765",
        "surface_residual_path": str(residual_path),
        "quality_metrics": {"residual_stroke_energy_ratio": 0.0},
    }

    record = generate_template_sample(
        template_record=template_record,
        surface_record=surface_record,
        label="51A-1234",
        font_path=FONT_PATH,
        output_root=tmp_path / "generated",
        split="train",
        seed=12,
        target_ocr_predict=lambda _image: ("51A-1234", 0.99),
        secondary_ocr_predict=lambda _image: ("30A-98765", 0.98),
        text_probability_predict=_zero_text_probability,
        generation_models=GENERATION_MODELS,
        target_ocr_model=TARGET_OCR_MODEL,
        secondary_ocr_model=SECONDARY_OCR_MODEL,
        validation_text_model=TEXT_VALIDATOR_MODEL,
    )

    assert record["status"] == "rejected"
    assert record["source_vn_label"] == "30A-98765"
    assert "source_vn_label_or_substring_detected" in record["reject_reasons"]


def test_v4_dataset_generation_excludes_existing_real_vn_labels(tmp_path: Path) -> None:
    template_manifest = tmp_path / "templates.jsonl"
    build_template_bank(
        output_dir=tmp_path / "templates",
        accepted_manifest=template_manifest,
        layouts=("long",),
        appearance_classes=("state_blue_white",),
    )
    excluded = generate_vn_plate_label_for_appearance(
        appearance_class="state_blue_white",
        layout_name="long",
        seed=50,
    )
    accepted = tmp_path / "accepted.jsonl"
    counts = generate_template_dataset(
        template_records=iter_v4_manifest(template_manifest),
        output_root=tmp_path / "generated",
        accepted_manifest=accepted,
        rejected_manifest=tmp_path / "rejected.jsonl",
        manual_review_manifest=tmp_path / "manual.jsonl",
        font_path=FONT_PATH,
        counts={"long": 1, "short": 0, "motor": 0},
        split="train",
        seed=50,
        excluded_labels=(excluded,),
        target_ocr_predict=lambda image: ("", 0.0),
        secondary_ocr_predict=lambda image: ("UNRELATED", 0.99),
        text_probability_predict=_zero_text_probability,
        generation_models=GENERATION_MODELS,
        target_ocr_model=TARGET_OCR_MODEL,
        secondary_ocr_model=SECONDARY_OCR_MODEL,
        validation_text_model=TEXT_VALIDATOR_MODEL,
    )
    generated_records = list(iter_v4_manifest(tmp_path / "rejected.jsonl", accepted_only=False))

    assert counts["rejected"] == 1
    assert generated_records[0]["target_label"] != excluded


def test_local_contrast_validator_detects_glyphs_for_dark_and_light_polarities() -> None:
    for appearance_class in ("civil_white_black", "state_blue_white"):
        rendered = render_vn_plate(
            "51A-1234",
            "long",
            FONT_PATH,
            appearance_class=appearance_class,
        )
        image = cv2.cvtColor(np.asarray(rendered.image), cv2.COLOR_RGB2BGR)
        glyph = np.asarray(rendered.glyph_mask) > 0
        border = np.asarray(rendered.border_protect_mask) > 0
        probability = local_contrast_text_probability(image)
        exclusion_kernel = np.ones((21, 21), dtype=np.uint8)
        interior_non_glyph = (
            ~cv2.dilate(glyph.astype(np.uint8), exclusion_kernel).astype(bool)
            & ~cv2.dilate(border.astype(np.uint8), exclusion_kernel).astype(bool)
        )

        assert float(np.mean(probability[glyph])) >= 0.25
        assert float(np.mean(probability[interior_non_glyph])) <= 0.05


def test_v4_reference_style_uses_zero_mean_residual_and_preserves_template_border(tmp_path: Path) -> None:
    accepted_manifest = tmp_path / "template_accepted.jsonl"
    build_template_bank(
        output_dir=tmp_path / "bank",
        accepted_manifest=accepted_manifest,
        layouts=("long",),
        appearance_classes=("state_blue_white",),
    )
    template_record = next(iter_v4_manifest(accepted_manifest))
    blank = cv2.imread(template_record["template_blank_path"], cv2.IMREAD_COLOR)
    border = cv2.imread(template_record["border_protect_mask_path"], cv2.IMREAD_GRAYSCALE)
    assert blank is not None and border is not None
    residual = np.full_like(blank, 128)
    residual[:, ::2] = 134
    residual[:, 1::2] = 122
    residual_path = tmp_path / "style_residual.png"
    cv2.imwrite(str(residual_path), residual)
    reference_record = {
        "pipeline_version": V4_PIPELINE_VERSION,
        "record_id": "reference-1",
        "task": "lplc_reference_bank",
        "status": "accepted",
        "reject_reasons": [],
        "appearance_class": "civil_white_black",
        "layout": "long",
        "source_ocr": "BRA9876",
        "donor_group": "nonmotor",
        "style_residual_path": str(residual_path),
        "quality_metrics": {"residual_stroke_energy_ratio": 0.0},
    }

    record = generate_template_sample(
        template_record=template_record,
        reference_record=reference_record,
        label="51A-1234",
        font_path=FONT_PATH,
        output_root=tmp_path / "generated",
        split="train",
        seed=9,
        target_ocr_predict=lambda _image: ("51A-1234", 0.99),
        secondary_ocr_predict=lambda _image: ("51A-1234", 0.98),
        text_probability_predict=_zero_text_probability,
        generation_models=GENERATION_MODELS,
        target_ocr_model=TARGET_OCR_MODEL,
        secondary_ocr_model=SECONDARY_OCR_MODEL,
        validation_text_model=TEXT_VALIDATOR_MODEL,
    )

    generated = cv2.imread(record["ocr_crop_path"], cv2.IMREAD_COLOR)
    glyph = cv2.imread(record["target_glyph_mask_path"], cv2.IMREAD_GRAYSCALE)
    assert generated is not None and glyph is not None
    assert record["source_lplc_label"] == "BRA9876"
    assert np.array_equal(generated[border > 0], blank[border > 0])
    non_glyph = (glyph == 0) & (border == 0)
    assert np.max(np.abs(np.mean(generated[non_glyph], axis=0) - np.mean(blank[non_glyph], axis=0))) <= 1.0


def test_all_required_v4_appearances_are_registered() -> None:
    assert {
        "civil_white_black",
        "state_blue_white",
        "commercial_yellow_black",
        "diplomatic_white_red_black",
        "military_red_light",
    } <= set(APPEARANCES)


def test_appearance_aware_grammar_blocks_unverified_military_red_labels() -> None:
    blue = generate_vn_plate_label_for_appearance(
        appearance_class="state_blue_white",
        layout_name="long",
        seed=3,
    )
    diplomatic = generate_vn_plate_label_for_appearance(
        appearance_class="diplomatic_white_red_black",
        layout_name="long",
        seed=4,
    )

    assert validate_vn_plate_label_for_appearance(blue, "state_blue_white")
    assert validate_vn_plate_label_for_appearance(diplomatic, "diplomatic_white_red_black")
    assert any(serial in diplomatic for serial in ("NG", "QT"))
    assert not validate_vn_plate_label_for_appearance("51A-1234", "military_red_light")
    with pytest.raises(ValueError, match="verified grammar"):
        generate_vn_plate_label_for_appearance(
            appearance_class="military_red_light",
            layout_name="long",
            seed=5,
        )


def test_validation_only_model_must_use_a_different_family_and_checkpoint() -> None:
    generator = (ModelIdentity(family="dbnet", checkpoint="mask.pt"),)

    with pytest.raises(ValueError, match="family"):
        assert_independent_validator(
            generator,
            ModelIdentity(family="dbnet", checkpoint="other.pt"),
        )
    with pytest.raises(ValueError, match="checkpoint"):
        assert_independent_validator(
            generator,
            ModelIdentity(family="craft", checkpoint="mask.pt"),
        )

    assert_independent_validator(
        generator,
        ModelIdentity(family="craft", checkpoint="validator.pt"),
    )


def test_surface_residual_candidate_without_independent_validators_is_manual_review(tmp_path: Path) -> None:
    rendered = render_vn_plate(
        "51A-1234",
        "long",
        FONT_PATH,
        appearance_class="state_blue_white",
    )
    source = cv2.cvtColor(np.asarray(rendered.image), cv2.COLOR_RGB2BGR)

    record = build_surface_residual_candidate(
        image_bgr=source,
        source_image="source.png",
        source_label="51A-1234",
        source_kind="test",
        layout="long",
        appearance_class="state_blue_white",
        output_root=tmp_path,
        coarse_prompt=np.asarray(rendered.glyph_mask),
    )

    assert record["status"] == "manual_review"
    assert "missing_independent_text_validator" in record["reject_reasons"]
    assert "missing_independent_ocr_validator" in record["reject_reasons"]
    assert Path(record["reconstructed_blank_path"]).exists()
    assert Path(record["surface_residual_path"]).exists()


def test_surface_residual_candidate_passes_only_after_independent_gates(tmp_path: Path) -> None:
    rendered = render_vn_plate(
        "51A-1234",
        "long",
        FONT_PATH,
        appearance_class="state_blue_white",
    )
    source = cv2.cvtColor(np.asarray(rendered.image), cv2.COLOR_RGB2BGR)
    expected_glyph = np.asarray(rendered.glyph_mask)

    def independent_text_probability(image_bgr: np.ndarray) -> np.ndarray:
        color = np.linalg.norm(image_bgr.astype(np.float32) - np.asarray([244, 248, 248]), axis=2)
        probability = np.where(color < 30.0, 1.0, 0.0).astype(np.float32)
        return probability * (expected_glyph > 0)

    record = build_surface_residual_candidate(
        image_bgr=source,
        source_image="source.png",
        source_label="51A-1234",
        source_kind="test",
        layout="long",
        appearance_class="state_blue_white",
        output_root=tmp_path,
        coarse_prompt=expected_glyph,
        text_probability_predict=independent_text_probability,
        blank_ocr_predict=lambda _image: ("", 0.01),
        generation_models=(ModelIdentity(family="mask_ensemble", checkpoint="v4"),),
        validation_text_model=ModelIdentity(family="test_text_validator", checkpoint="text-v1"),
        validation_ocr_model=ModelIdentity(family="test_ocr_validator", checkpoint="ocr-v1"),
    )

    blank = cv2.imread(record["reconstructed_blank_path"], cv2.IMREAD_COLOR)
    assert record["status"] == "accepted"
    assert record["quality_metrics"]["residual_stroke_energy_ratio"] <= 0.05
    assert blank is not None
    assert np.mean(independent_text_probability(blank)) == 0.0


def test_v4_lplc_reference_cleanup_handles_light_glyphs_and_independent_gates(tmp_path: Path) -> None:
    rendered = render_vn_plate(
        "51A-1234",
        "long",
        FONT_PATH,
        appearance_class="state_blue_white",
    )
    plate = cv2.cvtColor(np.asarray(rendered.image), cv2.COLOR_RGB2BGR)
    frame = np.full((300, 1200, 3), 35, dtype=np.uint8)
    frame[40:260, 80:1120] = plate
    quad = np.asarray([[80, 40], [1119, 40], [1119, 259], [80, 259]], dtype=np.float32)
    record = LplcRecord(
        image_path=tmp_path / "source.jpg",
        polygon=quad,
        raw_polygon=quad,
        polygon_method="annotation_quad",
        ocr="BRA1234",
        cam=1,
        time="day",
        rain=False,
        day=1,
        legibility=1,
        occluded=False,
        faulty=False,
        car_type="CAR",
    )

    def independent_text_probability(image_bgr: np.ndarray) -> np.ndarray:
        light = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        probability = np.where(light > 220, 1.0, 0.0).astype(np.float32)
        margin_y = max(1, int(round(image_bgr.shape[0] * 0.15)))
        margin_x = max(1, int(round(image_bgr.shape[1] * 0.04)))
        probability[:margin_y] = 0
        probability[-margin_y:] = 0
        probability[:, :margin_x] = 0
        probability[:, -margin_x:] = 0
        return probability

    result = build_lplc_reference_candidate(
        record=record,
        frame_bgr=frame,
        output_root=tmp_path / "reference",
        inpainter=TeleaInpainter(),
        text_probability_predict=independent_text_probability,
        blank_ocr_predict=lambda _image: ("", 0.01),
        generation_models=(ModelIdentity(family="v4_mask_ensemble", checkpoint="opencv-v1"),),
        validation_text_model=ModelIdentity(family="reference_text_validator", checkpoint="text-v1"),
        validation_ocr_model=ModelIdentity(family="reference_ocr_validator", checkpoint="ocr-v1"),
    )

    rectified_blank = cv2.imread(result["rectified_style_blank_path"], cv2.IMREAD_COLOR)
    assert result["status"] == "accepted"
    assert result["quality_metrics"]["residual_stroke_energy_ratio"] <= 0.05
    assert rectified_blank is not None
    assert np.mean(independent_text_probability(rectified_blank)) == 0.0


def test_text_removal_training_pairs_have_exact_masks_and_text_free_targets(tmp_path: Path) -> None:
    template_manifest = tmp_path / "templates.jsonl"
    build_template_bank(
        output_dir=tmp_path / "templates",
        accepted_manifest=template_manifest,
        layouts=("long",),
        appearance_classes=("state_blue_white", "civil_white_black"),
    )
    output_manifest = tmp_path / "pairs.jsonl"

    written = prepare_text_removal_dataset(
        template_records=iter_v4_manifest(template_manifest),
        output_root=tmp_path / "pairs",
        output_manifest=output_manifest,
        font_path=FONT_PATH,
        count=3,
        split="train",
        seed=11,
    )
    records = list(iter_v4_manifest(output_manifest))

    assert written == 3
    assert len(records) == 3
    for record in records:
        source = cv2.imread(record["text_image_path"], cv2.IMREAD_COLOR)
        target = cv2.imread(record["clean_blank_path"], cv2.IMREAD_COLOR)
        mask = cv2.imread(record["stroke_mask_path"], cv2.IMREAD_GRAYSCALE)
        assert source is not None and target is not None and mask is not None
        difference = np.any(source != target, axis=2)
        assert np.any(difference)
        assert np.all(mask[difference] > 0)
        assert np.array_equal(source[mask == 0], target[mask == 0])
        assert Path(record["text_image_path"]).parent.name == "all_images"
        assert Path(record["clean_blank_path"]).parent.name == "all_labels"
        assert Path(record["stroke_mask_path"]).parent.name == "mask"

    dataset_root = tmp_path / "pairs" / "text_rmv" / "VNPlate"
    assert validate_tmim_dataset_root(dataset_root, required_splits=("train",))["train"] == 3


def test_tmim_training_launcher_requires_custom_small_model_config_and_resource_override(tmp_path: Path) -> None:
    tmim_root = tmp_path / "TMIM"
    (tmim_root / "configs").mkdir(parents=True)
    (tmim_root / "train.py").write_text("print('train')\n", encoding="utf-8")
    official_b = tmim_root / "configs" / "uformer_b_str.py"
    official_b.write_text("data_root = 'data/text_rmv/VNPlate'\n", encoding="utf-8")
    custom_t = tmim_root / "configs" / "uformer_t_vn_plate.py"
    custom_t.write_text("data_root = 'data/text_rmv/VNPlate'\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Uformer-B"):
        build_tmim_training_command(
            tmim_root=tmim_root,
            config_path=official_b,
            model_size="B",
            checkpoint_name="vn-plate-b",
            nproc_per_node=2,
        )
    with pytest.raises(ValueError, match="custom"):
        build_tmim_training_command(
            tmim_root=tmim_root,
            config_path=official_b,
            model_size="T",
            checkpoint_name="vn-plate-t",
            nproc_per_node=2,
        )

    command = build_tmim_training_command(
        tmim_root=tmim_root,
        config_path=custom_t,
        model_size="T",
        checkpoint_name="vn-plate-t",
        nproc_per_node=2,
        python_executable="/env/bin/python",
    )

    assert command[:3] == ["/env/bin/python", "-m", "torch.distributed.run"]
    assert "--nproc_per_node=2" in command
    assert str(custom_t.resolve()) in command


def test_v4_metadata_audit_never_promotes_missing_independent_gates(tmp_path: Path) -> None:
    crop_path = tmp_path / "crop.png"
    glyph_path = tmp_path / "glyph.png"
    cv2.imwrite(str(crop_path), np.zeros((32, 128, 3), dtype=np.uint8))
    cv2.imwrite(str(glyph_path), np.zeros((32, 128), dtype=np.uint8))
    record = {
        "pipeline_version": V4_PIPELINE_VERSION,
        "record_id": "generated-1",
        "task": "ocr_template_synthesis",
        "status": "accepted",
        "reject_reasons": [],
        "appearance_class": "state_blue_white",
        "layout": "long",
        "ocr_crop_path": str(crop_path),
        "target_glyph_mask_path": str(glyph_path),
        "quality_metrics": {
            "target_ocr_confidence": 0.99,
            "secondary_ocr_confidence": 0.98,
            "outside_target_text_probability": 0.0,
            "appearance_distance_rgb": 2.0,
        },
        "models": {
            "generation": [{"family": "v4_template_renderer", "checkpoint": "v1"}],
            "target_ocr": {"family": "parseq", "checkpoint": "target"},
            "secondary_ocr": None,
            "validation_text": {"family": "local_contrast_validation", "checkpoint": "text"},
        },
    }

    audited = audit_v4_record(record)

    assert audited["status"] == "manual_review"
    assert "missing_secondary_ocr_gate" in audited["reject_reasons"]
    assert record["status"] == "accepted"


def test_v4_pilot_fails_closed_when_required_appearance_layout_slice_is_missing(tmp_path: Path) -> None:
    manifest = tmp_path / "accepted.jsonl"
    record = {
        "pipeline_version": V4_PIPELINE_VERSION,
        "record_id": "generated-blue-long",
        "task": "ocr_template_synthesis",
        "status": "accepted",
        "reject_reasons": [],
        "appearance_class": "state_blue_white",
        "layout": "long",
        "quality_metrics": {},
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    summary = summarize_v4_accepted_slices(
        manifest,
        required_slices=(("state_blue_white", "long"), ("military_red_light", "long")),
        target_per_slice=1,
    )
    decision = evaluate_v4_promotion_gate(
        V4PromotionMetrics(
            exact_ocr_match_rate=0.99,
            ghost_source_text_rate=0.0,
            visual_pass_rate=0.99,
            real_vn_hard_validation_delta=0.01,
            clean_validation_delta=0.0,
            manual_blank_ghost_count=0,
            manual_generated_ghost_count=0,
        ),
        acceptance_summary=summary,
    )

    assert summary.deficits == {"military_red_light/long": 1}
    assert not decision.promote
    assert "required_appearance_layout_slices_missing" in decision.reasons
