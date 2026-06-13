from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from synthetic_vn_lplc.reference_pilot import (
    PromotionMetrics,
    calibrate_review_thresholds,
    evaluate_promotion_gate,
    select_stratified_review_records,
    summarize_accepted_layouts,
)
from synthetic_vn_lplc.reference_schema import PIPELINE_VERSION


def test_pilot_acceptance_summary_reports_layout_deficits(tmp_path: Path) -> None:
    manifest = tmp_path / "accepted.jsonl"
    _write_manifest(
        manifest,
        [
            _record("long-1", "long"),
            _record("long-2", "long"),
            _record("short-1", "short"),
            _record("motor-rejected", "motor", filter_status="rejected"),
        ],
    )

    summary = summarize_accepted_layouts(manifest, target_per_layout=2)

    assert summary.counts == {"long": 2, "short": 1, "motor": 0}
    assert summary.deficits == {"short": 1, "motor": 2}
    assert not summary.is_balanced


def test_stratified_review_selection_is_deterministic_and_layout_balanced(tmp_path: Path) -> None:
    manifest = tmp_path / "accepted.jsonl"
    records = []
    for layout in ("long", "short", "motor"):
        for index, (time, rain, confidence) in enumerate(
            [
                ("day", False, 0.92),
                ("day", True, 0.74),
                ("night", False, 0.68),
                ("night", True, 0.51),
            ]
        ):
            records.append(_record(f"{layout}-{index}", layout, time=time, rain=rain, confidence=confidence))
    _write_manifest(manifest, records)

    first = select_stratified_review_records(manifest, sample_size=6, seed=13)
    second = select_stratified_review_records(manifest, sample_size=6, seed=13)

    assert [record["record_id"] for record in first] == [record["record_id"] for record in second]
    assert Counter(record["layout"] for record in first) == {"long": 2, "short": 2, "motor": 2}
    assert {record["review_stratum"]["layout"] for record in first} == {"long", "short", "motor"}
    assert {record["review_stratum"]["legibility"] for record in first} <= {"high", "medium", "hard"}


def test_promotion_gate_requires_visual_style_and_ocr_guardrails() -> None:
    passing = PromotionMetrics(
        visual_pass_rate=0.88,
        gray_blob_artifact_rate=0.01,
        ghost_brazil_text_rate=0.005,
        style_score_deltas=(0.03, 0.04, 0.05, 0.02, 0.06),
        real_vn_hard_validation_delta=0.012,
        clean_validation_delta=-0.004,
    )

    pass_decision = evaluate_promotion_gate(passing, bootstrap_iterations=200, seed=7)

    assert pass_decision.promote
    assert pass_decision.reasons == ()
    assert pass_decision.style_delta_ci[0] > 0

    failing = PromotionMetrics(
        visual_pass_rate=0.84,
        gray_blob_artifact_rate=0.03,
        ghost_brazil_text_rate=0.02,
        style_score_deltas=(-0.03, -0.02, 0.0, -0.01),
        real_vn_hard_validation_delta=-0.001,
        clean_validation_delta=-0.02,
    )

    fail_decision = evaluate_promotion_gate(failing, bootstrap_iterations=200, seed=7)

    assert not fail_decision.promote
    assert "visual_pass_rate_below_0.85" in fail_decision.reasons
    assert "gray_blob_artifact_rate_above_0.02" in fail_decision.reasons
    assert "ghost_brazil_text_rate_above_0.01" in fail_decision.reasons
    assert "style_score_not_better_than_baseline" in fail_decision.reasons
    assert "real_vn_hard_validation_not_improved" in fail_decision.reasons
    assert "clean_validation_drop_above_0.01" in fail_decision.reasons


def test_calibration_recommends_thresholds_from_reviewed_artifacts(tmp_path: Path) -> None:
    review_manifest = tmp_path / "reviewed.jsonl"
    records = []
    for index in range(10):
        records.append(
            _record(
                f"good-{index}",
                "long",
                metrics={
                    "ip_adapter_embedding_cosine": 0.80 + index * 0.005,
                    "masked_lpips_similarity": 0.72 + index * 0.004,
                    "gray_blob_fraction": 0.01 + index * 0.001,
                    "inpaint_residual_seam_score": 6.0 + index * 0.2,
                },
                manual_review={"visual_good": True, "artifact_clear": False},
            )
        )
    for index in range(10):
        records.append(
            _record(
                f"artifact-{index}",
                "short",
                metrics={
                    "ip_adapter_embedding_cosine": 0.10 + index * 0.004,
                    "masked_lpips_similarity": 0.20 + index * 0.004,
                    "gray_blob_fraction": 0.21 + index * 0.01,
                    "inpaint_residual_seam_score": 36.0 + index,
                },
                manual_review={"visual_good": False, "artifact_clear": True},
            )
        )
    _write_manifest(review_manifest, records)

    calibration = calibrate_review_thresholds(review_manifest, min_good_retention=0.90, min_artifact_rejection=0.90)

    assert calibration.metrics["ip_adapter_embedding_cosine"].satisfied
    assert calibration.metrics["ip_adapter_embedding_cosine"].threshold >= 0.80
    assert calibration.metrics["masked_lpips_similarity"].satisfied
    assert calibration.metrics["gray_blob_fraction"].satisfied
    assert calibration.metrics["gray_blob_fraction"].threshold <= 0.03
    assert calibration.metrics["inpaint_residual_seam_score"].satisfied
    assert calibration.to_config_patch()["reference_v3"]["filter"]["min_ip_adapter_embedding_cosine"] >= 0.80
    assert calibration.to_config_patch()["reference_v3"]["filter"]["max_gray_blob_fraction"] <= 0.03


def _record(
    record_id: str,
    layout: str,
    *,
    filter_status: str = "accepted",
    time: str = "day",
    rain: bool = False,
    confidence: float = 0.9,
    metrics: dict[str, float] | None = None,
    manual_review: dict[str, bool] | None = None,
) -> dict[str, object]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "record_id": record_id,
        "layout": layout,
        "label": "51A-1234",
        "filter_status": filter_status,
        "metadata": {"time": time, "rain": rain},
        "quality_metrics": {"camera_ocr_confidence": confidence, **(metrics or {})},
        "manual_review": manual_review or {"visual_good": True, "artifact_clear": False},
    }


def _write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
