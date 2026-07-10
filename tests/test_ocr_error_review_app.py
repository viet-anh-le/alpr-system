from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "review_ocr_error_audit.py"
SPEC = importlib.util.spec_from_file_location("review_ocr_error_audit", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_errors_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "path",
        "gt",
        "pred",
        "global_pred",
        "layout",
        "categories",
        "image_flags",
        "edit_distance",
        "pred_valid_format",
        "global_was_correct",
        "width",
        "height",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_review_store_marks_bad_samples_and_exports_relative_exclude_list(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ocr"
    image_path = dataset_root / "valid" / "51A-123.45#1.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not-an-image")
    state_path = tmp_path / "review_state.json"

    store = MODULE.ReviewStore(state_path)
    store.mark(
        str(image_path),
        action="bad_crop",
        note="cropped character",
        gt="51A-123.45",
        pred="51A-123.4",
    )
    store.mark(
        str(dataset_root / "valid" / "keep.jpg"),
        action="keep",
        note="model mistake",
    )

    output_path = tmp_path / "exclude_paths.txt"
    count = store.export_exclude_list(output_path, dataset_root=dataset_root)

    assert count == 1
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "valid/51A-123.45#1.jpg"
    ]
    reloaded = MODULE.ReviewStore(state_path)
    assert reloaded.records[str(image_path)]["action"] == "bad_crop"


def test_load_error_records_applies_status_and_category_filters(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ocr"
    bad_path = dataset_root / "valid" / "bad.jpg"
    keep_path = dataset_root / "valid" / "keep.jpg"
    errors_csv = tmp_path / "errors.csv"
    _write_errors_csv(
        errors_csv,
        [
            {
                "path": str(bad_path),
                "gt": "51A-123.45",
                "pred": "51A-123.4",
                "global_pred": "51A-123.45",
                "layout": "one_line",
                "categories": "line_decode_regression|length_error",
                "image_flags": "low_res",
                "edit_distance": "1",
                "pred_valid_format": "False",
                "global_was_correct": "True",
                "width": "40",
                "height": "16",
            },
            {
                "path": str(keep_path),
                "gt": "29A[SEP]123.45",
                "pred": "29A[SEP]123.46",
                "global_pred": "29A[SEP]123.46",
                "layout": "two_line",
                "categories": "digit_digit_confusion",
                "image_flags": "",
                "edit_distance": "1",
                "pred_valid_format": "True",
                "global_was_correct": "False",
                "width": "96",
                "height": "48",
            },
        ],
    )
    store = MODULE.ReviewStore(tmp_path / "state.json")
    store.mark(str(keep_path), action="keep")

    records = MODULE.load_error_records(errors_csv, store=store)
    pending = MODULE.filter_records(records, status="pending", category="line_decode_regression")

    assert [record.path for record in pending] == [str(bad_path)]
    assert pending[0].review_action == ""
    assert pending[0].priority >= records[1].priority


def test_apply_review_actions_quarantines_bad_samples_and_renames_fix_labels(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ocr"
    valid_dir = dataset_root / "valid"
    valid_dir.mkdir(parents=True)
    bad_image = valid_dir / "60-Y3[SEP]0756#bad.jpg"
    fix_image = valid_dir / "79-C1[SEP]148.11#nomer13091114.jpg"
    fix_sidecar = fix_image.with_suffix(".txt")
    bad_image.write_bytes(b"bad")
    fix_image.write_bytes(b"fix")
    fix_sidecar.write_text("sidecar", encoding="utf-8")

    state_path = tmp_path / "review_state.json"
    store = MODULE.ReviewStore(state_path)
    store.mark(str(bad_image), action="bad_crop")
    store.mark(str(fix_image), action="fix_label", corrected_label="79A-125.45")

    dry_run = MODULE.apply_review_actions(
        store,
        dataset_root=dataset_root,
        quarantine_dir=dataset_root / "_review_removed",
        dry_run=True,
    )

    assert {operation["action"] for operation in dry_run} == {"quarantine", "rename"}
    assert bad_image.exists()
    assert fix_image.exists()

    applied = MODULE.apply_review_actions(
        store,
        dataset_root=dataset_root,
        quarantine_dir=dataset_root / "_review_removed",
        dry_run=False,
    )

    renamed_image = valid_dir / "79A-125.45#nomer13091114.jpg"
    assert len(applied) == 2
    assert not bad_image.exists()
    assert (dataset_root / "_review_removed" / "valid" / bad_image.name).read_bytes() == b"bad"
    assert not fix_image.exists()
    assert renamed_image.read_bytes() == b"fix"
    assert renamed_image.with_suffix(".txt").read_text(encoding="utf-8") == "sidecar"
