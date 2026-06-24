from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "ddl_label_review_app.py"
SPEC = importlib.util.spec_from_file_location("ddl_label_review_app", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_image(path: Path, *, width: int = 120, height: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(245, 245, 245)).save(path)


@pytest.mark.unit
def test_label_conversion_for_ddl_pattern() -> None:
    assert MODULE.is_review_label("79A-179.19")
    assert MODULE.is_review_label("79A[SEP]179.19")
    assert MODULE.to_two_line_label("79A-179.19") == "79A[SEP]179.19"
    assert MODULE.to_one_line_label("79A[SEP]179.19") == "79A-179.19"


@pytest.mark.unit
def test_suggest_label_uses_ratio_threshold_only_for_unconverted_ddl_labels() -> None:
    assert MODULE.suggest_label("79A-179.19", width=80, height=80, threshold=1.97) == "79A[SEP]179.19"
    assert MODULE.suggest_label("79A-179.19", width=220, height=50, threshold=1.97) == "79A-179.19"
    assert MODULE.suggest_label("79A[SEP]179.19", width=220, height=50, threshold=1.97) == "79A[SEP]179.19"


@pytest.mark.unit
def test_build_filename_with_label_preserves_source_id_and_extension() -> None:
    new_name = MODULE.build_filename_with_label(
        Path("79A-179.19#nomer31555222.jpg"),
        "79A[SEP]179.19",
    )

    assert new_name == "79A[SEP]179.19#nomer31555222.jpg"


@pytest.mark.unit
def test_resolve_dataset_path_rejects_path_traversal(tmp_path: Path) -> None:
    dataset = tmp_path / "ocr"
    dataset.mkdir()

    with pytest.raises(MODULE.ValidationError):
        MODULE.resolve_dataset_path(dataset, "../outside.jpg")


@pytest.mark.unit
def test_scan_includes_original_and_converted_ddl_samples(tmp_path: Path) -> None:
    dataset = tmp_path / "ocr"
    _write_image(dataset / "train" / "79A-179.19#one.jpg", width=220, height=50)
    _write_image(dataset / "valid" / "79A[SEP]179.19#two.jpg", width=80, height=80)
    _write_image(dataset / "train" / "59-U1[SEP]027.95#skip.jpg", width=80, height=80)

    service = MODULE.DatasetReviewService(dataset_dir=dataset, threshold=1.97)
    result = service.list_samples(page=1, page_size=20)

    assert result["total"] == 2
    assert [item["label"] for item in result["items"]] == [
        "79A-179.19",
        "79A[SEP]179.19",
    ]
    assert result["items"][0]["two_line_label"] == "79A[SEP]179.19"
    assert result["items"][1]["one_line_label"] == "79A-179.19"


@pytest.mark.unit
def test_rename_sample_preserves_suffix_and_txt_sidecar(tmp_path: Path) -> None:
    dataset = tmp_path / "ocr"
    image_path = dataset / "train" / "79A-179.19#nomer.jpg"
    txt_path = dataset / "train" / "79A-179.19#nomer.txt"
    _write_image(image_path, width=80, height=80)
    txt_path.write_text("sidecar", encoding="utf-8")

    service = MODULE.DatasetReviewService(dataset_dir=dataset, threshold=1.97)
    renamed = service.rename_sample("train/79A-179.19#nomer.jpg", "79A[SEP]179.19")

    assert renamed["reviewed"] is True
    assert renamed["rel_path"] == "train/79A[SEP]179.19#nomer.jpg"
    assert not image_path.exists()
    assert (dataset / "train" / "79A[SEP]179.19#nomer.jpg").exists()
    assert not txt_path.exists()
    assert (dataset / "train" / "79A[SEP]179.19#nomer.txt").read_text(encoding="utf-8") == "sidecar"


@pytest.mark.unit
def test_rename_sample_can_convert_two_line_back_to_one_line(tmp_path: Path) -> None:
    dataset = tmp_path / "ocr"
    image_path = dataset / "valid" / "79A[SEP]179.19#nomer.jpg"
    _write_image(image_path, width=80, height=80)

    service = MODULE.DatasetReviewService(dataset_dir=dataset, threshold=1.97)
    renamed = service.rename_sample("valid/79A[SEP]179.19#nomer.jpg", "79A-179.19")

    assert renamed["reviewed"] is True
    assert renamed["label"] == "79A-179.19"
    assert renamed["rel_path"] == "valid/79A-179.19#nomer.jpg"
    assert not image_path.exists()
    assert (dataset / "valid" / "79A-179.19#nomer.jpg").exists()


@pytest.mark.unit
def test_rename_sample_reports_noop_when_label_is_already_current(tmp_path: Path) -> None:
    dataset = tmp_path / "ocr"
    image_path = dataset / "train" / "79A-179.19#nomer.jpg"
    _write_image(image_path, width=220, height=50)

    service = MODULE.DatasetReviewService(dataset_dir=dataset, threshold=1.97)
    renamed = service.rename_sample("train/79A-179.19#nomer.jpg", "79A-179.19")

    assert renamed["reviewed"] is True
    assert renamed["label"] == "79A-179.19"
    assert renamed["rel_path"] == "train/79A-179.19#nomer.jpg"
    assert image_path.exists()
    assert service.list_samples(page=1, page_size=20)["total"] == 0
    assert service.list_samples(page=1, page_size=20, status="reviewed")["total"] == 1


@pytest.mark.unit
def test_reviewed_samples_are_hidden_by_default_and_persist_across_runs(tmp_path: Path) -> None:
    dataset = tmp_path / "ocr"
    _write_image(dataset / "train" / "79A-179.19#one.jpg", width=220, height=50)
    _write_image(dataset / "train" / "30L-762.20#two.jpg", width=80, height=80)

    service = MODULE.DatasetReviewService(dataset_dir=dataset, threshold=1.97)
    reviewed = service.rename_sample("train/79A-179.19#one.jpg", "79A-179.19")

    assert reviewed["reviewed"] is True
    assert service.list_samples(page=1, page_size=20)["total"] == 1
    assert service.list_samples(page=1, page_size=20, status="reviewed")["total"] == 1

    reloaded = MODULE.DatasetReviewService(dataset_dir=dataset, threshold=1.97)

    assert reloaded.list_samples(page=1, page_size=20)["total"] == 1
    assert reloaded.list_samples(page=1, page_size=20, status="reviewed")["items"][0]["rel_path"] == (
        "train/79A-179.19#one.jpg"
    )


@pytest.mark.unit
def test_reviewed_state_tracks_renamed_final_path(tmp_path: Path) -> None:
    dataset = tmp_path / "ocr"
    _write_image(dataset / "valid" / "79A-179.19#nomer.jpg", width=80, height=80)

    service = MODULE.DatasetReviewService(dataset_dir=dataset, threshold=1.97)
    reviewed = service.rename_sample("valid/79A-179.19#nomer.jpg", "79A[SEP]179.19")

    assert reviewed["reviewed"] is True
    assert reviewed["rel_path"] == "valid/79A[SEP]179.19#nomer.jpg"

    reloaded = MODULE.DatasetReviewService(dataset_dir=dataset, threshold=1.97)

    assert reloaded.list_samples(page=1, page_size=20)["total"] == 0
    assert reloaded.list_samples(page=1, page_size=20, status="reviewed")["items"][0]["rel_path"] == (
        "valid/79A[SEP]179.19#nomer.jpg"
    )


@pytest.mark.unit
def test_bulk_apply_suggestions_keeps_renamed_samples_pending(tmp_path: Path) -> None:
    dataset = tmp_path / "ocr"
    _write_image(dataset / "train" / "79A-179.19#needs-two.jpg", width=80, height=80)
    _write_image(dataset / "train" / "30L-762.20#already-one.jpg", width=220, height=50)

    service = MODULE.DatasetReviewService(dataset_dir=dataset, threshold=1.97)
    result = service.bulk_apply_suggestions(
        [
            "train/79A-179.19#needs-two.jpg",
            "train/30L-762.20#already-one.jpg",
        ]
    )

    assert result["success"] is True
    assert [item["rel_path"] for item in result["renamed"]] == [
        "train/79A[SEP]179.19#needs-two.jpg"
    ]
    assert result["renamed"][0]["reviewed"] is False
    assert not (dataset / "train" / "79A-179.19#needs-two.jpg").exists()
    assert (dataset / "train" / "79A[SEP]179.19#needs-two.jpg").exists()
    assert service.list_samples(page=1, page_size=20)["total"] == 2
    assert service.list_samples(page=1, page_size=20, status="reviewed")["total"] == 0

    reloaded = MODULE.DatasetReviewService(dataset_dir=dataset, threshold=1.97)

    assert reloaded.list_samples(page=1, page_size=20)["total"] == 2
    assert reloaded.list_samples(page=1, page_size=20, status="reviewed")["total"] == 0


@pytest.mark.unit
def test_mark_reviewed_marks_visible_page_paths_and_persists(tmp_path: Path) -> None:
    dataset = tmp_path / "ocr"
    _write_image(dataset / "train" / "30L-762.20#first.jpg", width=220, height=50)
    _write_image(dataset / "train" / "79A[SEP]179.19#second.jpg", width=80, height=80)

    service = MODULE.DatasetReviewService(dataset_dir=dataset, threshold=1.97)
    first_page = service.list_samples(page=1, page_size=1, sort="path")
    first_path = first_page["items"][0]["rel_path"]
    result = service.mark_reviewed([first_path])

    assert result["reviewed"] == [first_path]
    assert result["skipped"] == []
    assert service.list_samples(page=1, page_size=20)["total"] == 1
    assert service.list_samples(page=1, page_size=20, status="reviewed")["total"] == 1

    reloaded = MODULE.DatasetReviewService(dataset_dir=dataset, threshold=1.97)

    assert reloaded.list_samples(page=1, page_size=20)["total"] == 1
    assert reloaded.list_samples(page=1, page_size=20, status="reviewed")["items"][0]["rel_path"] == first_path


@pytest.mark.unit
def test_rename_sample_refuses_to_overwrite_existing_file(tmp_path: Path) -> None:
    dataset = tmp_path / "ocr"
    _write_image(dataset / "train" / "79A-179.19#nomer.jpg")
    _write_image(dataset / "train" / "79A[SEP]179.19#nomer.jpg")

    service = MODULE.DatasetReviewService(dataset_dir=dataset, threshold=1.97)

    with pytest.raises(MODULE.ConflictError):
        service.rename_sample("train/79A-179.19#nomer.jpg", "79A[SEP]179.19")
