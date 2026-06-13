from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "collect_platesmania_vn_dataset.py"
SPEC = importlib.util.spec_from_file_location("collect_platesmania_vn_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _image_bytes(width: int = 80, height: int = 40) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (40, 90, 160)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return bytes(encoded)


def _write_image(path: Path, width: int = 160, height: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (20, 80, 120)
    assert cv2.imwrite(str(path), image)


@pytest.mark.unit
def test_parse_gallery_html_extracts_full_frame_and_plate_text_without_download_queue() -> None:
    html = """
    <div class="panel">
      <a href="/vn/nomer32392050">
        <img src="https://img03.platesmania.com/260605/o/3239205044675.jpg" alt="Honda Dream">
      </a>
      <a href="/vn/nomer32392050">
        <img src="https://img03.platesmania.com/260605/inf/3239205044675.png" alt="50E-190.54">
      </a>
    </div>
    """

    records = MODULE.parse_gallery_html(html, page_url="https://platesmania.com/vn/gallery-10")

    assert len(records) == 1
    record = records[0]
    assert record.record_id == "nomer32392050"
    assert record.detail_url == "https://platesmania.com/vn/nomer32392050"
    assert record.vehicle_image_url == "https://img03.platesmania.com/260605/o/3239205044675.jpg"
    assert record.plate_text_raw == "50E-190.54"
    assert record.plate_text_normalized == "50E-190.54"
    assert record.plate_ref_url == "https://img03.platesmania.com/260605/inf/3239205044675.png"
    assert MODULE.urls_to_download(record) == [record.vehicle_image_url]


@pytest.mark.unit
def test_parse_gallery_html_preserves_two_line_plate_space() -> None:
    html = """
    <a href="/vn/nomer1">
      <img src="https://img03.platesmania.com/260603/m/32366790.jpg" alt="">
    </a>
    <a href="/vn/nomer1">
      <img src="https://img03.platesmania.com/260603/inf/32366790.png" alt="84-L1 293.38">
    </a>
    """

    records = MODULE.parse_gallery_html(html, page_url="https://platesmania.com/vn/gallery")

    assert records[0].plate_text_normalized == "84-L1 293.38"


@pytest.mark.unit
def test_gallery_url_builder_uses_zero_based_gallery_index() -> None:
    assert MODULE.build_gallery_url(0) == "https://platesmania.com/vn/gallery"
    assert MODULE.build_gallery_url(10) == "https://platesmania.com/vn/gallery-10"


@pytest.mark.unit
def test_province_search_url_builder_uses_nomer_and_start_params() -> None:
    assert (
        MODULE.build_province_search_url(11, 0)
        == "https://platesmania.com/vn/gallery.php?&nomer=11&start=0"
    )
    assert (
        MODULE.build_province_search_url(99, 100)
        == "https://platesmania.com/vn/gallery.php?&nomer=99&start=100"
    )


@pytest.mark.unit
def test_collect_records_from_html_directory(tmp_path: Path) -> None:
    html_dir = tmp_path / "html_pages"
    html_dir.mkdir()
    (html_dir / "gallery-10.html").write_text(
        """
        <a href="/vn/nomer32392050">
          <img src="https://img03.platesmania.com/260605/o/3239205044675.jpg" alt="Honda Dream">
        </a>
        <a href="/vn/nomer32392050">
          <img src="https://img03.platesmania.com/260605/inf/3239205044675.png" alt="50E-190.54">
        </a>
        """,
        encoding="utf-8",
    )
    args = SimpleNamespace(
        source="html",
        html_dir=html_dir,
        all_vietnam=False,
        max_records=None,
    )

    records = MODULE.collect_records_from_sources(args)

    assert [record.record_id for record in records] == ["nomer32392050"]


@pytest.mark.unit
def test_collect_records_dedupes_same_vehicle_image_url_from_jsonl(tmp_path: Path) -> None:
    html_dir = tmp_path / "html_pages"
    html_dir.mkdir()
    (html_dir / "gallery_records.jsonl").write_text(
        "\n".join(
            [
                (
                    '{"record_id":"nomer1","page_url":"https://platesmania.com/vn/gallery.php?&nomer=11&start=0",'
                    '"detail_url":"https://platesmania.com/vn/nomer1",'
                    '"vehicle_image_url":"https://img03.platesmania.com/260605/m/same.jpg",'
                    '"plate_text_raw":"11A-123.45","plate_text_normalized":"11A-123.45"}'
                ),
                (
                    '{"record_id":"nomer2","page_url":"https://platesmania.com/vn/gallery.php?&nomer=12&start=0",'
                    '"detail_url":"https://platesmania.com/vn/nomer2",'
                    '"vehicle_image_url":"https://img03.platesmania.com/260605/m/same.jpg",'
                    '"plate_text_raw":"12A-123.45","plate_text_normalized":"12A-123.45"}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = MODULE.collect_records_from_sources(
        SimpleNamespace(source="html", html_dir=html_dir, all_vietnam=False, max_records=None)
    )

    assert len(records) == 1
    assert records[0].record_id == "nomer1"


@pytest.mark.unit
def test_collect_records_from_exported_jsonl(tmp_path: Path) -> None:
    html_dir = tmp_path / "html_pages"
    html_dir.mkdir()
    (html_dir / "gallery_records.jsonl").write_text(
        (
            '{"record_id":"nomer32392050","page_url":"https://platesmania.com/vn/gallery-10",'
            '"detail_url":"https://platesmania.com/vn/nomer32392050",'
            '"vehicle_image_url":"https://img03.platesmania.com/260605/o/3239205044675.jpg",'
            '"plate_ref_url":"https://img03.platesmania.com/260605/inf/3239205044675.png",'
            '"plate_text_raw":"50E-190.54","plate_text_normalized":"50E-190.54"}\n'
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        source="html",
        html_dir=html_dir,
        all_vietnam=False,
        max_records=None,
    )

    records = MODULE.collect_records_from_sources(args)

    assert len(records) == 1
    assert records[0].vehicle_image_url.endswith("3239205044675.jpg")


@pytest.mark.unit
def test_collect_records_from_exported_jsonl_preserves_space(tmp_path: Path) -> None:
    html_dir = tmp_path / "html_pages"
    html_dir.mkdir()
    (html_dir / "gallery_records.jsonl").write_text(
        (
            '{"record_id":"nomer1","page_url":"https://platesmania.com/vn/gallery",'
            '"detail_url":"https://platesmania.com/vn/nomer1",'
            '"vehicle_image_url":"https://img03.platesmania.com/260603/m/32366790.jpg",'
            '"plate_ref_url":"https://img03.platesmania.com/260603/inf/32366790.png",'
            '"plate_text_raw":"84-L1 293.38","plate_text_normalized":"84-L1 293.38"}\n'
        ),
        encoding="utf-8",
    )

    records = MODULE.collect_records_from_sources(
        SimpleNamespace(source="html", html_dir=html_dir, all_vietnam=False, max_records=None)
    )

    assert records[0].plate_text_normalized == "84-L1 293.38"


@pytest.mark.unit
def test_records_to_compact_html_round_trips_without_unrelated_markup() -> None:
    record = MODULE.PlateRecord(
        record_id="nomer32392050",
        page_url="https://platesmania.com/vn/gallery-10",
        detail_url="https://platesmania.com/vn/nomer32392050",
        vehicle_image_url="https://img03.platesmania.com/260605/o/3239205044675.jpg",
        plate_ref_url="https://img03.platesmania.com/260605/inf/3239205044675.png",
        plate_text_raw="50E-190.54",
        plate_text_normalized="50E-190.54",
    )

    compact = MODULE.records_to_compact_html([record], page_url=record.page_url)
    parsed = MODULE.parse_gallery_html(compact, page_url=record.page_url)

    assert "unrelated" not in compact
    assert parsed == [record]


@pytest.mark.unit
def test_download_gallery_html_pages_saves_compact_html(tmp_path: Path) -> None:
    html_dir = tmp_path / "html_pages"
    page_html = """
    <main><p>unrelated page chrome</p>
      <a href="/vn/nomer32392050">
        <img src="https://img03.platesmania.com/260605/o/3239205044675.jpg" alt="">
      </a>
      <a href="/vn/nomer32392050">
        <img src="https://img03.platesmania.com/260605/inf/3239205044675.png" alt="50E-190.54">
      </a>
    </main>
    """
    args = SimpleNamespace(
        html_dir=html_dir,
        start_index=10,
        end_index=10,
        all_vietnam=False,
        max_pages=1,
        max_records=None,
        html_save_mode="compact",
        delay=0.0,
        timeout=1.0,
    )

    summary = MODULE.download_gallery_html_pages(args, fetcher=lambda url, timeout: page_html)

    saved = html_dir / "gallery-10.html"
    assert summary == {"pages": 1, "records": 1}
    assert saved.exists()
    assert "unrelated page chrome" not in saved.read_text(encoding="utf-8")
    assert MODULE.collect_records_from_sources(
        SimpleNamespace(source="html", html_dir=html_dir, all_vietnam=False, max_records=None)
    )[0].plate_text_normalized == "50E-190.54"


@pytest.mark.unit
def test_prepare_output_dir_refuses_to_overwrite_nested_html_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "platesmania_vn"
    html_dir = output_dir / "html_pages"
    html_dir.mkdir(parents=True)
    args = SimpleNamespace(
        overwrite=True,
        resume=False,
        source="html",
        html_dir=html_dir,
    )

    with pytest.raises(SystemExit, match="Refusing --overwrite"):
        MODULE.prepare_output_dir(args, output_dir)

    assert html_dir.exists()


@pytest.mark.unit
def test_download_vehicle_image_writes_only_full_frame(tmp_path: Path) -> None:
    record = MODULE.PlateRecord(
        record_id="nomer1",
        page_url="https://platesmania.com/vn/gallery",
        detail_url="https://platesmania.com/vn/nomer1",
        vehicle_image_url="https://img03.platesmania.com/vehicle.jpg",
        plate_ref_url="https://img03.platesmania.com/inf/generated.png",
        plate_text_raw="50E-190.54",
        plate_text_normalized="50E-190.54",
    )

    def fetcher(url: str, timeout: float) -> bytes:
        assert url == record.vehicle_image_url
        assert timeout > 0
        return _image_bytes()

    image_path = MODULE.download_vehicle_image(record, tmp_path, fetcher=fetcher)

    assert image_path == tmp_path / "downloads" / "full_frames" / "nomer1.jpg"
    assert image_path.exists()
    assert not (tmp_path / "downloads" / "source_plate_refs").exists()


@pytest.mark.unit
def test_write_ocr_sample_preserves_dot_and_dash_label(tmp_path: Path) -> None:
    crop = np.zeros((24, 96, 3), dtype=np.uint8)
    record = MODULE.PlateRecord(
        record_id="nomer1",
        page_url="https://platesmania.com/vn/gallery",
        detail_url="https://platesmania.com/vn/nomer1",
        vehicle_image_url="https://img03.platesmania.com/vehicle.jpg",
        plate_ref_url=None,
        plate_text_raw="50E-190.54",
        plate_text_normalized="50E-190.54",
    )

    image_path, label_path = MODULE.write_ocr_sample(tmp_path, record, crop, split="train")

    assert image_path == tmp_path / "ocr" / "images" / "train" / "nomer1.jpg"
    assert label_path == tmp_path / "ocr" / "labels" / "train" / "nomer1.txt"
    assert image_path.exists()
    assert label_path.read_text(encoding="utf-8") == "50E-190.54\n"


@pytest.mark.unit
def test_write_detection_sample_creates_yolo_obb_label_and_data_yaml(tmp_path: Path) -> None:
    source_image = tmp_path / "source.jpg"
    _write_image(source_image, width=200, height=100)
    detection = MODULE.PlateDetection(
        confidence=0.91,
        class_id=0,
        class_name="plate",
        box_xyxy=(20, 30, 80, 50),
        points=((20, 30), (80, 30), (80, 50), (20, 50)),
    )
    record = MODULE.PlateRecord(
        record_id="nomer1",
        page_url="https://platesmania.com/vn/gallery",
        detail_url="https://platesmania.com/vn/nomer1",
        vehicle_image_url="https://img03.platesmania.com/vehicle.jpg",
        plate_ref_url=None,
        plate_text_raw="50E-190.54",
        plate_text_normalized="50E-190.54",
        vehicle_image_path=source_image,
    )

    image_path, label_path = MODULE.write_detection_sample(
        tmp_path,
        record,
        source_image,
        [detection],
        split="val",
    )
    MODULE.write_detection_yaml(tmp_path, names={0: "plate"})

    assert image_path == tmp_path / "detection" / "images" / "val" / "nomer1.jpg"
    assert label_path == tmp_path / "detection" / "labels" / "val" / "nomer1.txt"
    tokens = label_path.read_text(encoding="utf-8").strip().split()
    assert tokens[0] == "0"
    assert len(tokens) == 9
    assert all(0.0 <= float(token) <= 1.0 for token in tokens[1:])
    assert "names:" in (tmp_path / "detection" / "data.yaml").read_text(encoding="utf-8")


@pytest.mark.unit
def test_low_confidence_review_keeps_full_frame_instead_of_crop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "platesmania_vn"
    full_frame = output_dir / "downloads" / "full_frames" / "nomer1.jpg"
    _write_image(full_frame, width=200, height=100)
    detection = MODULE.PlateDetection(
        confidence=0.25,
        class_id=0,
        class_name="plate",
        box_xyxy=(20, 30, 80, 50),
        points=((20, 30), (80, 30), (80, 50), (20, 50)),
    )
    monkeypatch.setattr(MODULE, "detect_plates", lambda frame, model, conf, imgsz: [detection])
    record = MODULE.PlateRecord(
        record_id="nomer1",
        page_url="https://platesmania.com/vn/gallery",
        detail_url="https://platesmania.com/vn/nomer1",
        vehicle_image_url="https://img03.platesmania.com/vehicle.jpg",
        plate_ref_url=None,
        plate_text_raw="50E-190.54",
        plate_text_normalized="50E-190.54",
    )
    args = SimpleNamespace(
        timeout=1.0,
        overwrite=False,
        skip_detection=False,
        plate_conf=0.1,
        imgsz=640,
        val_ratio=0.0,
        seed=42,
        accept_conf=0.5,
    )

    processed = MODULE.process_record(record, output_dir, plate_model=object(), args=args)

    review_image = output_dir / "review" / "pending_review" / "nomer1.jpg"
    saved = cv2.imread(str(review_image))
    assert processed.status == "pending_review"
    assert processed.review_reason == "low_detector_confidence"
    assert saved.shape[:2] == (100, 200)
