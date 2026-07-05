from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_workbench_ocr_options_only_expose_default_line_ctc_and_vietnamese_yolov5() -> None:
    source = (ROOT / "web/src/components/workbench/constants.js").read_text(encoding="utf-8")
    match = re.search(r"export const OCR_OPTIONS = \[(.*?)\]", source, re.S)

    assert match is not None
    values = re.findall(r"value: '([^']+)'", match.group(1))
    labels = re.findall(r"label: '([^']+)'", match.group(1))

    assert values == ["default", "vietnamese_yolov5"]
    assert labels == ["SmallLPR-Line-CTC (mặc định)", "YOLOv5 Việt Nam"]


@pytest.mark.unit
def test_legacy_dropzone_uses_shared_ocr_options() -> None:
    source = (ROOT / "web/src/components/DropZone.jsx").read_text(encoding="utf-8")

    assert "const OCR_OPTIONS" not in source
    assert "from './workbench/constants'" in source
