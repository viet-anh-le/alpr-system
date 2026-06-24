"""
Review OCR error-audit samples and export an exclude list.

This tool is intentionally non-destructive: it records review decisions in a
JSON sidecar and exports paths that the training datamodule can skip.

Example:
    /home/vietanh/anaconda3/envs/myenv/bin/python scripts/review_ocr_error_audit.py serve \
        --errors-csv weights/ocr/small_lpr_line_ctc/line_ctc_reviewed_v1/error_audit/errors.csv

    /home/vietanh/anaconda3/envs/myenv/bin/python scripts/review_ocr_error_audit.py export \
        --state weights/ocr/small_lpr_line_ctc/line_ctc_reviewed_v1/error_audit/review_state.json \
        --dataset-root data/datasets/ocr \
        --out data/datasets/ocr/exclude_ocr_bad_samples.txt
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import re
import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ERRORS_CSV = (
    ROOT / "weights/ocr/small_lpr_line_ctc/line_ctc_reviewed_v1/error_audit/errors.csv"
)
DEFAULT_DATASET_ROOT = ROOT / "data/datasets/ocr"
DEFAULT_PAGE_SIZE = 80
MAX_PAGE_SIZE = 240
VALID_ACTIONS = {"keep", "bad_crop", "bad_label", "fix_label", "uncertain"}
DEFAULT_EXCLUDE_ACTIONS = {"bad_crop", "bad_label"}
SAFE_LABEL_RE = re.compile(r"^[0-9A-ZĐ.\-]+(?:\[SEP\][0-9A-ZĐ.\-]+)?$")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def split_pipe(value: str) -> tuple[str, ...]:
    return tuple(piece for piece in value.split("|") if piece)


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def parse_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def validate_corrected_label(label: str) -> str:
    normalized = label.strip().upper()
    if not normalized:
        raise ValueError("Corrected label is empty")
    if len(normalized) > 64:
        raise ValueError(f"Corrected label is too long: {label}")
    if any(char in normalized for char in ("#", "/", "\\", "\x00")):
        raise ValueError(f"Corrected label contains an invalid character: {label}")
    if not SAFE_LABEL_RE.fullmatch(normalized):
        raise ValueError(f"Corrected label contains unsupported characters: {label}")
    return normalized


def filename_suffix(path: Path) -> str:
    stem = path.stem
    return stem[stem.index("#") :] if "#" in stem else ""


def build_filename_with_label(path: Path, corrected_label: str) -> str:
    normalized = validate_corrected_label(corrected_label)
    return f"{normalized}{filename_suffix(path)}{path.suffix}"


def move_with_sidecar(source: Path, target: Path, *, dry_run: bool) -> None:
    if target.exists() and target.resolve() != source.resolve():
        raise FileExistsError(f"Target already exists: {target}")
    source_sidecar = source.with_suffix(".txt")
    target_sidecar = target.with_suffix(".txt")
    if source_sidecar.exists() and target_sidecar.exists() and target_sidecar.resolve() != source_sidecar.resolve():
        raise FileExistsError(f"Target sidecar already exists: {target_sidecar}")
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() != source.resolve():
        source.rename(target)
    if source_sidecar.exists() and target_sidecar.resolve() != source_sidecar.resolve():
        source_sidecar.rename(target_sidecar)


@dataclass(frozen=True)
class ErrorRecord:
    path: str
    gt: str
    pred: str
    global_pred: str
    layout: str
    categories: tuple[str, ...]
    image_flags: tuple[str, ...]
    edit_distance: float
    pred_valid_format: bool
    global_was_correct: bool
    width: int
    height: int
    review_action: str = ""
    review_note: str = ""
    corrected_label: str = ""

    @property
    def priority(self) -> float:
        score = self.edit_distance
        if self.image_flags:
            score += 20.0
        if "image_low_res" in self.categories or "low_res" in self.image_flags:
            score += 10.0
        if "image_blur_or_smooth" in self.categories or "blur_or_smooth" in self.image_flags:
            score += 8.0
        if not self.pred_valid_format:
            score += 6.0
        if "length_error" in self.categories:
            score += 3.0
        if "separator_error" in self.categories:
            score += 3.0
        if self.global_was_correct:
            score += 2.0
        return score

    @property
    def reviewed(self) -> bool:
        return bool(self.review_action)

    def to_item(self, *, dataset_root: Path | None = None) -> dict[str, Any]:
        image_path = Path(self.path)
        rel_path = self.path
        if dataset_root is not None:
            try:
                rel_path = image_path.resolve().relative_to(dataset_root.resolve()).as_posix()
            except ValueError:
                rel_path = self.path
        return {
            "path": self.path,
            "rel_path": rel_path,
            "filename": image_path.name,
            "gt": self.gt,
            "pred": self.pred,
            "global_pred": self.global_pred,
            "layout": self.layout,
            "categories": list(self.categories),
            "image_flags": list(self.image_flags),
            "edit_distance": self.edit_distance,
            "pred_valid_format": self.pred_valid_format,
            "global_was_correct": self.global_was_correct,
            "width": self.width,
            "height": self.height,
            "priority": self.priority,
            "review_action": self.review_action,
            "review_note": self.review_note,
            "corrected_label": self.corrected_label,
            "reviewed": self.reviewed,
        }


class ReviewStore:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.records: dict[str, dict[str, str]] = {}
        self.load()

    def load(self) -> None:
        if not self.state_path.exists():
            self.records = {}
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.records = {}
            return
        raw_records = payload.get("records", {}) if isinstance(payload, dict) else {}
        if not isinstance(raw_records, dict):
            self.records = {}
            return
        self.records = {
            str(path): {
                "action": str(record.get("action", "")),
                "note": str(record.get("note", "")),
                "corrected_label": str(record.get("corrected_label", "")),
                "gt": str(record.get("gt", "")),
                "pred": str(record.get("pred", "")),
            }
            for path, record in raw_records.items()
            if isinstance(record, dict)
        }

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "records": self.records,
        }
        tmp_path = self.state_path.with_name(f"{self.state_path.name}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.state_path)

    def mark(
        self,
        path: str,
        *,
        action: str,
        note: str = "",
        corrected_label: str = "",
        gt: str = "",
        pred: str = "",
    ) -> dict[str, str]:
        normalized = action.strip().lower()
        if normalized not in VALID_ACTIONS:
            raise ValueError(f"Invalid action: {action}")
        self.records[path] = {
            "action": normalized,
            "note": note.strip(),
            "corrected_label": corrected_label.strip().upper(),
            "gt": gt,
            "pred": pred,
        }
        self.save()
        return self.records[path]

    def clear(self, path: str) -> None:
        self.records.pop(path, None)
        self.save()

    def export_exclude_list(
        self,
        out_path: Path,
        *,
        dataset_root: Path,
        actions: set[str] | None = None,
    ) -> int:
        active_actions = actions or DEFAULT_EXCLUDE_ACTIONS
        lines: list[str] = []
        for path, record in sorted(self.records.items()):
            if record.get("action") not in active_actions:
                continue
            image_path = Path(path)
            try:
                line = image_path.resolve().relative_to(dataset_root.resolve()).as_posix()
            except ValueError:
                line = str(image_path)
            lines.append(line)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return len(lines)

    def counts(self) -> dict[str, int]:
        counts = {action: 0 for action in sorted(VALID_ACTIONS)}
        for record in self.records.values():
            action = record.get("action", "")
            if action in counts:
                counts[action] += 1
        counts["reviewed"] = sum(counts.values())
        counts["exclude"] = counts["bad_crop"] + counts["bad_label"]
        return counts


def apply_review_actions(
    store: ReviewStore,
    *,
    dataset_root: Path,
    quarantine_dir: Path,
    dry_run: bool,
) -> list[dict[str, str]]:
    operations: list[dict[str, str]] = []
    for path, record in sorted(store.records.items()):
        action = record.get("action", "")
        source = Path(path)
        if not source.exists():
            operations.append(
                {
                    "action": "missing",
                    "source": str(source),
                    "target": "",
                    "status": "skipped",
                }
            )
            continue
        if action in DEFAULT_EXCLUDE_ACTIONS:
            try:
                relative = source.resolve().relative_to(dataset_root.resolve())
            except ValueError:
                relative = Path(source.name)
            target = quarantine_dir / relative
            move_with_sidecar(source, target, dry_run=dry_run)
            operations.append(
                {
                    "action": "quarantine",
                    "source": str(source),
                    "target": str(target),
                    "status": "dry_run" if dry_run else "applied",
                }
            )
            continue
        if action == "fix_label":
            corrected_label = record.get("corrected_label", "")
            if not corrected_label:
                operations.append(
                    {
                        "action": "rename",
                        "source": str(source),
                        "target": "",
                        "status": "skipped_missing_corrected_label",
                    }
                )
                continue
            target = source.with_name(build_filename_with_label(source, corrected_label))
            move_with_sidecar(source, target, dry_run=dry_run)
            operations.append(
                {
                    "action": "rename",
                    "source": str(source),
                    "target": str(target),
                    "status": "dry_run" if dry_run else "applied",
                }
            )
    return operations


def load_error_records(errors_csv: Path, *, store: ReviewStore) -> list[ErrorRecord]:
    records: list[ErrorRecord] = []
    with errors_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = row.get("path", "")
            review = store.records.get(path, {})
            records.append(
                ErrorRecord(
                    path=path,
                    gt=row.get("gt", ""),
                    pred=row.get("pred", ""),
                    global_pred=row.get("global_pred", ""),
                    layout=row.get("layout", ""),
                    categories=split_pipe(row.get("categories", "")),
                    image_flags=split_pipe(row.get("image_flags", "")),
                    edit_distance=parse_float(row.get("edit_distance", "")),
                    pred_valid_format=parse_bool(row.get("pred_valid_format", "")),
                    global_was_correct=parse_bool(row.get("global_was_correct", "")),
                    width=parse_int(row.get("width", "")),
                    height=parse_int(row.get("height", "")),
                    review_action=review.get("action", ""),
                    review_note=review.get("note", ""),
                    corrected_label=review.get("corrected_label", ""),
                )
            )
    return sorted(records, key=lambda record: (-record.priority, record.path))


def filter_records(
    records: list[ErrorRecord],
    *,
    status: str = "pending",
    category: str = "all",
    layout: str = "all",
    query: str = "",
) -> list[ErrorRecord]:
    filtered = records
    if status == "pending":
        filtered = [record for record in filtered if not record.reviewed]
    elif status == "reviewed":
        filtered = [record for record in filtered if record.reviewed]
    elif status in VALID_ACTIONS:
        filtered = [record for record in filtered if record.review_action == status]
    if category != "all":
        filtered = [record for record in filtered if category in record.categories or category in record.image_flags]
    if layout != "all":
        filtered = [record for record in filtered if record.layout == layout]
    if query:
        needle = query.strip().upper()
        filtered = [
            record
            for record in filtered
            if needle in record.gt.upper()
            or needle in record.pred.upper()
            or needle in record.global_pred.upper()
            or needle in Path(record.path).name.upper()
        ]
    return filtered


def summarize(records: list[ErrorRecord], store: ReviewStore) -> dict[str, Any]:
    categories: dict[str, int] = {}
    layouts: dict[str, int] = {}
    for record in records:
        layouts[record.layout] = layouts.get(record.layout, 0) + 1
        for category in [*record.categories, *record.image_flags]:
            categories[category] = categories.get(category, 0) + 1
    return {
        "total": len(records),
        "pending": sum(1 for record in records if not record.reviewed),
        "review": store.counts(),
        "layouts": layouts,
        "categories": dict(sorted(categories.items(), key=lambda item: (-item[1], item[0]))),
    }


def default_state_path(errors_csv: Path) -> Path:
    return errors_csv.with_name("review_state.json")


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class ReviewServer:
    def __init__(
        self,
        *,
        errors_csv: Path,
        state_path: Path,
        dataset_root: Path,
        host: str,
        port: int,
    ):
        self.errors_csv = errors_csv
        self.state_path = state_path
        self.dataset_root = dataset_root
        self.store = ReviewStore(state_path)
        self.host = host
        self.port = port

    def records(self) -> list[ErrorRecord]:
        return load_error_records(self.errors_csv, store=self.store)

    def serve(self) -> None:
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/":
                    self._send_html()
                    return
                if parsed.path == "/api/items":
                    self._send_items(parsed)
                    return
                if parsed.path == "/api/summary":
                    json_response(self, summarize(service.records(), service.store))
                    return
                if parsed.path == "/image":
                    self._send_image(parsed)
                    return
                json_response(self, {"success": False, "error": "Not found"}, HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body.decode("utf-8") or "{}")
                except json.JSONDecodeError:
                    json_response(self, {"success": False, "error": "Invalid JSON"}, HTTPStatus.BAD_REQUEST)
                    return
                if parsed.path == "/api/mark":
                    self._mark(payload)
                    return
                if parsed.path == "/api/clear":
                    path = str(payload.get("path", ""))
                    service.store.clear(path)
                    json_response(self, {"success": True})
                    return
                if parsed.path == "/api/export_exclude":
                    out_path = resolve_path(
                        str(payload.get("out", service.dataset_root / "exclude_ocr_bad_samples.txt"))
                    )
                    count = service.store.export_exclude_list(out_path, dataset_root=service.dataset_root)
                    json_response(self, {"success": True, "count": count, "out": str(out_path)})
                    return
                json_response(self, {"success": False, "error": "Not found"}, HTTPStatus.NOT_FOUND)

            def _send_items(self, parsed: urllib.parse.ParseResult) -> None:
                query = urllib.parse.parse_qs(parsed.query)
                status = query.get("status", ["pending"])[0]
                category = query.get("category", ["all"])[0]
                layout = query.get("layout", ["all"])[0]
                search = query.get("query", [""])[0]
                page = max(1, parse_int(query.get("page", ["1"])[0]))
                page_size = min(MAX_PAGE_SIZE, max(1, parse_int(query.get("page_size", [str(DEFAULT_PAGE_SIZE)])[0])))
                records = filter_records(
                    service.records(),
                    status=status,
                    category=category,
                    layout=layout,
                    query=search,
                )
                total = len(records)
                total_pages = max(1, (total + page_size - 1) // page_size)
                active_page = min(page, total_pages)
                start = (active_page - 1) * page_size
                items = records[start : start + page_size]
                json_response(
                    self,
                    {
                        "success": True,
                        "items": [record.to_item(dataset_root=service.dataset_root) for record in items],
                        "total": total,
                        "page": active_page,
                        "page_size": page_size,
                        "total_pages": total_pages,
                        "summary": summarize(service.records(), service.store),
                    },
                )

            def _mark(self, payload: dict[str, Any]) -> None:
                path = str(payload.get("path", ""))
                action = str(payload.get("action", ""))
                record = service.store.mark(
                    path,
                    action=action,
                    note=str(payload.get("note", "")),
                    corrected_label=str(payload.get("corrected_label", "")),
                    gt=str(payload.get("gt", "")),
                    pred=str(payload.get("pred", "")),
                )
                json_response(self, {"success": True, "record": record})

            def _send_image(self, parsed: urllib.parse.ParseResult) -> None:
                query = urllib.parse.parse_qs(parsed.query)
                path = query.get("path", [""])[0]
                allowed = {record.path for record in service.records()}
                if path not in allowed:
                    self.send_error(HTTPStatus.NOT_FOUND, "Image not in audit CSV")
                    return
                image_path = Path(path)
                if not image_path.exists() or not image_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
                    return
                data = image_path.read_bytes()
                mime = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_html(self) -> None:
                data = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        print(f"OCR audit review: http://{self.host}:{self.port}")
        print(f"errors_csv={self.errors_csv}")
        print(f"state={self.state_path}")
        print(f"dataset_root={self.dataset_root}")
        server.serve_forever()


HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCR Error Audit Review</title>
  <style>
    :root {
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #151b23;
      --muted: #64748b;
      --border: #d8dee8;
      --accent: #0f766e;
      --danger: #b42318;
      --warn: #a16207;
      --ok: #047857;
      --line: #edf1f5;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      letter-spacing: 0;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(245, 246, 248, 0.96);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(8px);
    }
    .bar {
      max-width: 1500px;
      margin: 0 auto;
      padding: 14px 18px;
      display: grid;
      gap: 10px;
    }
    h1 { margin: 0; font-size: 20px; }
    .controls, .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    select, input, button, textarea {
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    select, input, button { min-height: 34px; padding: 6px 9px; }
    button { cursor: pointer; font-weight: 650; }
    button.primary { background: var(--accent); border-color: var(--accent); color: white; }
    button.danger { background: #fff1f0; border-color: #f3b4af; color: var(--danger); }
    button.warn { background: #fff7db; border-color: #f3d58b; color: var(--warn); }
    button.ok { background: #ecfdf3; border-color: #b8e6c9; color: var(--ok); }
    .pill {
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 999px;
      padding: 5px 9px;
      color: var(--muted);
    }
    main {
      max-width: 1500px;
      margin: 0 auto;
      padding: 18px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
      gap: 12px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    .image-wrap {
      height: 150px;
      display: grid;
      place-items: center;
      background: #111827;
    }
    img {
      max-width: 100%;
      max-height: 150px;
      image-rendering: auto;
    }
    .body { padding: 10px; display: grid; gap: 8px; }
    .line { display: flex; justify-content: space-between; gap: 8px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .gt { color: var(--ok); }
    .pred { color: var(--danger); }
    .global { color: #1d4ed8; }
    .tags { display: flex; flex-wrap: wrap; gap: 5px; }
    .tag {
      border-radius: 999px;
      padding: 3px 7px;
      background: #eef2f7;
      color: #334155;
      font-size: 12px;
    }
    .tag.flag { background: #fff1f0; color: var(--danger); }
    textarea { width: 100%; min-height: 40px; padding: 7px; resize: vertical; }
    .actions { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; }
    .reviewed { opacity: 0.62; }
    .footer { display: flex; justify-content: space-between; align-items: center; margin-top: 14px; }
    @media (max-width: 720px) {
      .grid { grid-template-columns: 1fr; }
      .actions { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <h1>OCR Error Audit Review</h1>
      <div class="stats" id="stats"></div>
      <div class="controls">
        <select id="status">
          <option value="pending">pending</option>
          <option value="reviewed">reviewed</option>
          <option value="bad_crop">bad_crop</option>
          <option value="bad_label">bad_label</option>
          <option value="keep">keep</option>
          <option value="fix_label">fix_label</option>
          <option value="uncertain">uncertain</option>
          <option value="all">all</option>
        </select>
        <select id="layout">
          <option value="all">all layouts</option>
          <option value="one_line">one_line</option>
          <option value="two_line">two_line</option>
        </select>
        <select id="category">
          <option value="all">all categories</option>
        </select>
        <input id="query" placeholder="GT / Pred / filename">
        <button id="reload" class="primary">Reload</button>
        <button id="export" class="danger">Export exclude</button>
      </div>
      <div class="pill">Phím tắt khi hover card: 1 keep, 2 bad_crop, 3 bad_label, 4 fix_label, 5 uncertain</div>
    </div>
  </header>
  <main>
    <div class="grid" id="grid"></div>
    <div class="footer">
      <button id="prev">Prev</button>
      <span id="pageInfo"></span>
      <button id="next">Next</button>
    </div>
  </main>
<script>
const state = { page: 1, pageSize: 80, totalPages: 1, activePath: null, summary: null };
const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[char]));
}

async function loadItems() {
  const params = new URLSearchParams({
    status: $('status').value,
    layout: $('layout').value,
    category: $('category').value,
    query: $('query').value,
    page: state.page,
    page_size: state.pageSize,
  });
  const res = await fetch(`/api/items?${params}`);
  const data = await res.json();
  if (!data.success) throw new Error(data.error || 'Load failed');
  state.totalPages = data.total_pages;
  state.summary = data.summary;
  renderStats(data.summary);
  renderCategoryOptions(data.summary.categories);
  renderItems(data.items);
  $('pageInfo').textContent = `Page ${data.page}/${data.total_pages} · ${data.total} items`;
}

function renderStats(summary) {
  const review = summary.review || {};
  $('stats').innerHTML = [
    ['total errors', summary.total],
    ['pending', summary.pending],
    ['reviewed', review.reviewed || 0],
    ['exclude', review.exclude || 0],
    ['bad_crop', review.bad_crop || 0],
    ['bad_label', review.bad_label || 0],
    ['keep', review.keep || 0],
  ].map(([k, v]) => `<span class="pill">${k}: <b>${v}</b></span>`).join('');
}

function renderCategoryOptions(categories) {
  const current = $('category').value;
  const options = ['<option value="all">all categories</option>'];
  for (const [name, count] of Object.entries(categories || {})) {
    options.push(`<option value="${esc(name)}">${esc(name)} (${count})</option>`);
  }
  $('category').innerHTML = options.join('');
  $('category').value = categories && current in categories ? current : 'all';
}

function renderItems(items) {
  $('grid').innerHTML = items.map((item) => {
    const tags = [...item.categories.map((x) => `<span class="tag">${esc(x)}</span>`),
      ...item.image_flags.map((x) => `<span class="tag flag">${esc(x)}</span>`)].join('');
    const reviewed = item.reviewed ? 'reviewed' : '';
    return `<article class="card ${reviewed}" data-path="${esc(item.path)}">
      <div class="image-wrap"><img src="/image?path=${encodeURIComponent(item.path)}" alt=""></div>
      <div class="body">
        <div class="line"><b>${esc(item.layout)}</b><span>${item.width}x${item.height} · edit ${item.edit_distance}</span></div>
        <div class="mono gt">GT: ${esc(item.gt)}</div>
        <div class="mono pred">Pred: ${esc(item.pred)}</div>
        <div class="mono global">Global: ${esc(item.global_pred)} ${item.global_was_correct ? '✓' : ''}</div>
        <div class="tags">${tags}</div>
        <input class="fix" placeholder="corrected label" value="${esc(item.corrected_label)}">
        <textarea class="note" placeholder="note">${esc(item.review_note)}</textarea>
        <div class="actions">
          <button class="ok" data-action="keep">1 Keep</button>
          <button class="danger" data-action="bad_crop">2 Bad crop</button>
          <button class="danger" data-action="bad_label">3 Bad label</button>
          <button class="warn" data-action="fix_label">4 Fix label</button>
          <button data-action="uncertain">5 Unsure</button>
        </div>
        <span class="pill">${esc(item.review_action || 'pending')} · ${esc(item.rel_path)}</span>
      </div>
    </article>`;
  }).join('');
}

async function mark(card, action) {
  const payload = {
    path: card.dataset.path,
    action,
    note: card.querySelector('.note').value,
    corrected_label: card.querySelector('.fix').value,
    gt: card.querySelector('.gt').textContent.replace(/^GT: /, ''),
    pred: card.querySelector('.pred').textContent.replace(/^Pred: /, ''),
  };
  const res = await fetch('/api/mark', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!data.success) throw new Error(data.error || 'Mark failed');
  card.remove();
  await loadItems();
}

document.addEventListener('click', async (event) => {
  const button = event.target.closest('button');
  if (!button) return;
  if (button.dataset.action) {
    await mark(button.closest('.card'), button.dataset.action);
  }
});
document.addEventListener('mouseover', (event) => {
  const card = event.target.closest('.card');
  if (card) state.activePath = card.dataset.path;
});
document.addEventListener('keydown', async (event) => {
  const map = { '1': 'keep', '2': 'bad_crop', '3': 'bad_label', '4': 'fix_label', '5': 'uncertain' };
  if (!(event.key in map) || !state.activePath) return;
  const card = document.querySelector(`.card[data-path="${CSS.escape(state.activePath)}"]`);
  if (card) await mark(card, map[event.key]);
});
$('reload').addEventListener('click', () => { state.page = 1; loadItems(); });
$('prev').addEventListener('click', () => { state.page = Math.max(1, state.page - 1); loadItems(); });
$('next').addEventListener('click', () => { state.page = Math.min(state.totalPages, state.page + 1); loadItems(); });
for (const id of ['status', 'layout', 'category']) {
  $(id).addEventListener('change', () => { state.page = 1; loadItems(); });
}
$('query').addEventListener('keydown', (event) => {
  if (event.key === 'Enter') { state.page = 1; loadItems(); }
});
$('export').addEventListener('click', async () => {
  const res = await fetch('/api/export_exclude', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  const data = await res.json();
  alert(`Exported ${data.count} paths to ${data.out}`);
});
loadItems().catch((error) => alert(error.message));
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review OCR audit errors and export bad-sample exclusions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the browser review UI.")
    serve.add_argument("--errors-csv", default=str(DEFAULT_ERRORS_CSV))
    serve.add_argument("--state", default=None)
    serve.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    export = subparsers.add_parser("export", help="Export reviewed bad samples to exclude_paths.txt.")
    export.add_argument("--state", required=True)
    export.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    export.add_argument("--out", default=str(DEFAULT_DATASET_ROOT / "exclude_ocr_bad_samples.txt"))

    apply = subparsers.add_parser("apply", help="Quarantine bad samples and rename fix_label samples.")
    apply.add_argument("--state", required=True)
    apply.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    apply.add_argument(
        "--quarantine-dir",
        default=str(DEFAULT_DATASET_ROOT / "_review_removed" / "ocr_error_audit"),
    )
    apply.add_argument(
        "--manifest",
        default=None,
        help="Optional JSONL file to write the planned/applied operations.",
    )
    apply.add_argument("--apply", action="store_true", help="Actually move/rename files. Omit for dry-run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "serve":
        errors_csv = resolve_path(args.errors_csv)
        state_path = resolve_path(args.state) if args.state else default_state_path(errors_csv)
        dataset_root = resolve_path(args.dataset_root)
        ReviewServer(
            errors_csv=errors_csv,
            state_path=state_path,
            dataset_root=dataset_root,
            host=args.host,
            port=args.port,
        ).serve()
        return

    if args.command == "export":
        store = ReviewStore(resolve_path(args.state))
        out_path = resolve_path(args.out)
        count = store.export_exclude_list(out_path, dataset_root=resolve_path(args.dataset_root))
        print(f"exported={count}")
        print(f"out={out_path}")
        return

    if args.command == "apply":
        store = ReviewStore(resolve_path(args.state))
        operations = apply_review_actions(
            store,
            dataset_root=resolve_path(args.dataset_root),
            quarantine_dir=resolve_path(args.quarantine_dir),
            dry_run=not args.apply,
        )
        if args.manifest:
            manifest_path = resolve_path(args.manifest)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                "".join(json.dumps(operation, ensure_ascii=False) + "\n" for operation in operations),
                encoding="utf-8",
            )
            print(f"manifest={manifest_path}")
        applied = sum(1 for operation in operations if operation["status"] == "applied")
        planned = sum(1 for operation in operations if operation["status"] == "dry_run")
        skipped = len(operations) - applied - planned
        print(f"mode={'apply' if args.apply else 'dry-run'}")
        print(f"operations={len(operations)} planned={planned} applied={applied} skipped={skipped}")
        for operation in operations[:40]:
            print(f"{operation['status']} {operation['action']}: {operation['source']} -> {operation['target']}")
        if len(operations) > 40:
            print(f"... {len(operations) - 40} more operations")


if __name__ == "__main__":
    main()
