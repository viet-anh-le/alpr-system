from __future__ import annotations

import argparse
import json
import mimetypes
import re
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "data" / "datasets" / "ocr"
DEFAULT_THRESHOLD = 1.97
DEFAULT_PAGE_SIZE = 80
MAX_PAGE_SIZE = 200
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_REVIEW_STATE_NAME = ".ddl_label_reviewed.json"

DDL_ONE_LINE_RE = re.compile(r"^(?P<head>\d{2}[A-ZĐ])-(?P<body>\d{3}\.\d{2})$")
DDL_TWO_LINE_RE = re.compile(r"^(?P<head>\d{2}[A-ZĐ])\[SEP\](?P<body>\d{3}\.\d{2})$")
SAFE_LABEL_RE = re.compile(r"^[0-9A-ZĐ.\-]+(?:\[SEP\][0-9A-ZĐ.\-]+)?$")


class AppError(Exception):
    status = HTTPStatus.BAD_REQUEST

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ValidationError(AppError):
    status = HTTPStatus.BAD_REQUEST


class NotFoundError(AppError):
    status = HTTPStatus.NOT_FOUND


class ConflictError(AppError):
    status = HTTPStatus.CONFLICT


def label_from_path(path: Path) -> str:
    return path.stem.split("#", 1)[0].upper()


def filename_suffix(path: Path) -> str:
    stem = path.stem
    if "#" not in stem:
        return ""
    return stem[stem.index("#") :]


def is_one_line_ddl_label(label: str) -> bool:
    return bool(DDL_ONE_LINE_RE.fullmatch(label.upper()))


def is_two_line_ddl_label(label: str) -> bool:
    return bool(DDL_TWO_LINE_RE.fullmatch(label.upper()))


def is_review_label(label: str) -> bool:
    text = label.upper()
    return is_one_line_ddl_label(text) or is_two_line_ddl_label(text)


def to_two_line_label(label: str) -> str:
    match = DDL_ONE_LINE_RE.fullmatch(label.upper())
    if not match:
        raise ValidationError(f"Label is not DDL-DDD.DD: {label}")
    return f"{match.group('head')}[SEP]{match.group('body')}"


def to_one_line_label(label: str) -> str:
    match = DDL_TWO_LINE_RE.fullmatch(label.upper())
    if not match:
        raise ValidationError(f"Label is not DDL[SEP]DDD.DD: {label}")
    return f"{match.group('head')}-{match.group('body')}"


def validate_label(label: str) -> str:
    normalized = label.strip().upper()
    if not normalized:
        raise ValidationError("Label is empty")
    if len(normalized) > 64:
        raise ValidationError("Label is too long")
    if any(char in normalized for char in ("#", "/", "\\", "\x00")):
        raise ValidationError("Label contains an invalid character")
    if not SAFE_LABEL_RE.fullmatch(normalized):
        raise ValidationError("Label contains unsupported characters")
    return normalized


def suggest_label(label: str, *, width: int, height: int, threshold: float) -> str:
    normalized = label.upper()
    if is_two_line_ddl_label(normalized):
        return normalized
    if not is_one_line_ddl_label(normalized):
        return normalized
    if height <= 0:
        return normalized
    ratio = width / height
    if ratio <= threshold:
        return to_two_line_label(normalized)
    return normalized


def build_filename_with_label(path: Path, new_label: str) -> str:
    normalized = validate_label(new_label)
    return f"{normalized}{filename_suffix(path)}{path.suffix}"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_dataset_path(dataset_dir: Path, rel_path: str) -> Path:
    if not rel_path or rel_path.startswith(("/", "\\")):
        raise ValidationError("Path must be relative to the dataset directory")
    candidate = (dataset_dir / rel_path).resolve()
    root = dataset_dir.resolve()
    if not _is_relative_to(candidate, root):
        raise ValidationError("Path escapes the dataset directory")
    return candidate


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def load_reviewed_paths(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if isinstance(payload, dict):
        values = payload.get("reviewed_paths", [])
    elif isinstance(payload, list):
        values = payload
    else:
        values = []
    return {value for value in values if isinstance(value, str)}


def save_reviewed_paths(path: Path, reviewed_paths: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "reviewed_paths": sorted(reviewed_paths),
    }
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


class DatasetReviewService:
    def __init__(
        self,
        *,
        dataset_dir: Path,
        threshold: float = DEFAULT_THRESHOLD,
        state_path: Path | None = None,
    ):
        self.dataset_dir = dataset_dir.resolve()
        self.threshold = float(threshold)
        self.state_path = (state_path or self.dataset_dir / DEFAULT_REVIEW_STATE_NAME).resolve()
        self.reviewed_paths = load_reviewed_paths(self.state_path)
        self._records: dict[str, dict[str, Any]] = {}
        self.refresh()

    def refresh(self) -> None:
        records: dict[str, dict[str, Any]] = {}
        for path in sorted(self.dataset_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = label_from_path(path)
            if not is_review_label(label):
                continue
            try:
                width, height = read_image_size(path)
            except OSError:
                continue
            rel_path = path.relative_to(self.dataset_dir).as_posix()
            records[rel_path] = {
                "rel_path": rel_path,
                "path": path,
                "filename": path.name,
                "label": label,
                "split": path.relative_to(self.dataset_dir).parts[0]
                if len(path.relative_to(self.dataset_dir).parts) > 1
                else "",
                "width": width,
                "height": height,
                "mtime": path.stat().st_mtime,
            }
        self._records = records

    def _record_to_item(self, record: dict[str, Any], *, threshold: float) -> dict[str, Any]:
        width = int(record["width"])
        height = int(record["height"])
        label = str(record["label"])
        suggested = suggest_label(label, width=width, height=height, threshold=threshold)
        ratio = width / height if height else 0.0
        current_layout = "two_line" if is_two_line_ddl_label(label) else "one_line"
        suggested_layout = "two_line" if is_two_line_ddl_label(suggested) else "one_line"
        one_line_label = to_one_line_label(label) if is_two_line_ddl_label(label) else label
        two_line_label = to_two_line_label(label) if is_one_line_ddl_label(label) else label
        return {
            "rel_path": record["rel_path"],
            "filename": record["filename"],
            "label": label,
            "suggested_label": suggested,
            "one_line_label": one_line_label,
            "two_line_label": two_line_label,
            "needs_change": label != suggested,
            "current_layout": current_layout,
            "suggested_layout": suggested_layout,
            "split": record["split"],
            "width": width,
            "height": height,
            "ratio": round(ratio, 4),
            "mtime": record["mtime"],
            "reviewed": record["rel_path"] in self.reviewed_paths,
        }

    def _items(self, *, threshold: float) -> list[dict[str, Any]]:
        return [
            self._record_to_item(record, threshold=threshold)
            for _, record in sorted(self._records.items())
        ]

    def list_samples(
        self,
        *,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        split: str = "all",
        status: str = "pending",
        query: str = "",
        sort: str = "path",
        threshold: float | None = None,
    ) -> dict[str, Any]:
        active_threshold = self.threshold if threshold is None else float(threshold)
        items = self._items(threshold=active_threshold)
        counts = self._counts(items)

        if split != "all":
            items = [item for item in items if item["split"] == split]
        if query:
            needle = query.strip().upper()
            items = [
                item
                for item in items
                if needle in item["label"] or needle in item["filename"].upper()
            ]

        items = self._filter_status(items, status)
        items = self._sort_items(items, sort)

        safe_page_size = min(MAX_PAGE_SIZE, max(1, int(page_size)))
        total = len(items)
        total_pages = max(1, (total + safe_page_size - 1) // safe_page_size)
        safe_page = min(max(1, int(page)), total_pages)
        start = (safe_page - 1) * safe_page_size
        end = start + safe_page_size

        return {
            "items": items[start:end],
            "total": total,
            "page": safe_page,
            "page_size": safe_page_size,
            "total_pages": total_pages,
            "threshold": active_threshold,
            "counts": counts,
        }

    def _counts(self, items: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "all": len(items),
            "pending": sum(1 for item in items if not item["reviewed"]),
            "reviewed": sum(1 for item in items if item["reviewed"]),
            "needs_change": sum(1 for item in items if item["needs_change"]),
            "current_one": sum(1 for item in items if item["current_layout"] == "one_line"),
            "current_two": sum(1 for item in items if item["current_layout"] == "two_line"),
            "suggested_one": sum(1 for item in items if item["suggested_layout"] == "one_line"),
            "suggested_two": sum(1 for item in items if item["suggested_layout"] == "two_line"),
        }

    def _filter_status(self, items: list[dict[str, Any]], status: str) -> list[dict[str, Any]]:
        if status == "pending":
            return [item for item in items if not item["reviewed"]]
        if status == "reviewed":
            return [item for item in items if item["reviewed"]]
        if status == "needs_change":
            return [item for item in items if item["needs_change"] and not item["reviewed"]]
        if status == "current_one":
            return [item for item in items if item["current_layout"] == "one_line" and not item["reviewed"]]
        if status == "current_two":
            return [item for item in items if item["current_layout"] == "two_line" and not item["reviewed"]]
        if status == "suggested_one":
            return [item for item in items if item["suggested_layout"] == "one_line" and not item["reviewed"]]
        if status == "suggested_two":
            return [item for item in items if item["suggested_layout"] == "two_line" and not item["reviewed"]]
        return items

    def _sort_items(self, items: list[dict[str, Any]], sort: str) -> list[dict[str, Any]]:
        if sort == "ratio_asc":
            return sorted(items, key=lambda item: (item["ratio"], item["rel_path"]))
        if sort == "ratio_desc":
            return sorted(items, key=lambda item: (-item["ratio"], item["rel_path"]))
        if sort == "needs_change":
            return sorted(items, key=lambda item: (not item["needs_change"], item["rel_path"]))
        return sorted(items, key=lambda item: item["rel_path"])

    def get_image_path(self, rel_path: str) -> Path:
        path = resolve_dataset_path(self.dataset_dir, rel_path)
        if not path.exists() or not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise NotFoundError("Image not found")
        return path

    def rename_sample(self, rel_path: str, new_label: str) -> dict[str, Any]:
        return self._rename_sample(rel_path, new_label, mark_reviewed=True)

    def _rename_sample(self, rel_path: str, new_label: str, *, mark_reviewed: bool) -> dict[str, Any]:
        old_path = self.get_image_path(rel_path)
        if old_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValidationError("Unsupported image extension")
        target_name = build_filename_with_label(old_path, new_label)
        target_path = old_path.with_name(target_name)

        if target_path.exists() and target_path.resolve() != old_path.resolve():
            raise ConflictError("Target file already exists")

        old_sidecar = old_path.with_suffix(".txt")
        target_sidecar = target_path.with_suffix(".txt")
        if old_sidecar.exists() and target_sidecar.exists() and target_sidecar.resolve() != old_sidecar.resolve():
            raise ConflictError("Target sidecar already exists")

        old_rel = old_path.relative_to(self.dataset_dir).as_posix()
        if target_path.resolve() != old_path.resolve():
            old_path.rename(target_path)
            if old_sidecar.exists():
                old_sidecar.rename(target_sidecar)

        self._records.pop(old_rel, None)
        try:
            width, height = read_image_size(target_path)
        except OSError as exc:
            raise ValidationError("Renamed image could not be read") from exc

        new_rel = target_path.relative_to(self.dataset_dir).as_posix()
        label = label_from_path(target_path)
        was_reviewed = old_rel in self.reviewed_paths
        self.reviewed_paths.discard(old_rel)
        if mark_reviewed or was_reviewed:
            self.reviewed_paths.add(new_rel)
        if mark_reviewed or was_reviewed:
            save_reviewed_paths(self.state_path, self.reviewed_paths)
        if is_review_label(label):
            self._records[new_rel] = {
                "rel_path": new_rel,
                "path": target_path,
                "filename": target_path.name,
                "label": label,
                "split": target_path.relative_to(self.dataset_dir).parts[0],
                "width": width,
                "height": height,
                "mtime": target_path.stat().st_mtime,
            }

        item = self._record_to_item(
            {
                "rel_path": new_rel,
                "path": target_path,
                "filename": target_path.name,
                "label": label,
                "split": target_path.relative_to(self.dataset_dir).parts[0],
                "width": width,
                "height": height,
                "mtime": target_path.stat().st_mtime,
            },
            threshold=self.threshold,
        )
        return {**item, "reviewed": new_rel in self.reviewed_paths}

    def mark_reviewed(self, rel_paths: list[str]) -> dict[str, Any]:
        reviewed: list[str] = []
        skipped: list[str] = []

        for rel_path in rel_paths:
            if rel_path not in self._records:
                skipped.append(rel_path)
                continue
            self.reviewed_paths.add(rel_path)
            reviewed.append(rel_path)

        if reviewed:
            save_reviewed_paths(self.state_path, self.reviewed_paths)

        return {
            "success": True,
            "reviewed": reviewed,
            "skipped": skipped,
        }

    def bulk_apply_suggestions(self, rel_paths: list[str], *, threshold: float | None = None) -> dict[str, Any]:
        active_threshold = self.threshold if threshold is None else float(threshold)
        renamed: list[dict[str, Any]] = []
        skipped: list[str] = []
        errors: list[dict[str, str]] = []

        for rel_path in rel_paths:
            record = self._records.get(rel_path)
            if record is None:
                skipped.append(rel_path)
                continue
            item = self._record_to_item(record, threshold=active_threshold)
            if not item["needs_change"]:
                skipped.append(rel_path)
                continue
            try:
                renamed.append(
                    self._rename_sample(
                        rel_path,
                        item["suggested_label"],
                        mark_reviewed=False,
                    )
                )
            except AppError as exc:
                errors.append({"path": rel_path, "error": exc.message})

        return {
            "success": not errors,
            "renamed": renamed,
            "skipped": skipped,
            "errors": errors,
        }


HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DDL Label Review</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5f6b7a;
      --border: #d7dde5;
      --line: #edf0f4;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --warn: #b7791f;
      --danger: #b42318;
      --ok: #047857;
      --shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      letter-spacing: 0;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(246, 247, 249, 0.96);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(10px);
    }
    .bar {
      max-width: 1440px;
      margin: 0 auto;
      padding: 14px 18px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 700;
    }
    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
    }
    .pill {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 5px 9px;
      box-shadow: var(--shadow);
      white-space: nowrap;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 8px;
      align-items: end;
    }
    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    input, select, button {
      min-height: 36px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      font: inherit;
      letter-spacing: 0;
    }
    input, select { padding: 0 10px; }
    button {
      cursor: pointer;
      padding: 0 12px;
      font-weight: 650;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: white;
    }
    button.primary:hover { background: var(--accent-strong); }
    button.secondary:hover { border-color: var(--accent); }
    button:disabled {
      opacity: .55;
      cursor: not-allowed;
    }
    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px 18px 32px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
      gap: 12px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
      min-width: 0;
    }
    .image-wrap {
      height: 150px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #f0f2f5;
      border-bottom: 1px solid var(--line);
    }
    .image-wrap img {
      max-width: 100%;
      max-height: 148px;
      object-fit: contain;
      display: block;
    }
    .body {
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .meta {
      display: grid;
      gap: 4px;
      min-width: 0;
    }
    .path {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .row {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      align-items: center;
    }
    .tag {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 3px 7px;
      font-size: 12px;
      color: var(--muted);
      background: #fafbfc;
      white-space: nowrap;
    }
    .tag.need { color: var(--warn); border-color: #e6c381; background: #fff8e8; }
    .tag.two { color: var(--accent); border-color: #9ed5ce; background: #eefaf8; }
    .tag.saved { color: var(--ok); border-color: #9bd2ba; background: #f0fbf6; }
    .tag.error { color: var(--danger); border-color: #f0aaa3; background: #fff1ef; }
    .edit {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 6px;
    }
    .edit input {
      width: 100%;
      min-width: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
    }
    .actions {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
    }
    .pager {
      margin-top: 16px;
      display: flex;
      justify-content: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .empty {
      grid-column: 1 / -1;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 28px;
      color: var(--muted);
      text-align: center;
    }
    .toast {
      position: fixed;
      right: 16px;
      bottom: 16px;
      min-width: 260px;
      max-width: min(460px, calc(100vw - 32px));
      background: #17202a;
      color: white;
      border-radius: 8px;
      padding: 11px 13px;
      box-shadow: 0 12px 24px rgba(16, 24, 40, .22);
      opacity: 0;
      transform: translateY(10px);
      transition: opacity .18s ease, transform .18s ease;
      pointer-events: none;
      overflow-wrap: anywhere;
    }
    .toast.show {
      opacity: 1;
      transform: translateY(0);
    }
    @media (max-width: 980px) {
      .controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 560px) {
      .controls { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      main, .bar { padding-left: 10px; padding-right: 10px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div class="row">
        <h1>DDL Label Review</h1>
        <button class="secondary" id="refreshBtn" type="button">Refresh</button>
      </div>
      <div class="stats" id="stats"></div>
      <div class="controls">
        <label>Split
          <select id="split">
            <option value="all">all</option>
            <option value="train">train</option>
            <option value="valid">valid</option>
          </select>
        </label>
        <label>Status
          <select id="status">
            <option value="pending" selected>pending</option>
            <option value="all">all</option>
            <option value="reviewed">reviewed</option>
            <option value="needs_change">needs change</option>
            <option value="current_one">current one</option>
            <option value="current_two">current two</option>
            <option value="suggested_one">suggest one</option>
            <option value="suggested_two">suggest two</option>
          </select>
        </label>
        <label>Sort
          <select id="sort">
            <option value="needs_change">needs first</option>
            <option value="path">path</option>
            <option value="ratio_asc">ratio asc</option>
            <option value="ratio_desc">ratio desc</option>
          </select>
        </label>
        <label>Threshold
          <input id="threshold" type="number" min="0.5" max="6" step="0.01" value="1.97">
        </label>
        <label>Search
          <input id="query" type="search" placeholder="79A, nomer...">
        </label>
        <label>Page size
          <select id="pageSize">
            <option value="40">40</option>
            <option value="80" selected>80</option>
            <option value="120">120</option>
            <option value="200">200</option>
          </select>
        </label>
      </div>
      <div class="row">
        <button class="primary" id="applyPageBtn" type="button">Apply visible suggestions</button>
        <button class="secondary" id="reviewPageBtn" type="button">Đánh dấu trang reviewed</button>
        <span class="pill" id="pageInfo">Loading</span>
      </div>
    </div>
  </header>
  <main>
    <section class="grid" id="grid"></section>
    <nav class="pager">
      <button class="secondary" id="prevBtn" type="button">Previous</button>
      <button class="secondary" id="nextBtn" type="button">Next</button>
    </nav>
  </main>
  <div class="toast" id="toast"></div>
  <script>
    const state = {
      page: 1,
      totalPages: 1,
      items: [],
      loading: false
    };

    const $ = (id) => document.getElementById(id);

    function params() {
      const qs = new URLSearchParams();
      qs.set('page', state.page);
      qs.set('page_size', $('pageSize').value);
      qs.set('split', $('split').value);
      qs.set('status', $('status').value);
      qs.set('sort', $('sort').value);
      qs.set('threshold', $('threshold').value);
      const query = $('query').value.trim();
      if (query) qs.set('q', query);
      return qs;
    }

    function toast(message, kind = 'ok') {
      const node = $('toast');
      node.textContent = message;
      node.style.background = kind === 'error' ? '#7a271a' : '#17202a';
      node.classList.add('show');
      window.clearTimeout(node._timer);
      node._timer = window.setTimeout(() => node.classList.remove('show'), 2600);
    }

    async function load() {
      state.loading = true;
      $('grid').innerHTML = '<div class="empty">Loading</div>';
      try {
        const res = await fetch('/api/samples?' + params().toString());
        const data = await res.json();
        if (!res.ok || !data.success) throw new Error(data.error || 'Load failed');
        state.items = data.items;
        state.totalPages = data.total_pages;
        state.page = data.page;
        renderStats(data);
        renderGrid();
        renderPager(data);
      } catch (err) {
        $('grid').innerHTML = '<div class="empty">Load failed</div>';
        toast(String(err.message || err), 'error');
      } finally {
        state.loading = false;
      }
    }

    function renderStats(data) {
      const counts = data.counts;
      $('stats').innerHTML = [
        ['all', counts.all],
        ['pending', counts.pending],
        ['reviewed', counts.reviewed],
        ['needs', counts.needs_change],
        ['current 1', counts.current_one],
        ['current 2', counts.current_two],
        ['suggest 1', counts.suggested_one],
        ['suggest 2', counts.suggested_two]
      ].map(([label, value]) => `<span class="pill">${label}: ${value}</span>`).join('');
    }

    function renderPager(data) {
      $('pageInfo').textContent = `page ${data.page}/${data.total_pages} | ${data.total} shown`;
      $('prevBtn').disabled = data.page <= 1;
      $('nextBtn').disabled = data.page >= data.total_pages;
      $('applyPageBtn').disabled = !state.items.some((item) => item.needs_change);
      $('reviewPageBtn').disabled = !state.items.some((item) => !item.reviewed);
    }

    function layoutText(value) {
      return value === 'two_line' ? '2 dòng' : '1 dòng';
    }

    function escapeAttr(value) {
      return String(value).replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    }

    function renderGrid() {
      const grid = $('grid');
      if (!state.items.length) {
        grid.innerHTML = '<div class="empty">No samples</div>';
        return;
      }
      grid.innerHTML = state.items.map((item, index) => {
        const needClass = item.needs_change ? 'need' : 'saved';
        const twoClass = item.suggested_layout === 'two_line' ? 'two' : '';
        const imageSrc = '/image?path=' + encodeURIComponent(item.rel_path);
        const oneLine = item.one_line_label;
        const twoLine = item.two_line_label;
        const reviewedClass = item.reviewed ? 'saved' : '';
        return `
          <article class="card" data-index="${index}">
            <a class="image-wrap" href="${imageSrc}" target="_blank" rel="noreferrer">
              <img loading="lazy" src="${imageSrc}" alt="${escapeAttr(item.label)}">
            </a>
            <div class="body">
              <div class="meta">
                <strong>${escapeAttr(item.label)}</strong>
                <span class="path">${escapeAttr(item.rel_path)}</span>
              </div>
              <div class="row">
                <span class="tag ${needClass}">${item.needs_change ? 'needs change' : 'saved'}</span>
                <span class="tag ${reviewedClass}">${item.reviewed ? 'reviewed' : 'pending'}</span>
                <span class="tag ${twoClass}">suggest ${layoutText(item.suggested_layout)}</span>
                <span class="tag">${item.width}x${item.height}</span>
                <span class="tag">r ${item.ratio}</span>
              </div>
              <div class="edit">
                <input id="label-${index}" value="${escapeAttr(item.suggested_label)}" autocomplete="off">
                <button class="primary" type="button" onclick="saveLabel(${index})">Save</button>
              </div>
              <div class="actions">
                <button class="secondary" type="button" onclick="setAndSave(${index}, '${escapeAttr(oneLine)}')">Lưu 1 dòng</button>
                <button class="secondary" type="button" onclick="setAndSave(${index}, '${escapeAttr(twoLine)}')">Lưu 2 dòng</button>
                <button class="secondary" type="button" onclick="resetInput(${index})">Reset</button>
              </div>
            </div>
          </article>
        `;
      }).join('');
    }

    function resetInput(index) {
      $('label-' + index).value = state.items[index].label;
    }

    async function setAndSave(index, value) {
      $('label-' + index).value = value;
      await saveLabel(index);
    }

    async function saveLabel(index) {
      const item = state.items[index];
      const newLabel = $('label-' + index).value.trim();
      try {
        const res = await fetch('/api/rename', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ path: item.rel_path, new_label: newLabel })
        });
        const data = await res.json();
        if (!res.ok || !data.success) throw new Error(data.error || 'Rename failed');
        toast('Reviewed ' + data.item.filename);
        await load();
      } catch (err) {
        toast(String(err.message || err), 'error');
      }
    }

    async function applyVisibleSuggestions() {
      const paths = state.items.filter((item) => item.needs_change).map((item) => item.rel_path);
      if (!paths.length) return;
      if (!window.confirm(`Apply ${paths.length} visible suggestions?`)) return;
      try {
        const res = await fetch('/api/bulk_apply', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ paths, threshold: Number($('threshold').value) })
        });
        const data = await res.json();
        if (!res.ok || !data.success) {
          const count = data.errors ? data.errors.length : 0;
          throw new Error(data.error || `${count} rename errors`);
        }
        toast(`Applied ${data.renamed.length} suggestions`);
        await load();
      } catch (err) {
        toast(String(err.message || err), 'error');
        await load();
      }
    }

    async function markVisibleReviewed() {
      const paths = state.items.filter((item) => !item.reviewed).map((item) => item.rel_path);
      if (!paths.length) return;
      if (!window.confirm(`Mark ${paths.length} visible samples as reviewed?`)) return;
      try {
        const res = await fetch('/api/mark_reviewed', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ paths })
        });
        const data = await res.json();
        if (!res.ok || !data.success) throw new Error(data.error || 'Mark reviewed failed');
        toast(`Reviewed ${data.reviewed.length} visible samples`);
        await load();
      } catch (err) {
        toast(String(err.message || err), 'error');
        await load();
      }
    }

    async function refreshServer() {
      try {
        const res = await fetch('/api/refresh', {method: 'POST'});
        const data = await res.json();
        if (!res.ok || !data.success) throw new Error(data.error || 'Refresh failed');
        toast('Refreshed');
        await load();
      } catch (err) {
        toast(String(err.message || err), 'error');
      }
    }

    for (const id of ['split', 'status', 'sort', 'pageSize']) {
      $(id).addEventListener('change', () => { state.page = 1; load(); });
    }
    $('threshold').addEventListener('change', () => { state.page = 1; load(); });
    $('query').addEventListener('input', () => {
      window.clearTimeout($('query')._timer);
      $('query')._timer = window.setTimeout(() => { state.page = 1; load(); }, 250);
    });
    $('prevBtn').addEventListener('click', () => { if (state.page > 1) { state.page -= 1; load(); } });
    $('nextBtn').addEventListener('click', () => { if (state.page < state.totalPages) { state.page += 1; load(); } });
    $('applyPageBtn').addEventListener('click', applyVisibleSuggestions);
    $('reviewPageBtn').addEventListener('click', markVisibleReviewed);
    $('refreshBtn').addEventListener('click', refreshServer);
    load();
  </script>
</body>
</html>
"""


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def parse_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def parse_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def make_handler(service: DatasetReviewService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send(
            self,
            status: HTTPStatus,
            body: bytes,
            *,
            content_type: str = "application/json; charset=utf-8",
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            self._send(status, json_bytes(payload))

        def _read_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValidationError("Invalid Content-Length") from exc
            if length <= 0 or length > 65536:
                raise ValidationError("Invalid request body size")
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValidationError("Invalid JSON") from exc

        def _handle_error(self, exc: Exception) -> None:
            if isinstance(exc, AppError):
                self._send_json(exc.status, {"success": False, "error": exc.message})
                return
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"success": False, "error": "Internal server error"},
            )

        def do_GET(self) -> None:
            try:
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/":
                    self._send(HTTPStatus.OK, HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
                    return
                if parsed.path == "/api/samples":
                    query = urllib.parse.parse_qs(parsed.query)
                    result = service.list_samples(
                        page=parse_int(_first(query, "page"), 1),
                        page_size=parse_int(_first(query, "page_size"), DEFAULT_PAGE_SIZE),
                        split=_first(query, "split") or "all",
                        status=_first(query, "status") or "pending",
                        query=_first(query, "q") or "",
                        sort=_first(query, "sort") or "path",
                        threshold=parse_float(_first(query, "threshold"), service.threshold),
                    )
                    self._send_json(HTTPStatus.OK, {"success": True, **result})
                    return
                if parsed.path == "/image":
                    query = urllib.parse.parse_qs(parsed.query)
                    image_path = service.get_image_path(_first(query, "path") or "")
                    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                    self._send(HTTPStatus.OK, image_path.read_bytes(), content_type=content_type)
                    return
                raise NotFoundError("Route not found")
            except Exception as exc:  # noqa: BLE001 - all request errors become API responses.
                self._handle_error(exc)

        def do_POST(self) -> None:
            try:
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/api/rename":
                    data = self._read_json()
                    item = service.rename_sample(str(data.get("path", "")), str(data.get("new_label", "")))
                    self._send_json(HTTPStatus.OK, {"success": True, "item": item})
                    return
                if parsed.path == "/api/bulk_apply":
                    data = self._read_json()
                    paths = data.get("paths")
                    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                        raise ValidationError("paths must be a list of strings")
                    result = service.bulk_apply_suggestions(
                        paths,
                        threshold=parse_float(str(data.get("threshold")), service.threshold)
                        if data.get("threshold") is not None
                        else service.threshold,
                    )
                    status = HTTPStatus.OK if result["success"] else HTTPStatus.CONFLICT
                    self._send_json(status, result)
                    return
                if parsed.path == "/api/mark_reviewed":
                    data = self._read_json()
                    paths = data.get("paths")
                    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                        raise ValidationError("paths must be a list of strings")
                    self._send_json(HTTPStatus.OK, service.mark_reviewed(paths))
                    return
                if parsed.path == "/api/refresh":
                    service.refresh()
                    self._send_json(HTTPStatus.OK, {"success": True})
                    return
                raise NotFoundError("Route not found")
            except Exception as exc:  # noqa: BLE001 - all request errors become API responses.
                self._handle_error(exc)

    return Handler


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review and rename DDL-DDD.DD OCR labels.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = DatasetReviewService(dataset_dir=args.dataset_dir, threshold=args.threshold)
    handler = make_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"DDL label review: http://{args.host}:{args.port}")
    print(f"Dataset: {service.dataset_dir}")
    print(f"Review samples: {len(service._records)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
