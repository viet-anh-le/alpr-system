from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


def _load_script_module():
    path = Path("scripts/evaluate_quality_router.py")
    spec = importlib.util.spec_from_file_location("evaluate_quality_router", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_quality_router_confusion_matrix_uses_true_rows_predicted_columns() -> None:
    mod = _load_script_module()

    matrix = mod.build_confusion_matrix(
        y_true=["perfect", "good", "poor", "illegible", "poor"],
        y_pred=["perfect", "poor", "poor", "good", "perfect"],
        labels=mod.LEGIBILITY_LABELS,
    )

    assert matrix == [
        [0, 0, 1, 0],  # true illegible -> predicted good once
        [0, 1, 0, 1],  # true poor -> predicted poor once, perfect once
        [0, 1, 0, 0],  # true good -> predicted poor once
        [0, 0, 0, 1],  # true perfect -> predicted perfect once
    ]


@pytest.mark.unit
def test_quality_router_report_contains_critical_router_metrics() -> None:
    mod = _load_script_module()

    report = mod.evaluate_predictions(
        y_true=["perfect", "good", "poor", "illegible", "poor"],
        y_pred=["perfect", "poor", "poor", "good", "perfect"],
    )

    assert report["n"] == 5
    assert report["accuracy"] == pytest.approx(0.4)
    assert report["per_class"]["poor"]["recall"] == pytest.approx(0.5)
    assert report["per_class"]["illegible"]["recall"] == pytest.approx(0.0)
    assert report["critical"]["poor_recall"] == pytest.approx(0.5)
    assert report["critical"]["illegible_recall"] == pytest.approx(0.0)
    assert report["critical"]["false_suitable_rate"] == pytest.approx(2 / 3)
    assert report["binary"]["confusion_matrix"] == [
        [1, 2],
        [1, 1],
    ]


@pytest.mark.unit
def test_quality_router_loads_predictions_csv_with_alias_columns(tmp_path) -> None:
    mod = _load_script_module()
    csv_path = tmp_path / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "prediction", "path"])
        writer.writeheader()
        writer.writerow({"label": "3", "prediction": "perfect", "path": "a.jpg"})
        writer.writerow({"label": "poor", "prediction": "good", "path": "b.jpg"})

    rows = mod.load_predictions_csv(csv_path)

    assert rows == [
        ("perfect", "perfect", "a.jpg"),
        ("poor", "good", "b.jpg"),
    ]


@pytest.mark.unit
def test_quality_router_saves_report_and_confusion_csvs(tmp_path) -> None:
    mod = _load_script_module()
    report = mod.evaluate_predictions(
        y_true=["perfect", "poor"],
        y_pred=["perfect", "good"],
    )

    mod.save_report(report, tmp_path)

    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "confusion_matrix.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "true\\pred,illegible,poor,good,perfect"
    )
    assert (tmp_path / "binary_confusion_matrix.csv").exists()
