from __future__ import annotations

import csv
import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_ablation_runner_builds_train_command_without_checkpoint_init(tmp_path: Path) -> None:
    module = _load_script("run_small_lpr_line_ctc_ablation")
    matrix = {
        "python": "/tmp/python",
        "defaults": {
            "data_root": "data/datasets/ocr",
            "epochs": 50,
            "batch_size": 64,
            "lr": 0.0003,
            "devices": "1",
            "precision": "32",
            "seed": 42,
        },
        "runs": [
            {
                "id": "A0_global_no_aug",
                "train": True,
                "augment": False,
                "use_stn": False,
                "use_pos_enc": False,
                "decode_mode": "global",
                "line_prior_strength": 0.0,
                "loss_weights": {
                    "global": 1.0,
                    "one_line": 0.0,
                    "top": 0.0,
                    "bottom": 0.0,
                    "layout": 0.0,
                },
            }
        ],
    }
    opts = Namespace(
        out_dir=str(tmp_path / "ablation"),
        epochs=None,
        devices=None,
        precision=None,
        seed=None,
        config="LPRNet/config/small_lpr_line_ctc_config.yaml",
    )

    command = module.build_train_command(matrix, matrix["runs"][0], opts)

    assert command[:2] == ["/tmp/python", "scripts/train_small_lpr_line_ctc.py"]
    assert "--run-name" in command
    assert "A0_global_no_aug" in command
    assert "--no-augment" in command
    assert "--no-use-stn" in command
    assert "--no-use-pos-enc" in command
    assert "--decode-mode" in command
    assert "global" in command
    assert "--init-from" not in command
    assert "--resume" not in command


@pytest.mark.unit
def test_ablation_runner_rejects_resume_or_init_from() -> None:
    module = _load_script("run_small_lpr_line_ctc_ablation")

    with pytest.raises(ValueError, match="train from scratch"):
        module.validate_run({"id": "bad", "resume": "last.ckpt"})

    with pytest.raises(ValueError, match="train from scratch"):
        module.validate_run({"id": "bad", "init_from": "best.ckpt"})


@pytest.mark.unit
def test_ablation_runner_filters_only_runs() -> None:
    module = _load_script("run_small_lpr_line_ctc_ablation")
    matrix = {
        "runs": [
            {"id": "A0_global_no_aug"},
            {"id": "A5_full"},
            {"id": "A6_full_format_corrected"},
        ]
    }

    selected = module.select_runs(matrix, "A0_global_no_aug,A5_full")

    assert [run["id"] for run in selected] == ["A0_global_no_aug", "A5_full"]


@pytest.mark.unit
def test_ablation_summarizer_selects_best_epoch_from_metrics_csv(tmp_path: Path) -> None:
    module = _load_script("summarize_small_lpr_line_ctc_ablation")
    metrics_path = tmp_path / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "step", "val_acc", "val_loss"])
        writer.writeheader()
        writer.writerow({"epoch": "0", "step": "10", "val_acc": "0.50", "val_loss": "1.2"})
        writer.writerow({"epoch": "1", "step": "20", "val_acc": "0.75", "val_loss": "0.8"})
        writer.writerow({"epoch": "2", "step": "30", "val_acc": "0.70", "val_loss": "0.7"})

    best = module.best_metrics_from_csv(metrics_path)

    assert best["epoch"] == 1
    assert best["val_acc"] == pytest.approx(0.75)
    assert best["val_loss"] == pytest.approx(0.8)


@pytest.mark.unit
def test_ablation_summarizer_writes_lprnet_style_table(tmp_path: Path) -> None:
    module = _load_script("summarize_small_lpr_line_ctc_ablation")
    matrix_path = tmp_path / "matrix.yaml"
    matrix = {
        "runs": [
            {
                "id": "A5_full",
                "augment": True,
                "use_stn": True,
                "use_pos_enc": True,
                "decode_mode": "layout",
                "line_prior_strength": 1.0,
                "format_correction": False,
            }
        ]
    }
    matrix_path.write_text(yaml.safe_dump(matrix), encoding="utf-8")
    run_dir = tmp_path / "root" / "A5_full" / "logs" / "version_0"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.csv").write_text(
        "epoch,step,val_acc,val_loss\n0,1,0.91,0.2\n",
        encoding="utf-8",
    )

    rows = module.collect_rows(matrix, tmp_path / "root")
    out_dir = tmp_path / "summary"
    module.write_outputs(rows, out_dir)

    markdown = (out_dir / "ablation_results.md").read_text(encoding="utf-8")
    csv_text = (out_dir / "ablation_results.csv").read_text(encoding="utf-8")

    assert "| Run | Data aug. | STN | 2D PE | Layout heads | Vertical prior | Format correction | Accuracy (%) |" in markdown
    assert "| A5_full | ✓ | ✓ | ✓ | ✓ | ✓ |  | 91.00 |" in markdown
    assert "A5_full" in csv_text
