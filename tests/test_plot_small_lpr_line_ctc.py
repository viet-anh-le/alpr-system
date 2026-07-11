"""Contract tests for the PlotNeuralNet SmallLPR-Line-CTC diagram."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "PlotNeuralNet" / "pyexamples" / "small_lpr_line_ctc.py"


def _load_diagram_module():
    spec = importlib.util.spec_from_file_location("small_lpr_line_ctc_diagram", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagram_contains_the_model_forward_contract() -> None:
    module = _load_diagram_module()

    assert module.TENSOR_CONTRACT == {
        "input": "3 x 48 x 96",
        "encoder": "6 x 12 x 256",
        "global_logits": "72 x |V|",
        "one_line_logits": "16 x |V|",
        "top_logits": "12 x |V|",
        "bottom_logits": "12 x |V|",
        "layout_logits": "2",
    }


def test_generated_tex_uses_plotneuralnet_and_has_every_output_branch() -> None:
    module = _load_diagram_module()
    tex = "".join(module.build_architecture())

    for node_name in (
        "input",
        "stn",
        "stem",
        "shared_encoder",
        "projection",
        "posenc",
        "global",
        "oneline",
        "top",
        "bottom",
        "layout",
        "decode",
    ):
        assert f"name={node_name}" in tex or f"({node_name})" in tex

    for removed_stage in ("stage1", "stage2", "stage3"):
        assert f"name={removed_stage}" not in tex

    for expected_label in (
        "72 \\times |V|",
        "16 \\times |V|",
        "12 \\times |V|",
        "2 classes",
        "Shared CNN Encoder",
        "MixConv + CBAM",
        "[SEP]",
    ):
        assert expected_label in tex

    assert "RightBandedBox" in tex
    assert "Box" in tex
    assert "(posenc-east)" in tex
    assert "(0.90,1.20,0)" in tex
    assert r"{\Huge row + col}\\[4pt]{\Huge $6 \times 12 \times 256$}" in tex


def test_backbone_labels_are_larger_and_clear_the_3d_blocks() -> None:
    module = _load_diagram_module()

    label = module._label("input", "Input", r"RGB $3 \times 48 \times 96$")

    assert r"font=\LARGE" in label
    assert r"$(input-south)+(0,-1.3,0)$" in label
    assert r"\LARGE RGB $3 \times 48 \times 96$" in label


def test_legend_uses_the_37_character_vocabulary() -> None:
    module = _load_diagram_module()
    tex = "".join(module.build_architecture())

    assert r"$|V|=37$" in tex
    assert r"$|V|=38$" not in tex


def test_cli_writes_a_standalone_tex_file(tmp_path: Path) -> None:
    output = tmp_path / "architecture.tex"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    tex = output.read_text(encoding="utf-8")
    assert "\\documentclass[border=8pt, multi, tikz]{standalone}" in tex
    assert str(ROOT / "PlotNeuralNet" / "layers") in tex
    assert tex.rstrip().endswith("\\end{document}")


def test_main_writes_to_an_explicit_destination(tmp_path: Path) -> None:
    module = _load_diagram_module()
    output = tmp_path / "explicit.tex"

    written = module.main(["--output", str(output)])

    assert written == output.resolve()
    assert output.is_file()


def test_main_rejects_a_non_tex_destination(tmp_path: Path) -> None:
    module = _load_diagram_module()

    with pytest.raises(ValueError, match=r"must point to a \.tex file"):
        module.main(["--output", str(tmp_path / "architecture.pdf")])
