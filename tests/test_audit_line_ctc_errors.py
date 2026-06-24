from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "audit_line_ctc_errors.py"
SPEC = importlib.util.spec_from_file_location("audit_line_ctc_errors", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_prepare_dataset_for_audit_disables_train_augmentation_by_default() -> None:
    dataset = SimpleNamespace(transform=object(), img_paths=["b.jpg", "a.jpg"])

    prepared = MODULE.prepare_dataset_for_audit(
        dataset,
        split="train",
        use_train_augment=False,
    )

    assert prepared is dataset
    assert prepared.transform is None
    assert prepared.img_paths == ["a.jpg", "b.jpg"]


def test_prepare_dataset_for_audit_can_keep_train_augmentation() -> None:
    transform = object()
    dataset = SimpleNamespace(transform=transform, img_paths=["b.jpg", "a.jpg"])

    prepared = MODULE.prepare_dataset_for_audit(
        dataset,
        split="train",
        use_train_augment=True,
    )

    assert prepared.transform is transform
    assert prepared.img_paths == ["b.jpg", "a.jpg"]
