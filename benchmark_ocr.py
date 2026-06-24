import time
import torch
import sys
from pathlib import Path
import argparse
import glob

torch.serialization.add_safe_globals([argparse.Namespace])

ROOT = Path(__file__).resolve().parent
LPRNET_ROOT = ROOT / "LPRNet"
if str(LPRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(LPRNET_ROOT))

from lprnet.small_lpr_lightning import SmallLPRLightning
from lprnet.small_lpr_ctc import SmallLPRCTC
from lprnet.small_lpr_nar import SmallLPRNAR
from ocr.parseq_model import load_parseq_checkpoint


def measure_fps(model, dummy_input, device, num_warmup=10, num_iters=200):
    model.eval()
    model.to(device)
    dummy_input = dummy_input.to(device)

    with torch.no_grad():
        for _ in range(num_warmup):
            model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()

    total_time = end_time - start_time
    return num_iters / total_time, total_time


def _find_latest_ctc_ckpt() -> Path | None:
    """Tìm checkpoint CTC mới nhất trong weights/ocr/small_lpr_ctc/."""
    pattern = str(ROOT / "weights" / "ocr" / "small_lpr_ctc" / "**" / "*.ckpt")
    ckpts = sorted(glob.glob(pattern, recursive=True))
    best = [c for c in ckpts if "last" not in Path(c).name]
    return Path(best[-1]) if best else (Path(ckpts[-1]) if ckpts else None)


def _find_latest_nar_ckpt() -> Path | None:
    """Tìm checkpoint NAR mới nhất trong weights/ocr/small_lpr_nar/."""
    pattern = str(ROOT / "weights" / "ocr" / "small_lpr_nar" / "**" / "*.ckpt")
    ckpts = sorted(glob.glob(pattern, recursive=True))
    best = [c for c in ckpts if "last" not in Path(c).name]
    return Path(best[-1]) if best else (Path(ckpts[-1]) if ckpts else None)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}

    # ── 1. SmallLPR (Autoregressive) ──────────────────────────────────────────
    print("-" * 60)
    print("1. Benchmarking SmallLPR (Autoregressive)")
    small_lpr_ckpt = ROOT / "weights" / "ocr" / "small_lpr-epoch=136-val_acc=0.914.ckpt"
    fps_small = None
    if small_lpr_ckpt.exists():
        model_small = SmallLPRLightning.load_from_checkpoint(str(small_lpr_ckpt))
        dummy_small = torch.randn(1, 3, 48, 96)
        fps_small, _ = measure_fps(model_small, dummy_small, device)
        results["SmallLPR (AR)"] = fps_small
        print(f"  → FPS: {fps_small:.1f}")
    else:
        print(f"  Weight not found: {small_lpr_ckpt}")

    # ── 2. PARSeq ─────────────────────────────────────────────────────────────
    print("-" * 60)
    print("2. Benchmarking PARSeq")
    parseq_ckpt = ROOT / "weights" / "ocr" / "parseq" / "parseq_vn_plate_best.pt"
    fps_parseq = None
    if parseq_ckpt.exists():
        model_parseq, ckpt_info = load_parseq_checkpoint(str(parseq_ckpt), device=device)
        h = ckpt_info.get("image_height", 32)
        w = ckpt_info.get("image_width", 128)
        dummy_parseq = torch.randn(1, 3, h, w)
        fps_parseq, _ = measure_fps(model_parseq, dummy_parseq, device)
        results["PARSeq"] = fps_parseq
        print(f"  → FPS: {fps_parseq:.1f}  (Input: {h}×{w})")
    else:
        print(f"  Weight not found: {parseq_ckpt}")

    # ── 3. SmallLPR-CTC (new) ─────────────────────────────────────────────────
    print("-" * 60)
    print("3. Benchmarking SmallLPR-CTC")
    ctc_ckpt = _find_latest_ctc_ckpt()
    if ctc_ckpt is not None:
        print(f"  Loading checkpoint: {ctc_ckpt.name}")
        model_ctc = SmallLPRCTCLightning.load_from_checkpoint(str(ctc_ckpt))
        dummy_ctc = torch.randn(1, 3, 48, 96)
        fps_ctc, _ = measure_fps(model_ctc, dummy_ctc, device)
        results["SmallLPR-CTC"] = fps_ctc
        print(f"  → FPS: {fps_ctc:.1f}")
    else:
        # Chưa có checkpoint → benchmark với random-init model để kiểm tra tốc độ kiến trúc
        print("  Chưa có checkpoint — benchmark với random-weight model (kiểm tra tốc độ kiến trúc)")
        model_ctc_raw = SmallLPRCTC(vocab_size=40)
        dummy_ctc = torch.randn(1, 3, 48, 96)
        fps_ctc, _ = measure_fps(model_ctc_raw, dummy_ctc, device)
        results["SmallLPR-CTC (no ckpt)"] = fps_ctc
        print(f"  → FPS: {fps_ctc:.1f}")

    # ── 4. SmallLPR-NAR (new) ─────────────────────────────────────────────────
    print("-" * 60)
    print("4. Benchmarking SmallLPR-NAR")
    nar_ckpt = _find_latest_nar_ckpt()
    if nar_ckpt is not None:
        print(f"  Loading checkpoint: {nar_ckpt.name}")
        model_nar = SmallLPRNARLightning.load_from_checkpoint(str(nar_ckpt))
        dummy_nar = torch.randn(1, 3, 48, 96)
        fps_nar, _ = measure_fps(model_nar, dummy_nar, device)
        results["SmallLPR-NAR"] = fps_nar
        print(f"  → FPS: {fps_nar:.1f}")
    else:
        print("  Chưa có checkpoint — benchmark với random-weight model")
        model_nar_raw = SmallLPRNAR(vocab_size=40)
        dummy_nar = torch.randn(1, 3, 48, 96)
        fps_nar, _ = measure_fps(model_nar_raw, dummy_nar, device)
        results["SmallLPR-NAR (no ckpt)"] = fps_nar
        print(f"  → FPS: {fps_nar:.1f}")

    # ── Summary Table ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"{'Model':<25} {'FPS':>8}  {'vs SmallLPR-AR':>16}")
    print("-" * 60)
    baseline = results.get("SmallLPR (AR)", None)
    for name, fps in results.items():
        ratio = f"×{fps / baseline:.1f}" if (baseline and name != "SmallLPR (AR)") else "—"
        print(f"  {name:<23} {fps:>8.1f}  {ratio:>16}")
    print("=" * 60)


if __name__ == "__main__":
    from lprnet.small_lpr_ctc_lightning import SmallLPRCTCLightning   # noqa: E402
    from lprnet.small_lpr_nar_lightning import SmallLPRNARLightning    # noqa: E402
    main()
