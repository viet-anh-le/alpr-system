import argparse, sys
from pathlib import Path
import cv2, numpy as np, torch

ROOT = Path("/home/vietanh/Documents/DATN/ALPR_Vietnamese")
sys.path.insert(0, str(ROOT / "LPRNet"))
torch.serialization.add_safe_globals([argparse.Namespace])

from lprnet.small_lpr_ctc_lightning import SmallLPRCTCLightning
from lprnet.small_lpr_ctc import ctc_greedy_decode
from lprnet.small_lpr import smart_resize

ckpt = (
    ROOT
    / "weights/ocr/small_lpr_ctc/ctc_20260609_155238/small_lpr_ctc-epoch=055-val_acc=0.9358.ckpt"
)
imgs = ["/home/vietanh/Pictures/Screenshots/Screenshot from 2026-06-16 16-39-50.png"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallLPRCTCLightning.load_from_checkpoint(str(ckpt), map_location=device).to(device).eval()


def preprocess(path):
    img = cv2.imread(path)
    img = smart_resize(img, target_hw=(48, 96)).astype(np.float32)
    img = (img - 127.5) * 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).to(device)


for path in imgs:
    with torch.no_grad():
        logits = model(preprocess(path))
        pred = ctc_greedy_decode(logits, model.args.chars)[0]
    print(f"{Path(path).name}: {pred} | display: {pred.replace('[SEP]', ' / ')}")
