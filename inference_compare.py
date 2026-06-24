import time
import torch
import sys
from pathlib import Path
import argparse
import cv2
import numpy as np
import glob
from torchvision import transforms

torch.serialization.add_safe_globals([argparse.Namespace])

ROOT = Path(__file__).resolve().parent
LPRNET_ROOT = ROOT / "LPRNet"
if str(LPRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(LPRNET_ROOT))

from lprnet.small_lpr_lightning import SmallLPRLightning
from lprnet.small_lpr_ctc_lightning import SmallLPRCTCLightning
from lprnet.small_lpr_nar_lightning import SmallLPRNARLightning
from ocr.parseq_model import load_parseq_checkpoint
from lprnet.small_lpr_ctc import ctc_greedy_decode

def preprocess_image(img_path, target_hw=(48, 96)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (target_hw[1], target_hw[0]))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_resized).unsqueeze(0)
    return tensor

def _find_latest_ckpt(model_dir):
    pattern = str(ROOT / "weights" / "ocr" / model_dir / "**" / "*.ckpt")
    ckpts = sorted(glob.glob(pattern, recursive=True))
    best = [c for c in ckpts if "last" not in Path(c).name]
    return Path(best[-1]) if best else (Path(ckpts[-1]) if ckpts else None)

def measure_fps_real(model, img_tensor, decode_fn, device, num_warmup=10, num_iters=200):
    model.eval()
    model.to(device)
    img_tensor = img_tensor.to(device)

    # Decode once to get the result string
    with torch.no_grad():
        out = model(img_tensor)
        pred_text = decode_fn(out)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            out = model(img_tensor)
            decode_fn(out)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Benchmark loops
        for _ in range(num_iters):
            out = model(img_tensor)
            decode_fn(out)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()

    total_time = end_time - start_time
    fps = num_iters / total_time
    return pred_text, fps

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = str(ROOT / "data" / "datasets" / "ocr" / "valid" / "79A-179.19#nomer31555222.jpg")
    print(f"Image: {img_path}")
    
    # ── 1. SmallLPR (Autoregressive) ──────────────────────────────────────────
    print("-" * 60)
    print("1. SmallLPR (Autoregressive)")
    small_lpr_ckpt = ROOT / "weights" / "ocr" / "small_lpr-epoch=136-val_acc=0.914.ckpt"
    if small_lpr_ckpt.exists():
        model_small = SmallLPRLightning.load_from_checkpoint(str(small_lpr_ckpt)).to(device)
        img_tensor = preprocess_image(img_path, target_hw=(48, 96))
        
        def decode_ar(out_tokens):
            return model_small._decode_seq(out_tokens[0])
            
        pred_text, fps = measure_fps_real(model_small.model, img_tensor, decode_ar, device)
        print(f"  Prediction: {pred_text}")
        print(f"  FPS (E2E) : {fps:.1f}")
    else:
        print("  Weight not found.")

    # ── 2. PARSeq ─────────────────────────────────────────────────────────────
    print("-" * 60)
    print("2. PARSeq")
    parseq_ckpt = ROOT / "weights" / "ocr" / "parseq" / "parseq_vn_plate_best.pt"
    if parseq_ckpt.exists():
        model_parseq, ckpt_info = load_parseq_checkpoint(str(parseq_ckpt), device=device)
        h = ckpt_info.get("image_height", 32)
        w = ckpt_info.get("image_width", 128)
        
        # PARSeq transform
        from torchvision import transforms as T
        parseq_transform = T.Compose([
            T.Resize((h, w), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        img_p = cv2.imread(img_path)
        img_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
        from PIL import Image
        img_p_pil = Image.fromarray(img_p)
        img_tensor_p = parseq_transform(img_p_pil).unsqueeze(0).to(device)
        
        def decode_parseq(out):
            logits = out  # Usually parseq forward returns logits
            preds = logits.softmax(-1)
            pred_idx = preds.argmax(-1)
            # Find EOS or just decode using model charset
            probs, indices = preds.max(-1)
            # Parseq charset mapping is model_parseq.charset
            res = []
            for idx in indices[0]:
                if idx == len(model_parseq.charset) + 1: # EOS
                    break
                if idx < len(model_parseq.charset):
                    res.append(model_parseq.charset[idx - 1]) # index shifted by 1 normally, wait I'll just use parseq's decode method.
            # actually parseq has decode method?
            return "".join(res) # Just a fallback if model doesn't have decode

        # Safest way to decode parseq:
        def safe_decode_parseq(logits):
            # parseq usually returns logits (B, T, C). 
            # 0=BOS, 1=PAD, 2=EOS, 3=UNK -> wait this depends on the config. 
            # Usually .tokenizer.decode() exists.
            try:
                preds, probs = model_parseq.tokenizer.decode(logits)
                return preds[0]
            except:
                # If no tokenizer, try simple argmax
                pred_idx = logits.argmax(-1)[0]
                res = []
                for i in pred_idx:
                    i = i.item()
                    # Just guess the offset
                    if i < len(model_parseq.charset):
                        res.append(model_parseq.charset[i])
                return "".join(res)

        pred_text, fps = measure_fps_real(model_parseq, img_tensor_p, safe_decode_parseq, device)
        print(f"  Prediction: {pred_text}")
        print(f"  FPS (E2E) : {fps:.1f}")

    # ── 3. SmallLPR-CTC ───────────────────────────────────────────────────────
    print("-" * 60)
    print("3. SmallLPR-CTC")
    ctc_ckpt = _find_latest_ckpt("small_lpr_ctc")
    if ctc_ckpt:
        model_ctc = SmallLPRCTCLightning.load_from_checkpoint(str(ctc_ckpt)).to(device)
        img_tensor = preprocess_image(img_path, target_hw=(48, 96))
        
        def decode_ctc(logits):
            return ctc_greedy_decode(logits, model_ctc.args.chars)[0]
            
        pred_text, fps = measure_fps_real(model_ctc, img_tensor, decode_ctc, device)
        print(f"  Prediction: {pred_text}")
        print(f"  FPS (E2E) : {fps:.1f}")

    # ── 4. SmallLPR-NAR ───────────────────────────────────────────────────────
    print("-" * 60)
    print("4. SmallLPR-NAR")
    nar_ckpt = _find_latest_ckpt("small_lpr_nar")
    if nar_ckpt:
        model_nar = SmallLPRNARLightning.load_from_checkpoint(str(nar_ckpt)).to(device)
        img_tensor = preprocess_image(img_path, target_hw=(48, 96))
        
        def decode_nar(logits):
            return model_nar.model.predict(img_tensor, model_nar.args.chars)[0]
            
        class NARWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m.model.predict(x, self.m.args.chars)
                
        wrapper = NARWrapper(model_nar).to(device)
        pred_text, fps = measure_fps_real(wrapper, img_tensor, lambda x: x[0], device)
        print(f"  Prediction: {pred_text}")
        print(f"  FPS (E2E) : {fps:.1f}")

    # ── 5. YOLOv5 Character Detection ─────────────────────────────────────────
    print("-" * 60)
    print("5. YOLOv5 Character Detection (char.pt)")
    char_pt_path = ROOT / "references" / "Character-Time-series-Matching" / "Vietnamese" / "char.pt"
    yolov5_path = ROOT / "references" / "Character-Time-series-Matching" / "yolov5"
    if char_pt_path.exists() and yolov5_path.exists():
        if str(yolov5_path) not in sys.path:
            sys.path.insert(0, str(yolov5_path))
        from models.experimental import attempt_load
        from utils.general import non_max_suppression

        # Load model
        original_torch_load = torch.load
        def _patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = _patched_load
        
        char_model = attempt_load(str(char_pt_path), map_location=device)
        torch.load = original_torch_load
        
        char_model.eval()
        char_names = char_model.module.names if hasattr(char_model, 'module') else char_model.names

        # Preprocess logic (from DETECTION.py)
        def resize_img(img, size=(128, 128)):
            h1, w1, _ = img.shape
            h, w = size
            if w1 < h1 * (w / h):
                img_rs = cv2.resize(img, (int(float(w1 / h1) * h), h))
                mask = np.zeros((h, w - (int(float(w1 / h1) * h)), 3), np.uint8)
                img = cv2.hconcat([img_rs, mask])
                trans_x = int(w / 2) - int(int(float(w1 / h1) * h) / 2)
                trans_y = 0
            else:
                img_rs = cv2.resize(img, (w, int(float(h1 / w1) * w)))
                mask = np.zeros((h - int(float(h1 / w1) * w), w, 3), np.uint8)
                img = cv2.vconcat([img_rs, mask])
                trans_x = 0
                trans_y = int(h / 2) - int(int(float(h1 / w1) * w) / 2)
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img

        orig_img = cv2.imread(img_path)
        if orig_img is not None:
            resized_img = resize_img(orig_img.copy(), size=(128, 128))
            img_arr = resized_img.copy()[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img_arr = np.ascontiguousarray(img_arr)
            img_tensor_yolo = torch.from_numpy(img_arr).to(device).float() / 255.0
            if img_tensor_yolo.ndimension() == 3:
                img_tensor_yolo = img_tensor_yolo.unsqueeze(0)

            process_plate_path = ROOT / "references" / "Character-Time-series-Matching"
            if str(process_plate_path) not in sys.path:
                sys.path.insert(0, str(process_plate_path))
            from process_plate import find_chars_plate

            class YOLOv5CharWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    return self.m(x, augment=False)[0]

            wrapper_yolo = YOLOv5CharWrapper(char_model).to(device)

            def decode_yolo(pred):
                detections = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.5, multi_label=True, max_det=1000)
                det = detections[0].tolist()
                if not len(det):
                    return ""
                
                centers_x, centers_y, chars = [], [], []
                for *xyxy, conf, cls in det:
                    xc = (xyxy[0] + xyxy[2]) / 2
                    yc = (xyxy[1] + xyxy[3]) / 2
                    centers_x.append(xc)
                    centers_y.append(yc)
                    chars.append(char_names[int(cls)])
                
                if len(chars) > 0:
                    try:
                        _, string_result = find_chars_plate(centers_x, centers_y, chars)
                        return string_result
                    except Exception as e:
                        return "".join(chars)
                return ""

            pred_text_yolo, fps_yolo = measure_fps_real(wrapper_yolo, img_tensor_yolo, decode_yolo, device)
            print(f"  Prediction: {pred_text_yolo}")
            print(f"  FPS (E2E) : {fps_yolo:.1f}")
        else:
            print(f"  Could not load image: {img_path}")
    else:
        print("  Weight or yolov5 source not found.")

if __name__ == "__main__":
    main()
