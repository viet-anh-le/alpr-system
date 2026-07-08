import os
import sys
from pathlib import Path
import cv2
import torch
import argparse
import yaml

ROOT = Path("/home/vietanh/Documents/DATN/ALPR_Vietnamese")
LPRNET_ROOT = ROOT / "LPRNet"
if str(LPRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(LPRNET_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lprnet.small_lpr_line_ctc_lightning import SmallLPRLineCTCLightning
from lprnet.small_lpr import smart_resize
from lprnet.small_lpr_line_ctc import line_ctc_greedy_decode

def load_model(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hyper_parameters = payload.get("hyper_parameters", {})
    
    if isinstance(hyper_parameters, dict) and "args" in hyper_parameters:
        ckpt_args = hyper_parameters["args"]
    elif isinstance(hyper_parameters, dict):
        ckpt_args = hyper_parameters
    else:
        ckpt_args = getattr(hyper_parameters, "args", hyper_parameters)

    config_path = ROOT / "LPRNet/config/small_lpr_line_ctc_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    from argparse import Namespace
    args = Namespace(**cfg)

    # Overwrite args with checkpoint args
    for name in (
        "chars", "d_model", "backbone_ch", "line_prior_strength", 
        "use_stn", "use_pos_enc", "use_global_head", "two_line_threshold", 
        "global_loss_weight", "one_line_loss_weight", "top_loss_weight", 
        "bottom_loss_weight", "layout_loss_weight", "label_mode", "line_separator"
    ):
        val = None
        if isinstance(ckpt_args, dict):
            val = ckpt_args.get(name)
        elif hasattr(ckpt_args, name):
            val = getattr(ckpt_args, name)
            
        if val is not None:
            setattr(args, name, val)

    model = SmallLPRLineCTCLightning(args).to(device).eval()
    
    # Tự động xử lý nếu checkpoint có 257 chiều nhưng model hiện tại chỉ có 256 chiều
    state_dict = payload["state_dict"]
    if "model.line_attention.weight" in state_dict:
        ckpt_shape = state_dict["model.line_attention.weight"].shape
        model_shape = model.model.line_attention.weight.shape
        if ckpt_shape[1] == 257 and model_shape[1] == 256:
            print("WARNING: Phát hiện checkpoint 257 chiều. Khôi phục lại hàm _line_features để nối tọa độ Y (fair evaluation).")
            import torch.nn as nn
            model.model.line_attention = nn.Conv2d(257, 2, kernel_size=1).to(device)
            
            def patched_line_features(self, feat):
                import torch
                feat_bdhw = feat.permute(0, 3, 1, 2)
                batch, _, height, width = feat_bdhw.shape
                y = torch.linspace(-1.0, 1.0, height, device=feat.device, dtype=feat.dtype).view(1, 1, height, 1).expand(batch, 1, height, width)
                feat_with_y = torch.cat([feat_bdhw, y], dim=1)
                attn_logits = self.line_attention(feat_with_y)
                if hasattr(self, "line_prior_strength") and self.line_prior_strength != 0.0:
                    prior_y = torch.linspace(-1.0, 1.0, height, device=feat.device, dtype=feat.dtype).view(1, 1, height, 1)
                    prior = torch.cat((-prior_y, prior_y), dim=1) * self.line_prior_strength
                    attn_logits = attn_logits + prior
                attention = torch.softmax(attn_logits, dim=2)
                line_feat = torch.einsum("bdhw,blhw->blwd", feat_bdhw, attention)
                return line_feat, attention
                
            import types
            model.model._line_features = types.MethodType(patched_line_features, model.model)

    model.load_state_dict(state_dict, strict=True)
    return model, args

def clean_text(text: str) -> str:
    text = text.replace("[SEP]", "")
    text = text.replace("-", "")
    text = text.replace(".", "")
    return text.upper()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, args_model = load_model(args_cli.checkpoint, device)

    dataset_dir = Path(args_cli.dataset)
    image_paths = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))
    
    target_hw = (args_model.img_size[1], args_model.img_size[0])

    correct = 0
    total = 0
    errors = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_resized = smart_resize(img, target_hw=target_hw)
        img_norm = img_resized.astype("float32")
        img_norm = (img_norm - 127.5) * 0.0078125
        img_norm = img_norm.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0).to(device)
        
        basename = img_path.name
        # The license plate text is typically the first part of the filename before any underscore
        raw_label = img_path.stem.split("_")[0].split("#")[0]
        cleaned_label = clean_text(raw_label)

        with torch.no_grad():
            outputs = model(img_tensor)
            pred_texts = line_ctc_greedy_decode(
                outputs, 
                args_model.chars, 
                two_line_threshold=getattr(args_model, "two_line_threshold", 0.5), 
                line_separator=getattr(args_model, "line_separator", "[SEP]")
            )
            raw_pred = pred_texts[0]
            cleaned_pred = clean_text(raw_pred)
            
            total += 1
            if cleaned_pred == cleaned_label:
                correct += 1
            else:
                errors.append(f"File: {basename}\n  Target (Cleaned): {cleaned_label}\n  Pred (Cleaned)  : {cleaned_pred}\n  Target (Raw)    : {raw_label}\n  Pred (Raw)      : {raw_pred}\n")

    acc = correct / total if total > 0 else 0
    
    with open(args_cli.output, "w", encoding="utf-8") as f:
        f.write(f"Total: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("\nErrors:\n")
        for error in errors:
            f.write(error + "\n")

    print(f"Evaluated {total} samples. Accuracy: {acc:.4f}")
    print(f"Results saved to {args_cli.output}")

if __name__ == '__main__':
    main()
