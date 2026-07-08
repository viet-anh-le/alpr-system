import torch
import argparse
from pathlib import Path

def average_checkpoints(ckpt1_path, ckpt2_path, output_path, alpha=0.5):
    print(f"Loading checkpoint 1: {ckpt1_path}")
    ckpt1 = torch.load(ckpt1_path, map_location="cpu", weights_only=False)
    
    print(f"Loading checkpoint 2: {ckpt2_path}")
    ckpt2 = torch.load(ckpt2_path, map_location="cpu", weights_only=False)
    
    sd1 = ckpt1["state_dict"]
    sd2 = ckpt2["state_dict"]
    
    # Handle the 257-dim anomaly in ckpt2 if present
    if "model.line_attention.weight" in sd2:
        if sd2["model.line_attention.weight"].shape[1] == 257 and sd1["model.line_attention.weight"].shape[1] == 256:
            print("Slicing ckpt2 line_attention from 257 to 256 to match ckpt1...")
            sd2["model.line_attention.weight"] = sd2["model.line_attention.weight"][:, :256, :, :]
            
    # Averaging
    print(f"Averaging weights with alpha={alpha} (alpha * ckpt1 + (1-alpha) * ckpt2)...")
    averaged_sd = {}
    for key in sd2.keys():
        if key in sd1 and sd1[key].shape == sd2[key].shape:
            averaged_sd[key] = alpha * sd1[key] + (1.0 - alpha) * sd2[key]
        else:
            # Fallback to ckpt2 if ckpt1 doesn't have it or has a shape mismatch (e.g., vocab size diff)
            averaged_sd[key] = sd2[key]
            
    # Save the averaged checkpoint using ckpt2 as the base so the hyper_parameters match the 37-dim heads
    ckpt2["state_dict"] = averaged_sd
    # Remove val_acc from callbacks so it doesn't cause confusion
    if "callbacks" in ckpt2:
        del ckpt2["callbacks"]
        
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt2, out_file)
    print(f"Saved averaged checkpoint to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt1", required=True, help="Path to epoch 8 (95%)")
    parser.add_argument("--ckpt2", required=True, help="Path to epoch 43 (93.9%)")
    parser.add_argument("--out", required=True, help="Output path")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for ckpt1 (default: 0.5)")
    args = parser.parse_args()
    
    average_checkpoints(args.ckpt1, args.ckpt2, args.out, args.alpha)
