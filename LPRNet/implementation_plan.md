# Kế Hoạch Xây Dựng TransLPRNet cho Biển Số Việt Nam

## Tổng Quan

Dựa trên paper **arXiv:2507.17335v1** và code hiện tại, đây là kế hoạch toàn diện để xây dựng TransLPRNet nhận dạng **biển số 1 dòng và 2 dòng Việt Nam** với độ chính xác > 90%, sử dụng pretrained weights thay vì xây dựng từ đầu.

---

## Kết Quả Nghiên Cứu Kiến Trúc Sẵn Có

### 1. MobileViTv3 (Encoder)

| Nguồn                                    | Trạng thái                                | Cách dùng              |
| ---------------------------------------- | ----------------------------------------- | ---------------------- |
| `github.com/micronDLA/MobileViTv3`       | ✅ Archived (read-only), có code          | Clone và dùng weights  |
| `huggingface.co/apple/mobilevit-small`   | ✅ MobileViT **v1** — **duy nhất có sẵn** | Tải qua `transformers` |
| `huggingface.co/apple/mobilevit-x-small` | ✅ MobileViT **v1 extra-small**           | Tải qua `transformers` |
| `apple/mobilevitv2-small`                | ❌ **Không tồn tại** trên HuggingFace     | —                      |
| timm                                     | ❌ Không support MobileViTv1/v2/v3        | Không dùng             |

> **Quyết định**: Dùng **MobileViT-small (v1)** từ Hugging Face (`apple/mobilevit-small`) làm encoder vì:
>
> - **Đây là DUY NHẤT** pretrained MobileViT có sẵn trên HuggingFace với ImageNet weights
> - MobileViTv2/v3 **không có** model chính thức trên HuggingFace
> - MobileViT-small (v1) đã được Apple open-source với pretrained ImageNet weights, đủ mạnh cho task
> - Nếu muốn dùng MobileViTv3 đúng paper: phải clone `micronDLA/MobileViTv3` (CVNets framework) và tự convert weights — tốn nhiều công
>
> **Nguồn**: `https://huggingface.co/apple/mobilevit-small`
>
> **Output shape** khi input `224×224`:
>
> - `last_hidden_state`: `(B, 640, 7, 7)` — feature map cuối (640 channels)
> - `hidden_states[-1]`: tương tự nếu dùng `output_hidden_states=True`

### 2. MiniLMv2 (Decoder)

| Nguồn                                                                   | Trạng thái               | Cách dùng                     |
| ----------------------------------------------------------------------- | ------------------------ | ----------------------------- |
| `github.com/microsoft/unilm/tree/master/minilm`                         | ✅ Official              | Download weights OneDrive     |
| `huggingface.co/nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large` | ✅ Community upload      | `AutoModel.from_pretrained()` |
| `huggingface.co/microsoft/MiniLM-L6-H384-uncased`                       | ✅ Microsoft official HF | Kiến trúc BERT encoder        |

> **Quyết định**: Code hiện tại đã implement đúng — dùng `nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large` và map trọng số encoder sang decoder. Giữ nguyên strategy này.
>
> **Nguồn**: `https://huggingface.co/nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large`

### 3. PTN (Perspective Transformation Network)

> Code PTN đã được implement từ đầu theo đúng paper (Section 3.2.2). PTN **không có pretrained weights công khai** — không cần clone. Chỉ cần train PTN theo strategy 3-stage.

---

## Phân Tích Vấn Đề Hiện Tại

### Vấn đề kiến trúc Encoder (Cần Fix)

Code hiện tại (`components.py :: MobileViTv3Encoder`) được xây thủ công, **không load được pretrained weights** vì:

1. Không dùng HF `transformers` API
2. Kiến trúc `InvertedResidual` đang dùng từ `torchvision.MobileNetV3` — sai cấu trúc
3. Dimensions không khớp với paper (paper dùng 7×7×320 → 16 tokens × 128 dims; code hiện tại dùng 384 dims)

### Vấn đề Decoder

- `MiniLMv2Decoder(d_model=384)` đúng với HF model
- Nhưng `TransLPRNet` truyền `d_model=384, dim_feedforward=1536` → không match với Table 2 paper (128 dims)
- `memory_pe = nn.Parameter(torch.randn(1, 1024, 384))` kích thước 1024 quá lớn, encoder chỉ output ~49 tokens

### Vấn đề Vocab & Charset

Config hiện tại đã đúng cho biển số Việt Nam (không có Hán tự), `vocab_size = 37`:

- `<PAD>(0) <SOS>(1) <EOS>(2)` + 10 số + 23 chữ cái + `Đ` + `-` + `_`

---

## Kế Hoạch Triển Khai

### Phase 1: Sửa Kiến Trúc Encoder với Pretrained `apple/mobilevit-small`

#### [MODIFY] `lprnet/components.py` — `MobileViTv3Encoder`

Thay toàn bộ `MobileViTv3Encoder` bằng wrapper HuggingFace `MobileViTModel` (v1):

```python
# Nguồn pretrained: https://huggingface.co/apple/mobilevit-small
# (MobileViT v1 — duy nhất có weights chính thức trên HuggingFace)
from transformers import MobileViTModel

class MobileViTv3Encoder(nn.Module):
    """
    Encoder dựa trên MobileViT-small (v1) từ Apple/HuggingFace.
    Source: https://huggingface.co/apple/mobilevit-small
    Kiến trúc tương đồng với MobileViTv3 trong paper.
    """
    def __init__(self, pretrained=True, output_dim=128):
        super().__init__()
        if pretrained:
            # Pull weights từ HuggingFace: apple/mobilevit-small
            self.backbone = MobileViTModel.from_pretrained("apple/mobilevit-small")
        else:
            from transformers import MobileViTConfig
            self.backbone = MobileViTModel(MobileViTConfig())

        # apple/mobilevit-small output shape khi input 224x224:
        # last_hidden_state: (B, 640, 7, 7) — feature map spatial
        # hidden_states[-1]: tương tự
        backbone_out_channels = 640  # mobilevit-small final channels

        # Projection: 640 → 256 → output_dim (128 theo paper)
        self.proj_conv = nn.Conv2d(backbone_out_channels, 256, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.act = nn.GELU()
        self.linear_proj = nn.Linear(256, output_dim)  # 128 theo paper Table 1

    def forward(self, x):
        # MobileViT từ HF nhận 'pixel_values' (B, C, H, W)
        # Lưu ý: apple/mobilevit-small expect input BGR, normalize theo ImageNet
        outputs = self.backbone(pixel_values=x, output_hidden_states=False)

        # last_hidden_state: (B, 640, 7, 7) với input 224x224
        feat_map = outputs.last_hidden_state  # (B, 640, H, W)

        feat_map = self.act(self.bn(self.proj_conv(feat_map)))  # (B, 256, H, W)

        B, C, H, W = feat_map.shape
        # Flatten spatial dims: (B, H*W, 256) → 49 tokens khi H=W=7
        tokens = feat_map.view(B, C, H * W).transpose(1, 2)  # (B, 49, 256)
        tokens = self.linear_proj(tokens)  # (B, 49, 128) — match paper
        return tokens
```

> **Lưu ý**: `apple/mobilevit-small` expect input BGR (không phải RGB thông thường), và normalize theo ImageNet stats. `AutoImageProcessor` của HF xử lý tự động nếu dùng. Với pipeline custom, cần normalize thủ công.

#### Normalization cần điều chỉnh

HF MobileViTv2 cần input normalized theo ImageNet (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).
Code hiện tại normalize theo `(img - 127.5) * 0.0078125` → **phải cập nhật `transform()`** trong DataModule.

---

### Phase 2: Sửa Kiến Trúc Decoder và TransLPRNet

#### [MODIFY] `lprnet/components.py` — `MiniLMv2Decoder`

Điều chỉnh d_model xuống 128 theo đúng paper (Table 2):

```python
class MiniLMv2Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=512, max_seq_len=14):
        ...
        # d_model=128 theo paper Table 2 (không phải 384)
        # nhead=4, num_layers=4, dim_feedforward=512
```

**Chiến lược init_from_pretrained với d_model=128**:

- MiniLMv2 H384 → cần projection từ 384→128
- **Option A**: Dùng `MiniLM-L6-H384` → project down (mất thông tin, nhưng ổn)
- **Option B**: Tìm/dùng MiniLM L6-H128 variant (compact hơn)
  - `microsoft/MiniLM-L12-H384-uncased` → `nn.Linear(384, 128)` adapter
  - Hoặc khởi tạo ngẫu nhiên d_model=128 và dùng knowledge distillation từ 384-model
- **Khuyến nghị**: Giữ d_model=384 (như paper: "L6×H384 MiniLMv2") nhưng adapter encoder output từ 128→384

#### [MODIFY] `lprnet/transLPRNet.py`

```python
class TransLPRNet(nn.Module):
    def __init__(self, vocab_size, target_size=(224,224), max_seq_len=14,
                 start_token_idx=1, use_pretrained=True):
        ...
        # Encoder output: (B, N, 128)
        encoder_dim = 128
        decoder_dim = 384  # MiniLMv2 L6-H384

        self.encoder = MobileViTv3Encoder(pretrained=use_pretrained, output_dim=encoder_dim)

        # Adapter: map encoder tokens từ 128 dim sang decoder 384 dim
        self.memory_adapter = nn.Linear(encoder_dim, decoder_dim)

        self.decoder = MiniLMv2Decoder(
            vocab_size=vocab_size,
            d_model=decoder_dim,  # 384 để load MiniLMv2
            nhead=4,  # paper Table 2: 4 heads
            num_layers=4,
            dim_feedforward=decoder_dim * 4,  # 1536
            max_seq_len=max_seq_len
        )
```

---

### Phase 3: Cập Nhật DataModule - Normalization

#### [MODIFY] `lprnet/trans_datamodule.py`

```python
def transform(self, img):
    """ImageNet normalization cho MobileViTv2 pretrained weights"""
    img = img[:, :, ::-1].astype('float32')  # BGR→RGB
    # ImageNet normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    return img

def resize_to_224(self, img):
    """Thêm logic resize đặc biệt cho biển số 2 dòng"""
    h, w = img.shape[:2]
    if h > w:  # Biển dọc (2 dòng thường cao hơn rộng)
        # Giữ aspect ratio, pad để thành 224x224
        img = resize_letterbox(img, (224, 224))
    else:
        img = cv2.resize(img, (224, 224))
    return img
```

---

### Phase 4: Chiến Lược Training (SOTA)

#### 4.1. Chiến lược fine-tuning với pretrained weights (Differential LR)

Code hiện tại đã implement differential LR tốt. Cần **tinh chỉnh thêm**:

```python
def configure_optimizers(self):
    param_groups = [
        # PTN: train từ đầu, LR cao
        {"params": ptn_params, "lr": args.lr},

        # Encoder backbone: pretrained, LR rất thấp (5-10x nhỏ hơn)
        {"params": encoder_backbone_params, "lr": args.lr * 0.05},

        # Encoder projection head: khởi tạo mới, LR trung bình
        {"params": encoder_proj_params, "lr": args.lr * 0.5},

        # Memory adapter: mới, LR cao
        {"params": memory_adapter_params, "lr": args.lr},

        # Decoder frozen layers (self-attn, FFN): LR rất thấp
        {"params": decoder_frozen_params, "lr": args.lr * 0.01},

        # Decoder new layers (cross-attn, output): mới, LR cao
        {"params": decoder_new_params, "lr": args.lr},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
```

#### 4.2. Chiến lược 3 giai đoạn (Theo paper + SOTA)

**Giai đoạn 1 (Epoch 1–30): Warm-up — Train các lớp mới**

- Freeze backbone encoder, freeze decoder self-attn/FFN
- Chỉ train: PTN, memory_adapter, cross-attention, output head
- LR = 1e-3 (scratch params), Scheduler = LinearWarmup (5 epochs)
- Mục tiêu: Hội tụ nhanh, khớp embedding space

**Giai đoạn 2 (Epoch 31–100): Fine-tune toàn bộ**

- Unfreeze encoder backbone với LR rất thấp (5e-6)
- Unfreeze decoder với LR thấp (1e-5)
- Gradient clipping = 1.0
- Mục tiêu: Adapter mới → backbone quen với task

**Giai đoạn 3 (Epoch 101–200): Fine-tune với augmentation mạnh**

- Tất cả layers trainable, LR nhỏ (Cosine Annealing)
- Bật augmentation nặng hơn (perspective, blur, noise)

#### 4.3. Data Augmentation (SOTA)

```python
# Theo paper Section 4.3 + SOTA techniques
augmentations = [
    RandomResizedCrop(224, scale=(0.8, 1.0)),    # Simulate localization errors
    RandomRotation(degrees=15),                   # Rotation variation
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # Lighting
    RandomPerspective(distortion_scale=0.3),      # Camera angle
    RandomErasing(p=0.3, scale=(0.02, 0.15)),     # Occlusion
    GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Motion blur
    RandomHorizontalFlip(p=0.0),                  # Biển số KHÔNG flip
]

# Test-Time Augmentation (TTA) — không cần cho inference
```

#### 4.4. Loss Function

```python
# Label Smoothing CrossEntropy (đã implement) + thêm:
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

# Tùy chọn thêm: Focal Loss để tập trung vào class khó (Đ, số dễ nhầm)
```

#### 4.5. Scheduler SOTA

```python
from torch.optim.lr_scheduler import OneCycleLR

# OneCycleLR: SOTA cho fine-tuning, nhanh hơn CosineAnnealing
scheduler = OneCycleLR(
    optimizer,
    max_lr=[lr*1, lr*0.05, lr*0.5, lr*1, lr*0.01, lr*1],  # per group
    total_steps=max_epochs * len(train_dataloader),
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos'
)
```

---

### Phase 5: Config Việt Nam Cụ Thể

#### [MODIFY] `config/trans_vietnam_config.yaml`

```yaml
# Biển số Việt Nam — không có Hán tự
# Biển 1 dòng: "29A-12345" (8-9 chars)
# Biển 2 dòng: "29" + "MĐ1-12345" (9-10 chars, ký tự Đ đặc biệt)
# Biển xe máy: "29-AB-234.56" (~11 chars)

max_seq_len: 14 # 12 chars + SOS + EOS (đủ cho mọi loại)
img_size: [224, 224]
batch_size: 32
lr: 0.0001
max_epochs: 200
weight_decay: 0.0001 # tăng từ 0.00002
gradient_clip_val: 1.0
warmup_epochs: 10
```

#### Charset không cần Hán tự (đã đúng)

```yaml
chars: ["<PAD>", "<SOS>", "<EOS>",
        "0"-"9",               # 10 chars
        "A","B","C","D","E","F","G","H","K","L","M","N",
        "P","Q","R","S","T","U","V","X","Y","Z",  # 22 chars
        "Đ",                  # ký tự đặc biệt Việt Nam
        "-", "_"]             # dấu gạch & CTC blank
# Total: 37 chars
```

---

### Phase 6: Xử Lý Biển Số 2 Dòng

#### Chiến lược encode biển 2 dòng

Paper dùng cách đơn giản: **đọc theo thứ tự từ trái sang phải, trên xuống dưới**, nối 2 dòng thành 1 chuỗi. Không cần xử lý đặc biệt.

```
Biển 2 dòng:      29
                  MĐ1-12345
→ Label: "29MĐ1-12345"  (max 11 chars, trong tầm max_seq_len=14)
```

Transformer encoder-decoder với **self-attention toàn cục** trên 2D feature map tự học được spatial relationship giữa 2 dòng — đây là ưu điểm lớn so với CTC+CRNN.

---

## Kế Hoạch Pull/Clone Pretrained Weights

| Model                    | Nguồn                               | Lệnh                                                                                                                              |
| ------------------------ | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **MobileViT-small (v1)** | HuggingFace `apple/mobilevit-small` | `from transformers import MobileViTModel; model = MobileViTModel.from_pretrained("apple/mobilevit-small")`                        |
| **MiniLMv2-L6-H384**     | HuggingFace `nreimers/...`          | `from transformers import AutoModel; model = AutoModel.from_pretrained("nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large")` |

> **Ghi chú quan trọng**: `apple/mobilevitv2-small` **KHÔNG tồn tại** trên HuggingFace. Chỉ có `apple/mobilevit-small` và `apple/mobilevit-x-small` (cả hai đều là v1). MobileViTv2/v3 chưa có pretrained weights công khai tương thích với HF transformers.

---

## Verification Plan

### Automated Tests

1. `python train_trans.py` — training chạy không lỗi
2. `val_acc > 0.50` sau epoch 10 (convergence check)
3. `val_acc > 0.90` sau epoch 200 (target accuracy)
4. Kiểm tra riêng biển 1 dòng vs 2 dòng

### Manual Verification

- Chạy inference trên ảnh test 1 dòng + 2 dòng
- Kiểm tra PTN có hiệu chỉnh góc đúng không (visualize warped image)
- So sánh val_acc giữa experiment A (không pretrained) và D (cả hai pretrained)

---

## User Review Required

> [!IMPORTANT]
> **Quyết định kiến trúc — đã làm rõ sau khi xác nhận với user:**
>
> **Option A ✅ (Sẽ implement)**: Dùng **`apple/mobilevit-small` (v1)** từ HuggingFace — **model MobileViT duy nhất có pretrained weights công khai**. Kiến trúc Transformer hybrid tương đồng với v3, đủ mạnh cho task.
>
> **Option B**: Clone `micronDLA/MobileViTv3` (CVNets framework) + tự convert weights thủ công → phức tạp, không cần thiết vì v1 đủ mạnh.
>
> `apple/mobilevitv2-small` **KHÔNG tồn tại** — đã được user xác nhận.

> [!WARNING]
> **Encoder dimension mismatch**: Paper dùng `encoder_output=128 dim, 16 tokens`. MobileViTv2 từ HF output `~512 dim, ~49 tokens`. Cần adapter layer để bridge. Điều này làm tăng trainable params nhưng không ảnh hưởng đến accuracy.

> [!NOTE]
> **Training cost estimate**: Với 200 epochs, batch_size=32, dataset khoảng vài chục nghìn ảnh → ~4-8 giờ trên GPU TITAN X (giống paper). Có thể bắt đầu với 50 epochs để validate architecture trước.
