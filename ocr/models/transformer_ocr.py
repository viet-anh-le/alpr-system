from .components import *


class TransLPRNet(nn.Module):
    """
    Kiến trúc TransLPRNet hoàn chỉnh: PTN → MobileViT Encoder → MiniLMv2 Decoder.

    Pipeline:
    1. PTN: Nắn chỉnh phối cảnh ảnh đầu vào (true perspective warp via H^-1)
    2. MobileViT Encoder: Trích xuất visual tokens (B, 16, 128) từ ảnh 224×224
    3. Memory Adapter: Chiếu tokens từ encoder space (128) sang decoder space (384)
    4. MiniLMv2 Decoder: Giải mã chuỗi ký tự từ visual tokens

    Dimensions:
    - Encoder output: (B, 16, 128) — 4×4 patch pooling × 128 dims (paper Table 1)
    - Memory adapter: Linear(128 → 384) — bridge encoder→decoder
    - Decoder input: (B, 16, 384) — 384 dims tương thích MiniLMv2-L6-H384
    """

    def __init__(
        self,
        vocab_size,
        target_size=(224, 224),
        max_seq_len=14,
        start_token_idx=1,
        use_pretrained=True,
    ):
        super(TransLPRNet, self).__init__()
        self.max_seq_len = max_seq_len
        self.start_token_idx = start_token_idx

        # Encoder output dim = 128, Decoder dim = 384 (MiniLMv2-L6-H384)
        encoder_dim = 128
        decoder_dim = 384

        # 1. PTN — Perspective Transformation Network
        self.ptn = PTN(target_size=target_size)

        # 2. Visual Encoder — MobileViT-small pretrained từ HuggingFace
        self.encoder = MobileViTv3Encoder(
            pretrained=use_pretrained, encoder_dim=encoder_dim
        )

        # 3. Memory Adapter — Bridge encoder (128d) sang decoder (384d)
        self.memory_adapter = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.GELU(),
        )

        # 4. Positional encoding cho memory tokens
        # 16 tokens sau 4×4 patch pooling, dùng 32 để có margin
        self.memory_pe = nn.Parameter(torch.randn(1, 32, decoder_dim) * 0.02)

        # 5. Text Decoder — MiniLMv2-based decoder
        self.decoder = MiniLMv2Decoder(
            vocab_size=vocab_size,
            d_model=decoder_dim,
            nhead=4,
            num_layers=4,
            dim_feedforward=decoder_dim * 4,  # 1536
            max_seq_len=max_seq_len,
        )

        # Load pretrained MiniLMv2 weights và freeze self-attn/FFN layers
        if use_pretrained:
            self.decoder.init_from_pretrained()
            for name, param in self.decoder.named_parameters():
                # Chỉ đóng băng self_attn và các lớp linear của FFN (giữ cross-attn trainable)
                if "self_attn" in name or "linear1" in name or "linear2" in name or "norm" in name:
                    param.requires_grad = False

    def forward(self, images, targets=None):
        # Bước 1: PTN nắn chỉnh hình học
        aligned_images = self.ptn(images)

        # Bước 2: Visual Encoder — (B, 16, 128) sau 4×4 patch pooling
        memory = self.encoder(aligned_images)

        # Bước 3: Memory Adapter — (B, 16, 128) → (B, 16, 384)
        memory = self.memory_adapter(memory)

        # Bước 4: Thêm positional encoding cho memory tokens
        num_tokens = memory.size(1)
        memory = memory + self.memory_pe[:, :num_tokens, :].to(memory.device)

        # Bước 5: Text Decoder
        if targets is not None:
            # Teacher forcing: dùng targets[:, :-1] làm decoder input
            tgt_input = targets[:, :-1]
            logits = self.decoder(tgt_tokens=tgt_input, memory_features=memory)
            return logits
        else:
            # Autoregressive inference: sinh từng token một
            B = memory.size(0)
            device = memory.device
            generated_tokens = torch.full(
                (B, 1), self.start_token_idx, dtype=torch.long, device=device
            )

            for _ in range(self.max_seq_len - 1):
                logits = self.decoder(tgt_tokens=generated_tokens, memory_features=memory)
                next_token_logits = logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            return generated_tokens
