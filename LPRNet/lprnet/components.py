import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PART 1: PTN (Perspective Transformation Network) — Giữ nguyên từ paper
# =============================================================================


class VertexRegressionNet(nn.Module):
    """
    Mạng nơ-ron tích chập (CNN) chịu trách nhiệm hồi quy 4 đỉnh của biển số xe.
    Cấu trúc mạng tuân thủ chặt chẽ Bảng 3 trong tài liệu TransLPRNet.
    Đầu vào: Ảnh kích thước 94x24 (Width x Height), tương đương Tensor (B, 3, 24, 94).
    Đầu ra: 8 giá trị đại diện cho (x, y) của 4 đỉnh.
    """

    def __init__(self):
        super(VertexRegressionNet, self).__init__()

        self.features = nn.Sequential(
            # Conv 3x3: (B, 3, 24, 94) -> (B, 32, 22, 92)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            # MaxPool 2x2: (B, 32, 22, 92) -> (B, 32, 11, 46)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # Conv 5x5: (B, 32, 11, 46) -> (B, 32, 7, 42)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0),
            # Conv 3x3: (B, 32, 7, 42) -> (B, 32, 7, 42)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            # Conv 3x3: (B, 32, 7, 42) -> (B, 64, 7, 42)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # Conv 3x3: (B, 64, 7, 42) -> (B, 128, 7, 42)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # Conv 1x1: (B, 128, 7, 42) -> (B, 64, 7, 42)
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # Conv 3x3: (B, 64, 7, 42) -> (B, 32, 7, 42)
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            # MaxPool 3x3, stride 3: (B, 32, 7, 42) -> (B, 32, 2, 14)
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # Mạng kết nối đầy đủ (Fully Connected Layers)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 14, 2084),  # 896 -> 2084
            nn.ReLU(inplace=True),
            nn.Linear(2084, 32),  # 2084 -> 32
            nn.ReLU(inplace=True),
            nn.Linear(32, 8),  # 32 -> 8 (tọa độ 4 đỉnh)
        )

        # Khởi tạo trọng số lớp cuối cùng để mô hình bắt đầu bằng việc
        # tạo ra một ánh xạ không biến đổi (Identity Mapping),
        # tương ứng với 4 góc của toàn bộ bức ảnh [-1, 1]
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.copy_(
            torch.tensor([-1, -1, 1, -1, -1, 1, 1, 1], dtype=torch.float)
        )

    def forward(self, x):
        features = self.features(x)
        corners = self.regressor(features)
        return corners


class PerspectiveTransformSolver(nn.Module):
    """
    Mô-đun giải phương trình toán học tính toán ma trận phối cảnh và
    thực hiện lấy mẫu lưới (Grid Sampling) khả vi.
    """

    def __init__(self, target_size=(224, 224)):
        super(PerspectiveTransformSolver, self).__init__()
        self.target_size = target_size

        # Tọa độ không gian đích chuẩn hóa [-1, 1]
        self.register_buffer(
            "target_corners",
            torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        )

    def forward(self, image_tensor, source_corners):
        B = image_tensor.size(0)
        device = image_tensor.device

        # Thiết lập cấu trúc hệ phương trình DLT: M * Theta = U
        M = torch.zeros((B, 8, 8), device=device, dtype=torch.float32)
        U = torch.zeros((B, 8, 1), device=device, dtype=torch.float32)

        src = source_corners.view(B, 4, 2)
        tgt = self.target_corners.unsqueeze(0).expand(B, -1, -1)

        for i in range(4):
            sx, sy = src[:, i, 0], src[:, i, 1]
            tx, ty = tgt[:, i, 0], tgt[:, i, 1]

            M[:, 2 * i, 0] = sx
            M[:, 2 * i, 1] = sy
            M[:, 2 * i, 2] = 1.0
            M[:, 2 * i, 6] = -tx * sx
            M[:, 2 * i, 7] = -tx * sy

            M[:, 2 * i + 1, 3] = sx
            M[:, 2 * i + 1, 4] = sy
            M[:, 2 * i + 1, 5] = 1.0
            M[:, 2 * i + 1, 6] = -ty * sx
            M[:, 2 * i + 1, 7] = -ty * sy

            U[:, 2 * i, 0] = tx
            U[:, 2 * i + 1, 0] = ty

        theta = torch.linalg.solve(M, U)

        ones = torch.ones((B, 1, 1), device=device, dtype=torch.float32)
        theta_3x3 = torch.cat([theta, ones], dim=1).view(B, 3, 3)

        # True perspective warp: invert H (src→tgt) to get H_inv (tgt→src) for grid_sample
        theta_inv = torch.linalg.inv(theta_3x3)  # (B, 3, 3)

        H_out, W_out = self.target_size[1], self.target_size[0]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H_out, device=device, dtype=torch.float32),
            torch.linspace(-1, 1, W_out, device=device, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack(
            [grid_x.flatten(), grid_y.flatten(),
             torch.ones(H_out * W_out, device=device, dtype=torch.float32)],
            dim=0,
        )  # (3, H*W)

        src_coords = theta_inv @ coords.unsqueeze(0)  # (B, 3, H*W)
        src_xy = src_coords[:, :2] / (src_coords[:, 2:3] + 1e-8)  # (B, 2, H*W)
        grid = src_xy.permute(0, 2, 1).view(B, H_out, W_out, 2)  # (B, H, W, 2)

        warped_image = F.grid_sample(
            image_tensor, grid, align_corners=True, mode="bilinear", padding_mode="border"
        )
        return warped_image


class PTN(nn.Module):
    """
    Mạng Hiệu chỉnh Phối cảnh hoàn chỉnh (Perspective Transformation Network).
    Bao bọc cả quá trình hồi quy và biến đổi không gian.
    """

    def __init__(self, target_size=(224, 224)):
        super(PTN, self).__init__()
        self.regression_net = VertexRegressionNet()
        self.transform_solver = PerspectiveTransformSolver(target_size=target_size)

    def forward(self, x):
        x_resized = F.interpolate(x, size=(24, 94), mode="bilinear", align_corners=True)
        corners = self.regression_net(x_resized)

        aligned_image = self.transform_solver(x, corners)
        return aligned_image


# =============================================================================
# PART 2: ENCODER — MobileViT-small (v1) từ HuggingFace
# Nguồn pretrained: https://huggingface.co/apple/mobilevit-small
# =============================================================================


class MobileViTv3Encoder(nn.Module):
    """
    MobileViTv3 Encoder - sử dụng kiến trúc CVNets và pretrained weights từ Github micronDLA.
    Pretrained variant: MobileViTv3-S (small_v3) trained on ImageNet-1K.

    Output shape: (B, num_tokens, encoder_dim)
    """

    def __init__(self, pretrained=True, encoder_dim=128):
        super(MobileViTv3Encoder, self).__init__()

        # Sử dụng patch CVNets để khởi tạo MobileViTv3 
        import sys
        import os
        cvnets_path = os.path.abspath("lprnet/cvnets_core")
        if cvnets_path not in sys.path:
            sys.path.insert(0, cvnets_path)
            
        from cvnets.models.classification.mobilevit import MobileViTv3

        # 1. Giả lập args theo CVNets "small_v3"
        class OptsObj: pass
        opts_obj = OptsObj()
        args = {
            'model.classification.mit.mode': 'small_v3',
            'model.classification.mitv3.width_multiplier': 1.0,
            'model.classification.mitv3.attn_norm_layer': 'layer_norm_2d',
            'model.classification.mitv3.ffn_dropout': 0.0,
            'model.classification.mitv3.attn_dropout': 0.0,
            'model.classification.mitv3.dropout': 0.1,
            'model.classification.mitv3.number_heads': 4,
            'model.classification.mitv3.no_fusion': False,
            'model.classification.activation.name': 'swish',
            'model.classification.classifier_dropout': 0.1,
            'model.classification.mitv3.transformer_dropout': 0.1,
            'model.normalization.name': 'batch_norm_2d',
            'model.normalization.momentum': 0.1,
            'model.activation.name': 'swish',
            'model.layer.global_pool': 'mean',
            'model.classification.n_classes': 1000,
            'model.classification.mitv3.conv_ksize': 3,
            'model.classification.mitv3.head_dim': 32
        }
        for k, v in args.items():
            setattr(opts_obj, k, v)
        
        # 2. Khởi tạo backbone
        print("Initializing CVNets MobileViTv3-S (small_v3 mode)...")
        self.backbone = MobileViTv3(opts_obj)
        
        # 3. Load pretrained weights nếu được yêu cầu
        if pretrained:
            ckpt_path = "lprnet/cvnets_core/mobilevitv3_s.pt"
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
                
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
                print(f"✅ Loaded MobileViTv3 pretrained weights from {ckpt_path}! (Missing: {len(missing)} params)")
            except Exception as e:
                print(f"⚠️ Warning: Could not load pretrained weights ({e}). Training from scratch!")
                
        # Trong MobileViTv3-S "small_v3", đầu ra output convolution `conv_1x1_exp` là 1280.
        backbone_out_dim = getattr(self.backbone.conv_1x1_exp, 'out_channels', 1280)
        
        self.proj_conv = nn.Conv2d(backbone_out_dim, 256, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.act = nn.GELU()
        self.linear_proj = nn.Linear(256, encoder_dim)

    def forward(self, x):
        # Trích xuất feature qua interface của CVNets
        res = self.backbone.extract_features(x)
        # res có thể chứa out_l4, out_l5 hoặc 'out' nếu có các hook
        # Tuy nhiên ta có thể tự gọi đến conv_1x1_exp vì extract_features có thể bỏ lỡ nó
        x = self.backbone.conv_1(x) 
        x = self.backbone.layer_1(x)
        x = self.backbone.layer_2(x) 
        x = self.backbone.layer_3(x) 
        x = self.backbone.layer_4(x) 
        x = self.backbone.layer_5(x) 
        
        feat_map = self.backbone.conv_1x1_exp(x) # (B, 1280, 7, 7)
        
        # Projection: 1280 → 256 → encoder_dim (128)
        feat_map = self.act(self.bn(self.proj_conv(feat_map)))  # (B, 256, 7, 7)
        
        # 2x2 patch pooling: 7x7 → 4x4 = 16 tokens (paper Table 1: patch embedding → 16x128)
        feat_map = F.adaptive_avg_pool2d(feat_map, (4, 4))  # (B, 256, 4, 4)
        B, C, H, W = feat_map.shape
        tokens = feat_map.view(B, C, H * W).transpose(1, 2)  # (B, 16, 256)
        tokens = self.linear_proj(tokens)                     # (B, 16, 128)

        return tokens


# =============================================================================
# PART 3: DECODER — MiniLMv2 (L6-H384) từ HuggingFace
# Nguồn pretrained: https://huggingface.co/nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large
# =============================================================================


class MiniLMv2Decoder(nn.Module):
    """
    Bộ giải mã dựa trên cấu trúc MiniLMv2, tích hợp khả năng tải
    pre-trained weights trực tiếp từ thư viện Hugging Face Transformers.

    Theo paper Table 2:
    - d_model = 384 (để tương thích với MiniLMv2-L6-H384 pretrained)
    - nhead = 4 (paper Table 2 ghi 4 attention heads)
    - num_layers = 4 (paper: chỉ giữ 4 layers đầu tiên)
    - dim_feedforward = 1536 (384 × 4)
    """

    def __init__(
        self, vocab_size, d_model=384, nhead=4, num_layers=4, dim_feedforward=1536, max_seq_len=14
    ):
        super(MiniLMv2Decoder, self).__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, self.d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def init_from_pretrained(
        self, hf_model_name="nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large"
    ):
        """
        Ánh xạ trọng số từ Hugging Face MiniLMv2 sang PyTorch TransformerDecoder.
        Yêu cầu thiết lập d_model=384.
        """
        if self.d_model != 384:
            print(
                f"Lỗi: d_model hiện tại là {self.d_model}. Bắt buộc phải là 384 để load MiniLMv2."
            )
            return

        try:
            from transformers import AutoModel

            print(f"Đang tải pre-trained weights từ {hf_model_name}...")
            hf_model = AutoModel.from_pretrained(hf_model_name)

            hf_state = hf_model.state_dict()
            own_state = self.state_dict()

            mapped_tensors = 0
            num_layers = len(self.transformer_decoder.layers)

            # Lặp qua từng Transformer Layer (Encoder của MiniLM -> Decoder của PyTorch)
            for i in range(num_layers):
                hf_prefix = f"encoder.layer.{i}."
                pt_prefix = f"transformer_decoder.layers.{i}."

                try:
                    # 1. Gộp Q, K, V cho Self-Attention
                    q_w = hf_state[f"{hf_prefix}attention.self.query.weight"]
                    k_w = hf_state[f"{hf_prefix}attention.self.key.weight"]
                    v_w = hf_state[f"{hf_prefix}attention.self.value.weight"]
                    own_state[f"{pt_prefix}self_attn.in_proj_weight"].copy_(
                        torch.cat([q_w, k_w, v_w], dim=0)
                    )

                    q_b = hf_state[f"{hf_prefix}attention.self.query.bias"]
                    k_b = hf_state[f"{hf_prefix}attention.self.key.bias"]
                    v_b = hf_state[f"{hf_prefix}attention.self.value.bias"]
                    own_state[f"{pt_prefix}self_attn.in_proj_bias"].copy_(
                        torch.cat([q_b, k_b, v_b], dim=0)
                    )

                    # 2. Đầu ra của Self-Attention (Out Proj & Norm 1)
                    own_state[f"{pt_prefix}self_attn.out_proj.weight"].copy_(
                        hf_state[f"{hf_prefix}attention.output.dense.weight"]
                    )
                    own_state[f"{pt_prefix}self_attn.out_proj.bias"].copy_(
                        hf_state[f"{hf_prefix}attention.output.dense.bias"]
                    )
                    own_state[f"{pt_prefix}norm1.weight"].copy_(
                        hf_state[f"{hf_prefix}attention.output.LayerNorm.weight"]
                    )
                    own_state[f"{pt_prefix}norm1.bias"].copy_(
                        hf_state[f"{hf_prefix}attention.output.LayerNorm.bias"]
                    )

                    # 3. Feed Forward Network (Linear 1, Linear 2 & Norm 3)
                    own_state[f"{pt_prefix}linear1.weight"].copy_(
                        hf_state[f"{hf_prefix}intermediate.dense.weight"]
                    )
                    own_state[f"{pt_prefix}linear1.bias"].copy_(
                        hf_state[f"{hf_prefix}intermediate.dense.bias"]
                    )

                    own_state[f"{pt_prefix}linear2.weight"].copy_(
                        hf_state[f"{hf_prefix}output.dense.weight"]
                    )
                    own_state[f"{pt_prefix}linear2.bias"].copy_(
                        hf_state[f"{hf_prefix}output.dense.bias"]
                    )

                    own_state[f"{pt_prefix}norm3.weight"].copy_(
                        hf_state[f"{hf_prefix}output.LayerNorm.weight"]
                    )
                    own_state[f"{pt_prefix}norm3.bias"].copy_(
                        hf_state[f"{hf_prefix}output.LayerNorm.bias"]
                    )

                    mapped_tensors += 14
                except KeyError as e:
                    print(f"Bỏ qua một số trọng số ở layer {i} do không tìm thấy: {e}")

            print(f"Đã nội suy thành công {mapped_tensors} tensor cốt lõi từ MiniLMv2.")
            print(
                "(Các lớp Cross-attention và Embedding đang giữ nguyên khởi tạo ban đầu để huấn luyện)."
            )

        except ImportError:
            print("Lỗi: Vui lòng cài đặt thư viện 'transformers'.")

    def generate_causal_mask(self, seq_len, device):
        mask = (torch.triu(torch.ones(seq_len, seq_len, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt_tokens, memory_features, tgt_key_padding_mask=None):
        B, L = tgt_tokens.size()
        device = tgt_tokens.device

        tgt_emb = self.embedding(tgt_tokens) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float32, device=device)
        )
        tgt_emb = tgt_emb + self.positional_encoding[:, :L, :]
        tgt_mask = self.generate_causal_mask(L, device)

        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory_features,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.fc_out(output)
        return logits
