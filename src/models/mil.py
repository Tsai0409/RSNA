# models/mil.py

import torch
import torch.nn as nn

class SagittalMILModel(nn.Module):
    def __init__(self, backbone, num_classes, pooling="attention", input_size=224):
        """
        backbone: 2D CNN 模型 (e.g. ResNet50V2FPN, timm.create_model(...))
        num_classes: 分類數
        pooling: "mean", "max", 或 "attention"
        input_size: dummy input 的大小 (用來自動推斷 feature dim)
        """
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling

        in_features = None

        # === 1. 嘗試常見的屬性 ===
        if hasattr(self.backbone, "fc") and isinstance(self.backbone.fc, nn.Linear):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "classifier") and isinstance(self.backbone.classifier, nn.Linear):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "head") and isinstance(self.backbone.head, nn.Linear):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        # === 2. 如果沒有以上屬性，動態推斷 ===
        if in_features is None:
            with torch.no_grad():
                dummy = torch.randn(1, 3, input_size, input_size)
                out = self.backbone(dummy)
                if out.ndim == 4:  # e.g. [N, C, H, W] → 展平
                    out = torch.flatten(out, 1)
                in_features = out.shape[1]

        # === 3. Attention pooling (可選) ===
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        x: [N, S, C, H, W]
        """
        N, S, C, H, W = x.shape
        x = x.view(N * S, C, H, W)     # 攤平成 batch
        feats = self.backbone(x)       # [N*S, F]

        # 如果 backbone 輸出是 feature map → 展平
        if feats.ndim == 4:
            feats = torch.flatten(feats, 1)  # [N*S, F]

        feats = feats.view(N, S, -1)         # [N, S, F]

        # === 聚合 (bag-level feature) ===
        if self.pooling == "mean":
            bag_feats = feats.mean(dim=1)
        elif self.pooling == "max":
            bag_feats, _ = feats.max(dim=1)
        elif self.pooling == "attention":
            attn = self.attention(feats)        # [N, S, 1]
            attn = torch.softmax(attn, dim=1)   # normalize
            bag_feats = torch.sum(attn * feats, dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        # === 分類器 ===
        return self.classifier(bag_feats)       # [N, num_classes]
