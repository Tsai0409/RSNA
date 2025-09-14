# models/mil.py

import torch
import torch.nn as nn

class SagittalMILModel(nn.Module):
    def __init__(self, backbone, num_classes, pooling="attention"):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling

        # 移掉 backbone 原本的 classifier head
        if hasattr(self.backbone, "fc"):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            in_features = backbone.num_features

        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: [N, S, C, H, W]
        N, S, C, H, W = x.shape
        x = x.view(N*S, C, H, W)       # 攤平成 batch
        feats = self.backbone(x)       # [N*S, F]
        feats = feats.view(N, S, -1)   # [N, S, F]

        if self.pooling == "mean":
            bag_feats = feats.mean(dim=1)
        elif self.pooling == "max":
            bag_feats, _ = feats.max(dim=1)
        elif self.pooling == "attention":
            attn = self.attention(feats)       # [N, S, 1]
            attn = torch.softmax(attn, dim=1) 
            bag_feats = torch.sum(attn * feats, dim=1)
        else:
            raise ValueError("Unknown pooling")

        return self.classifier(bag_feats)      # [N, num_classes]
