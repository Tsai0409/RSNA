import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Conv2D + ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ResNet50V2FPN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, feature_size=256):
        super().__init__()
        # Backbone: ResNetV2-50 from timm
        self.backbone = timm.create_model(
            "resnetv2_50", pretrained=pretrained, features_only=True
        )

        # 取得各層 channel 數
        channels = self.backbone.feature_info.channels()  # [C2, C3, C4, C5]

        # FPN lateral conv layers
        self.lateral_c3 = nn.Conv2d(channels[1], feature_size, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(channels[2], feature_size, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(channels[3], feature_size, kernel_size=1)

        # FPN smoothing convs
        self.smooth_p3 = ConvBlock(feature_size, feature_size, 3)
        self.smooth_p4 = ConvBlock(feature_size, feature_size, 3)
        self.smooth_p5 = ConvBlock(feature_size, feature_size, 3)

        # Extra pyramid levels
        self.p6 = nn.Conv2d(channels[3], feature_size, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # Dropout + classifier head (類似原始 repo 的 Dense)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(feature_size * 5, num_classes)  # concat P3–P7

    def forward(self, x):
        # Backbone features: [C2, C3, C4, C5]
        # c2, c3, c4, c5 = self.backbone(x)
        features = self.backbone(x)   # timm 會回傳一個 list
        print([f.shape for f in features])  # ← 在這裡 debug 看看輸出有幾層

        # 你現在的程式碼假設只有 4 層
        c2, c3, c4, c5 = features[1:]  # 如果有 5 層，這樣寫比較保險

        # Lateral
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        # Smooth
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)

        # Extra layers
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))

        # Global pooling each pyramid level
        pooled_feats = []
        for p in [p3, p4, p5, p6, p7]:
            pooled = F.adaptive_avg_pool2d(p, (1, 1)).flatten(1)
            pooled_feats.append(pooled)

        concat = torch.cat(pooled_feats, dim=1)
        out = self.classifier(self.dropout(concat))
        return out
