# axial_ss_nfn_wrapper.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassConfusionMatrix
from src.lightning.lightning_modules.classification import MyLightningModule



class AxialSSNFNWrapper(MyLightningModule):
    def __init__(self, cfg, base_model, lr, criterion_nfn, criterion_ss):
        super().__init__(cfg)

        self.cfg = cfg
        self.base_model = base_model
        self.lr = lr
        self.criterion_nfn = criterion_nfn
        self.criterion_ss = criterion_ss

        self.is_axial_ss_nfn = True

        self.save_hyperparameters(ignore=["base_model", "criterion_nfn", "criterion_ss"])

    # -------------------------------------------------
    # 必須覆寫 forward()，否則 Lightning 用 base class forward
    # -------------------------------------------------
    def forward(self, x):
        return self.base_model(x)  # (B, 6)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        nfn_targets = batch["nfn_class"]
        ss_targets = batch["ss_class"]

        logits = self(images)
        nfn_logits = logits[:, :3]
        ss_logits  = logits[:, 3:]

        loss_nfn = self.criterion_nfn(nfn_logits, nfn_targets)
        loss_ss  = self.criterion_ss(ss_logits, ss_targets)

        loss = 0.5 * (loss_nfn + loss_ss)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_nfn_logits = []
        self.val_ss_logits = []
        self.val_nfn_targets = []
        self.val_ss_targets = []

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        nfn_targets = batch["nfn_class"]
        ss_targets = batch["ss_class"]

        logits = self(images)
        nfn_logits = logits[:, :3]
        ss_logits  = logits[:, 3:]

        # collect
        self.val_nfn_logits.append(nfn_logits.detach().cpu())
        self.val_ss_logits.append(ss_logits.detach().cpu())
        self.val_nfn_targets.append(nfn_targets.detach().cpu())
        self.val_ss_targets.append(ss_targets.detach().cpu())


# class AxialSSNFNWrapper(pl.LightningModule):
#     def __init__(self, base_model, lr, criterion_nfn, criterion_ss):
#         super().__init__()
#         self.save_hyperparameters(ignore=["base_model"])
        
#         self.base_model = base_model
#         self.lr = lr
#         self.criterion_nfn = criterion_nfn
#         self.criterion_ss  = criterion_ss

#         in_features = base_model.classifier.in_features

#         self.base_model.classifier = nn.Identity()

#         self.head_nfn = nn.Linear(in_features, 3)
#         self.head_ss  = nn.Linear(in_features, 3)

#         print(">>> AxialSSNFNWrapper initialized (multi-task)")

#     def forward(self, x):
#         feat = self.base_model(x)

#         nfn = self.head_nfn(feat)  # (B,3)
#         ss  = self.head_ss(feat)   # (B,3)

#         # ★ 最關鍵：回傳 6 logits，而不是 tuple
#         return torch.cat([nfn, ss], dim=1)  # (B,6)

#     def configure_optimizers(self):
#         return torch.optim.AdamW(self.parameters(), lr=self.lr)

