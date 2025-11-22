# axial_ss_nfn_wrapper.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassConfusionMatrix
from src.lightning.lightning_modules.classification import MyLightningModule



class AxialSSNFNWrapper(MyLightningModule):
    def __init__(self, base_model, lr, criterion_nfn, criterion_ss):
        super().__init__(cfg)
        self.base_model = base_model
        self.lr = lr
        self.criterion_nfn = criterion_nfn
        self.criterion_ss = criterion_ss

        # 這個 flag 你在 rsna_axial_ss_nfn_ResNet50V2 有設 is_axial_ss_nfn = True
        # base 的 classification.py 很可能用這個判斷 multi-task 分支
        self.is_axial_ss_nfn = True

        self.save_hyperparameters(ignore=["base_model", "criterion_nfn", "criterion_ss"])

    # ---------------------------------------------------
    # Optimizer
    # ---------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    # ---------------------------------------------------
    # Train step: 把 6 個 logits 拆成 2 個 head
    # ---------------------------------------------------
    def training_step(self, batch, batch_idx):
        images = batch["image"]          # 依你 datamodule 實際 key 為準
        nfn_targets = batch["nfn_class"]
        ss_targets = batch["ss_class"]

        logits = self.base_model(images)     # (B, 6)
        nfn_logits = logits[:, :3]          # (B, 3)
        ss_logits = logits[:, 3:]           # (B, 3)

        loss_nfn = self.criterion_nfn(nfn_logits, nfn_targets)
        loss_ss = self.criterion_ss(ss_logits, ss_targets)

        loss = 0.5 * (loss_nfn + loss_ss)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_nfn", loss_nfn, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_loss_ss", loss_ss, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    # ---------------------------------------------------
    # Validation: 初始化 buffer + 收集 logits / targets
    # ---------------------------------------------------
    def on_validation_epoch_start(self):
        # 這四個就是 classification.py 期待的 multi-task buffer
        self.val_nfn_logits = []
        self.val_ss_logits = []
        self.val_nfn_targets = []
        self.val_ss_targets = []

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        nfn_targets = batch["nfn_class"]
        ss_targets = batch["ss_class"]

        logits = self.base_model(images)     # (B, 6)
        nfn_logits = logits[:, :3]
        ss_logits = logits[:, 3:]

        loss_nfn = self.criterion_nfn(nfn_logits, nfn_targets)
        loss_ss = self.criterion_ss(ss_logits, ss_targets)
        loss = 0.5 * (loss_nfn + loss_ss)

        # 收集給 validation_epoch_end 用
        self.val_nfn_logits.append(nfn_logits.detach().cpu())
        self.val_ss_logits.append(ss_logits.detach().cpu())
        self.val_nfn_targets.append(nfn_targets.detach().cpu())
        self.val_ss_targets.append(ss_targets.detach().cpu())

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss_nfn", loss_nfn, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_loss_ss", loss_ss, prog_bar=False, on_step=False, on_epoch=True)

        # 如果 base 的 validation_epoch_end 有用 outputs，
        # 你也可以 return 一個 dict
        return {
            "val_loss": loss.detach(),
        }


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

