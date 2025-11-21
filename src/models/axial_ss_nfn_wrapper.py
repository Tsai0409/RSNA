# axial_ss_nfn_wrapper.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassConfusionMatrix


class AxialSSNFNWrapper(pl.LightningModule):
    """
    Multi-task LightningModule wrapper for:
      - Neural Foraminal Narrowing (3-class)
      - Subarticular Stenosis    (3-class)
    """

    def __init__(self, base_model, lr, criterion_nfn, criterion_ss):
        super().__init__()

        self.base_model = base_model
        self.lr = lr
        self.criterion_nfn = criterion_nfn
        self.criterion_ss = criterion_ss

        # confusion matrices
        self.cm_nfn = MulticlassConfusionMatrix(num_classes=3)
        self.cm_ss = MulticlassConfusionMatrix(num_classes=3)

        self.val_nfn_preds = []
        self.val_nfn_trues = []
        self.val_ss_preds = []
        self.val_ss_trues = []

        print(">>> AxialSSNFNWrapper initialized (multi-task)")

    # -------------------------------------------------
    def forward(self, x):
        return self.base_model(x)

    # -------------------------------------------------
    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)

        loss = (
            self.criterion_nfn(logits[:, :3], targets[:, 0]) +
            self.criterion_ss(logits[:, 3:], targets[:, 1])
        )

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # -------------------------------------------------
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)

        nfn_pred = torch.argmax(logits[:, :3], dim=1)
        ss_pred = torch.argmax(logits[:, 3:], dim=1)

        self.val_nfn_preds.append(nfn_pred.cpu())
        self.val_nfn_trues.append(targets[:, 0].cpu())
        self.val_ss_preds.append(ss_pred.cpu())
        self.val_ss_trues.append(targets[:, 1].cpu())

        return {}

    # -------------------------------------------------
    def validation_epoch_end(self, outputs):
        nfn_preds = torch.cat(self.val_nfn_preds)
        nfn_trues = torch.cat(self.val_nfn_trues)
        ss_preds = torch.cat(self.val_ss_preds)
        ss_trues = torch.cat(self.val_ss_trues)

        print("\n========== VALIDATION RESULTS ==========")

        print("\nNFN Confusion Matrix (3x3):")
        print(self.cm_nfn(nfn_preds, nfn_trues).cpu().numpy())

        print("\nSS Confusion Matrix (3x3):")
        print(self.cm_ss(ss_preds, ss_trues).cpu().numpy())

        # clear buffers
        self.val_nfn_preds.clear()
        self.val_nfn_trues.clear()
        self.val_ss_preds.clear()
        self.val_ss_trues.clear()

    # -------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
