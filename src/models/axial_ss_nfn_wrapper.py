import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassConfusionMatrix


class AxialSSNFNWrapper(pl.LightningModule):
    """
    Multi-task wrapper for:
      - Neural Foraminal Narrowing (3-class)
      - Subarticular Stenosis    (3-class)
    """

    def __init__(self, base_model, lr, criterion_nfn, criterion_ss):
        super().__init__()

        self.save_hyperparameters(ignore=["base_model"])

        self.base_model = base_model
        self.lr = lr
        self.criterion_nfn = criterion_nfn
        self.criterion_ss = criterion_ss

        # number of features after FPN pooling concat
        # base_model.classifier = nn.Linear(feature_size*5, num_classes)
        # → 所以 in_features = base_model.classifier.in_features
        in_features = base_model.classifier.in_features

        # remove original classifier
        self.base_model.classifier = nn.Identity()

        # create multi-task heads
        self.head_nfn = nn.Linear(in_features, 3)
        self.head_ss  = nn.Linear(in_features, 3)

        # confusion matrix
        self.cm_nfn = MulticlassConfusionMatrix(num_classes=3)
        self.cm_ss = MulticlassConfusionMatrix(num_classes=3)

        self.val_nfn_preds, self.val_nfn_trues = [], []
        self.val_ss_preds, self.val_ss_trues = [], []

        print(">>> AxialSSNFNWrapper initialized (multi-task)")


    # ----------------------------------------
    def forward(self, x):
        """
        Run backbone + FPN + pooling → feature vec.
        Then apply 2 heads.
        """
        feat = self.base_model(x)  # this returns feature_vec because classifier=Identity()
        nfn_logits = self.head_nfn(feat)
        ss_logits  = self.head_ss(feat)
        return nfn_logits, ss_logits


    # ----------------------------------------
    def training_step(self, batch, batch_idx):
        images, targets = batch
        nfn_logits, ss_logits = self(images)

        loss = (
            self.criterion_nfn(nfn_logits, targets[:, 0]) +
            self.criterion_ss(ss_logits, targets[:, 1])
        )

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss


    # ----------------------------------------
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        nfn_logits, ss_logits = self(images)

        nfn_pred = nfn_logits.argmax(dim=1)
        ss_pred  = ss_logits.argmax(dim=1)

        self.val_nfn_preds.append(nfn_pred.cpu())
        self.val_nfn_trues.append(targets[:, 0].cpu())
        self.val_ss_preds.append(ss_pred.cpu())
        self.val_ss_trues.append(targets[:, 1].cpu())

        return {}


    # ----------------------------------------
    def validation_epoch_end(self, outputs):
        nfn_preds = torch.cat(self.val_nfn_preds)
        nfn_trues = torch.cat(self.val_nfn_trues)
        ss_preds = torch.cat(self.val_ss_preds)
        ss_trues = torch.cat(self.val_ss_trues)

        print("\n========== Multi-task Confusion Matrices ==========\n")

        print("[NFN Confusion Matrix]")
        print(self.cm_nfn(nfn_preds, nfn_trues).numpy())

        print("\n[SS Confusion Matrix]")
        print(self.cm_ss(ss_preds, ss_trues).numpy())

        self.val_nfn_preds.clear()
        self.val_nfn_trues.clear()
        self.val_ss_preds.clear()
        self.val_ss_trues.clear()

    # ----------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
