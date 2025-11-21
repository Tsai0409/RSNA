#configs.py
from pathlib import Path
from pprint import pprint
import timm
from src.utils.metric_learning_loss import *
from src.utils.metrics import *
from src.utils.loss import *
from src.global_objectives import AUCPRLoss
import os
import torch.nn as nn
import pandas as pd
import numpy as np
from pdb import set_trace as st
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import pytorch_lightning as pl
from types import MethodType
import warnings
warnings.simplefilter('ignore')

from src.models.resnet3d_csn import *
from src.models.uniformerv2 import *
from src.models.rsna import *
from src.models.rsna_2023_1st_models import *
from src.models.mil_3models import *
from src.models.layers import AdaptiveConcatPool2d, Flatten
from src.models.ch_mdl_dolg_efficientnet import ChMdlDolgEfficientnet, ArcFaceLossAdaptiveMargin
from src.models.rsna_multi_image import MultiLevelModel2
from src.models.backbones import *
from src.models.group_norm import convert_groupnorm
from src.models.batch_renorm import convert_batchrenorm
from src.models.multi_instance import MultiInstanceModel, MetaMIL, AttentionMILModel, MultiInstanceModelWithWataruAttention
from src.models.resnet import resnet18, resnet34, resnet101, resnet152
from src.models.nextvit import NextVitNet
from src.models.model_4channels import get_attention, get_resnet34, get_attention_inceptionv3
from src.models.vae import VAE, ResNet_VAE
from src.models.model_with_arcface import ArcMarginProduct, AddMarginProduct, ArcMarginProductSubcenter, ArcMarginProductOutCosine, ArcMarginProductSubcenterOutCosine, PudaeArcNet, WithArcface, WhalePrev1stModel, Guie2
from src.models.with_meta_models import WithMetaModel
from src.models.resnet50v2_fpn import ResNet50V2FPN  # multiscale model
from src.models.mil import SagittalMILModel  # multiscale model
from src.models.axial_ss_nfn_wrapper import AxialSSNFNWrapper

from src.utils.augmentations.strong_aug import *
from src.utils.augmentations.augmentation import *
from src.utils.augmentations.policy_transform import policy_transform
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, average_precision_score, accuracy_score
import numpy as np
from scipy.special import softmax

def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
WORKING_DIR="/kaggle/working/duplicate"
################

class Baseline:
    def __init__(self):
        self.memo = ''
        # self.gpu = 'small'
        self.gpu = 'v100'
        self.compe = 'rsna'
        self.batch_size = 16
        self.grad_accumulations = 1
        self.lr = 0.0001
        self.epochs = 20
        self.resume = False
        self.seed = 2023
        self.tta = 1
        self.model_name = 'convnext_small.fb_in22k_ft_in1k_384'
        # self.model_name = 'resnet50'
        self.num_classes = 1
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # self.criterion = torch.nn.BCELoss()
        # self.transform = medical_v1
        self.transform = kuma_aug
        self.image_size = 384
        self.label_features = ['target']
        self.metric = roc_auc_score # AUC().torch # MultiAP().torch # MultiAUC().torch
        self.fp16 = True
        # self.optimizer = 'adam'
        self.optimizer = 'adamw'
        self.scheduler = 'CosineAnnealingWarmRestarts'
        self.eta_min = 5e-7
        self.train_by_all_data = False
        self.early_stop_patience = 1000
        self.inference = False
        self.predict_valid = True
        self.predict_test = False
        self.logit_to = None
        self.pretrained_path = None
        self.sync_batchnorm = True
        # self.sync_batchnorm = False
        self.warmup_epochs = -1
        self.finetune_transform = base_aug_v1
        self.mixup = False
        self.arcface = False
        self.box_crop = None
        self.predicted_mask_crop = None
        self.pad_square = False
        self.resume_epoch = 0
        self.t_max=30
        self.save_top_k = 1
        self.meta_cols = []
        self.output_features = False
        self.force_use_model_path_config_when_inf = None
        self.reset_classifier_when_inf = False
        self.upsample = None
        self.in_chans = 3
        self.add_imsizes_when_inference = [(0, 0)]
        self.inf_fp16 = False
        self.distill = False
        self.reload_dataloaders_every_n_epochs = 0
        self.tranform_dataset_version = None
        self.no_trained_model_when_inf = False
        self.normalize_horiz_orientation = False
        self.upsample_batch_pos_n = None
        self.cut_200 = False
        self.affine_for_gbr = False
        self.half_dark = False
        self.crop_by_left_right_line_text = False
        self.use_wandb = True
        self.use_last_ckpt_when_inference = True
        self.inference_only = False
        self.valid_df = None
        self.valid_df_path = None
        self.ema = False
        self.awp = False
        self.save_every_epoch_val_preds = False

class rsna_v1(Baseline):
    def __init__(self):
        super().__init__()
        self.compe = 'rsna_2024'
        self.predict_valid = True
        self.predict_test = False
        self.predict_train = False
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.transform = medical_v3  # 定義在：src/utils/augmentations/augmentation.py
        self.batch_size = 8
        self.lr = 1e-5
        self.grad_accumulations = 2
        self.p_rand_order_v1 = 0

class rsna_sagittal_level_cl_spinal_v1(rsna_v1):  # inf_sagittal_slice_1st.sh
    def __init__(self, fold=0):
        super().__init__()
        self.fold = fold  # 如果你要在這裡使用 fold，我加
        # self.train_df_path = 'input/train_for_sagittal_level_cl_v1_for_train_spinal_only.csv'
        # self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_slice_estimation_5/train_for_sagittal_level_cl_v1_for_train_spinal_only.csv'  # 含 fold 資訊
        self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_slice_estimation_holdout_5/train_for_sagittal_level_cl_v1_for_train_spinal_only.csv'  # 含 fold 資訊
        print("I'm reading from path:", self.train_df_path)
        self.train_df = pd.read_csv(self.train_df_path)
        self.label_features = ['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal']
        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.image_size = 256
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        self.metric = MultiAUC(label_features=self.label_features).torch
        self.memo = ''
        self.batch_size = 16
        self.grad_accumulations = 1
        self.crop_by_xy = False
        self.rsna_2024_multi_image = False
        self.rsna_random_sample = False
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 15
        self.box_crop = None
        # self.test_df = pd.read_csv('input/sagittal_df.csv')
        # self.test_df = pd.read_csv(f'{WORKING_DIR}/csv_train/dcm_to_png_3/sagittal_df.csv')
        self.test_df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout_test.csv')  # -> 只有在 predict.py 中被使用到
        self.train_2nd_df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout.csv')
        self.predict_test = True
        self.predict_train = True

class rsna_sagittal_level_cl_nfn_v1(rsna_v1):  # inf_sagittal_slice_1st.sh
    def __init__(self, fold=0):
        super().__init__()
        self.fold = fold  # 我加
        # self.train_df_path = 'input/train_for_sagittal_level_cl_v1_for_train_nfn_only.csv'
        # self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_slice_estimation_5/train_for_sagittal_level_cl_v1_for_train_nfn_only.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_slice_estimation_holdout_5/train_for_sagittal_level_cl_v1_for_train_nfn_only.csv'
        print("I'm reading from path:", self.train_df_path)
        self.train_df = pd.read_csv(self.train_df_path)
        self.label_features = ['l1_right_neural', 'l2_right_neural', 'l3_right_neural', 'l4_right_neural', 'l5_right_neural', 'l1_left_neural', 'l2_left_neural', 'l3_left_neural', 'l4_left_neural', 'l5_left_neural']
        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.image_size = 256
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        self.metric = MultiAUC(label_features=self.label_features).torch
        self.memo = ''
        self.batch_size = 16
        self.grad_accumulations = 1
        self.crop_by_xy = False
        self.rsna_2024_multi_image = False
        self.rsna_random_sample = False
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 15
        self.box_crop = None
        # self.test_df = pd.read_csv('input/sagittal_df.csv')
        # self.test_df = pd.read_csv(f'{WORKING_DIR}/csv_train/dcm_to_png_3/sagittal_df.csv')
        self.test_df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout_test.csv')
        self.train_2nd_df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout.csv')
        self.predict_test = True
        self.predict_train = True

class rsna_sagittal_cl(rsna_v1):  # inf_sagittal_slice_2nd.sh
    def __init__(self, fold=0):
        super().__init__()
        self.fold = fold  # 我加
        # self.train_df_path = f'input/train_for_sagittal_level_cl_v1_for_train_spinal_nfn_fold{fold}.csv'
        # self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_slice_estimation_holdout_5/train_for_sagittal_level_cl_v1_for_train_spinal_nfn_fold{fold}.csv'
        # self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_slice_estimation_holdout_5/train_for_sagittal_level_cl_v1_for_train_spinal_nfn_fold1.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_slice_estimation_holdout_5/train_for_sagittal_level_cl_v1_for_train_spinal_nfn_fold1_no_axial.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.label_features = ['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal', 'l1_right_neural', 'l2_right_neural', 'l3_right_neural', 'l4_right_neural', 'l5_right_neural', 'l1_left_neural', 'l2_left_neural', 'l3_left_neural', 'l4_left_neural', 'l5_left_neural']
        self.label_features = ['pred_'+c for c in self.label_features]
        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.image_size = 256
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        self.metric = None
        self.memo = ''
        self.batch_size = 16
        self.grad_accumulations = 1
        self.crop_by_xy = False
        self.rsna_2024_multi_image = False
        self.rsna_random_sample = False
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 15
        self.box_crop = None
        # self.predict_test = False
        self.predict_test = True
        self.test_df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout_test.csv')
        self.predict_train = True
        self.train_2nd_df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout.csv')



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# classification for axial (stage 2)
class rsna_axial_ss_nfn_crop_base(rsna_v1):
    def __init__(self, fold=0):
        super().__init__()
        cols = []
        label_features = [
            'neural_foraminal_narrowing',
            'subarticular_stenosis',
        ]
        for col in label_features:
            cols.append(f'{col}_normal')
            cols.append(f'{col}_moderate')
            cols.append(f'{col}_severe')

        self.fold = fold  # 我加
        self.label_features = cols
        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.image_size = 384
        self.drop_rate = 0.0
        self.drop_path_rate = 0.0
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes, drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)  # 用 convnext_small.in12k_ft_in1k_384 訓練的
        self.metric = None
        self.memo = ''
        self.batch_size = 8
        self.grad_accumulations = 2
        self.crop_by_xy = False
        self.rsna_2024_multi_image = False
        self.rsna_random_sample = False
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 7
        self.transform = medical_v4
        self.box_crop = True
        self.box_crop_x_ratio = 0
        self.box_crop_y_ratio = 6

class rsna_axial_ss_nfn_x2_y2_center_pad0(rsna_axial_ss_nfn_crop_base):
    def __init__(self):
        super().__init__()
        image_width_ratio = 2  # 以 bounding box 中間，取影像寬度的 1/2
        self.box_crop_y_ratio = 2  # 以「偵測框中心」為基準，上下各取 1/2 範圍
        center_pad_ratio = 0  # 調整裁切區域是否左右偏移

        # self.train_df_path = 'input/axial_classification.csv'
        # self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_7/axial_classification.csv'
        self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_holdout_7/axial_classification_holdout.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['x_min'] = (self.train_df.x_max + self.train_df.x_min)/2
        del self.train_df['x_max']
        self.train_df['left_right'] = 'left'
        cols = [
            'left_neural_foraminal_narrowing_normal',
            'left_neural_foraminal_narrowing_moderate',
            'left_neural_foraminal_narrowing_severe',
            'left_subarticular_stenosis_normal',
            'left_subarticular_stenosis_moderate',
            'left_subarticular_stenosis_severe',
        ]
        for c in cols:
            self.train_df[c.replace('left_', '')] = self.train_df[c].values  # 創建新的 col，將原來 left 開頭的 ss, nfn 填到這邊
        self.train_df['x_max'] = self.train_df['x_min'] + self.train_df['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            self.train_df['x_min'] = self.train_df['x_min'] - self.train_df['image_width']/center_pad_ratio

        train_df_right = pd.read_csv(self.train_df_path)
        train_df_right['x_max'] = (train_df_right.x_max + train_df_right.x_min)/2
        del train_df_right['x_min']
        train_df_right['left_right'] = 'right'
        cols = [
            'right_neural_foraminal_narrowing_normal',
            'right_neural_foraminal_narrowing_moderate',
            'right_neural_foraminal_narrowing_severe',
            'right_subarticular_stenosis_normal',           
            'right_subarticular_stenosis_moderate',
            'right_subarticular_stenosis_severe',
        ]
        for c in cols:
            train_df_right[c.replace('right_', '')] = train_df_right[c].values  # 創建新的 col，將原來 right 開頭的 ss, nfn 填到這邊
        train_df_right['x_min'] = train_df_right['x_max'] - train_df_right['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            train_df_right['x_max'] = train_df_right['x_max'] + train_df_right['image_width']/center_pad_ratio

        self.train_df = pd.concat([self.train_df, train_df_right])  # 將左右合併起來 -> 最後在

class rsna_axial_ss_nfn_x2_y6_center_pad0(rsna_axial_ss_nfn_crop_base):
    def __init__(self):
        super().__init__()
        image_width_ratio = 2
        self.box_crop_y_ratio = 6
        center_pad_ratio = 0

        # self.train_df_path = 'input/axial_classification.csv'
        # self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_7/axial_classification.csv'
        self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_holdout_7/axial_classification_holdout.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['x_min'] = (self.train_df.x_max + self.train_df.x_min)/2
        del self.train_df['x_max']
        self.train_df['left_right'] = 'left'
        cols = [
            'left_neural_foraminal_narrowing_normal',
            'left_neural_foraminal_narrowing_moderate',
            'left_neural_foraminal_narrowing_severe',
            'left_subarticular_stenosis_normal',
            'left_subarticular_stenosis_moderate',
            'left_subarticular_stenosis_severe',
        ]
        for c in cols:
            self.train_df[c.replace('left_', '')] = self.train_df[c].values
        self.train_df['x_max'] = self.train_df['x_min'] + self.train_df['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            self.train_df['x_min'] = self.train_df['x_min'] - self.train_df['image_width']/center_pad_ratio

        train_df_right = pd.read_csv(self.train_df_path)
        train_df_right['x_max'] = (train_df_right.x_max + train_df_right.x_min)/2
        del train_df_right['x_min']
        train_df_right['left_right'] = 'right'
        cols = [
            'right_neural_foraminal_narrowing_normal',           
            'right_neural_foraminal_narrowing_moderate',
            'right_neural_foraminal_narrowing_severe',
            'right_subarticular_stenosis_normal',           
            'right_subarticular_stenosis_moderate',
            'right_subarticular_stenosis_severe',
        ]
        for c in cols:
            train_df_right[c.replace('right_', '')] = train_df_right[c].values
        train_df_right['x_min'] = train_df_right['x_max'] - train_df_right['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            train_df_right['x_max'] = train_df_right['x_max'] + train_df_right['image_width']/center_pad_ratio

        self.train_df = pd.concat([self.train_df, train_df_right])

class rsna_axial_ss_nfn_x2_y8_center_pad10(rsna_axial_ss_nfn_crop_base):
    def __init__(self):
        super().__init__()
        image_width_ratio = 2
        self.box_crop_y_ratio = 8
        center_pad_ratio = 10

        # self.train_df_path = 'input/axial_classification.csv'
        # self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_7/axial_classification.csv'
        self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_holdout_7/axial_classification_holdout.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['x_min'] = (self.train_df.x_max + self.train_df.x_min)/2
        del self.train_df['x_max']
        self.train_df['left_right'] = 'left'
        cols = [
            'left_neural_foraminal_narrowing_normal',           
            'left_neural_foraminal_narrowing_moderate',
            'left_neural_foraminal_narrowing_severe',
            'left_subarticular_stenosis_normal',           
            'left_subarticular_stenosis_moderate',
            'left_subarticular_stenosis_severe',
        ]
        for c in cols:
            self.train_df[c.replace('left_', '')] = self.train_df[c].values
        self.train_df['x_max'] = self.train_df['x_min'] + self.train_df['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            self.train_df['x_min'] = self.train_df['x_min'] - self.train_df['image_width']/center_pad_ratio


        train_df_right = pd.read_csv(self.train_df_path)
        train_df_right['x_max'] = (train_df_right.x_max + train_df_right.x_min)/2
        del train_df_right['x_min']
        train_df_right['left_right'] = 'right'
        cols = [
            'right_neural_foraminal_narrowing_normal',           
            'right_neural_foraminal_narrowing_moderate',
            'right_neural_foraminal_narrowing_severe',
            'right_subarticular_stenosis_normal',           
            'right_subarticular_stenosis_moderate',
            'right_subarticular_stenosis_severe',
        ]
        for c in cols:
            train_df_right[c.replace('right_', '')] = train_df_right[c].values
        train_df_right['x_min'] = train_df_right['x_max'] - train_df_right['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            train_df_right['x_max'] = train_df_right['x_max'] + train_df_right['image_width']/center_pad_ratio

        self.train_df = pd.concat([self.train_df, train_df_right])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# reduce_noise axial (stage 2)
# train(by clean data) vaild(by original data)
class rsna_axial_ss_nfn_x2_y2_center_pad0_with_valid(rsna_axial_ss_nfn_crop_base):
    def __init__(self, fold=0):
        super().__init__()
        image_width_ratio = 2
        self.box_crop_y_ratio = 2
        center_pad_ratio = 0
        self.fold = fold

        # 共同參數
        # self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_7/axial_classification.csv'
        self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_holdout_7/axial_classification_holdout.csv'

        # ------------------------
        # 建立原始 valid_df（不移除 noisy）
        # ------------------------
        def build_df(left_right='left'):
            df = pd.read_csv(self.train_df_path)
            df['level'] = df.pred_level.map({
                1: 'l1_l2',
                2: 'l2_l3',
                3: 'l3_l4',
                4: 'l4_l5',
                5: 'l5_s1',
            })
            df['study_level'] = df.study_id.astype(str) + '_' + df.level.str.replace('/', '_').str.lower()
            df['left_right'] = left_right

            if left_right == 'left':
                df['x_min'] = (df.x_max + df.x_min) / 2
                del df['x_max']
                cols = [
                    'left_neural_foraminal_narrowing_normal',
                    'left_neural_foraminal_narrowing_moderate',
                    'left_neural_foraminal_narrowing_severe',
                    'left_subarticular_stenosis_normal',
                    'left_subarticular_stenosis_moderate',
                    'left_subarticular_stenosis_severe',
                ]
                for c in cols:
                    df[c.replace('left_', '')] = df[c].values
                df['x_max'] = df['x_min'] + df['image_width'] / image_width_ratio
                if center_pad_ratio != 0:
                    df['x_min'] -= df['image_width'] / center_pad_ratio
            else:
                df['x_max'] = (df.x_max + df.x_min) / 2
                del df['x_min']
                cols = [
                    'right_neural_foraminal_narrowing_normal',           
                    'right_neural_foraminal_narrowing_moderate',
                    'right_neural_foraminal_narrowing_severe',
                    'right_subarticular_stenosis_normal',           
                    'right_subarticular_stenosis_moderate',
                    'right_subarticular_stenosis_severe',
                ]
                for c in cols:
                    df[c.replace('right_', '')] = df[c].values
                df['x_min'] = df['x_max'] - df['image_width'] / image_width_ratio
                if center_pad_ratio != 0:
                    df['x_max'] += df['image_width'] / center_pad_ratio
            return df

        valid_left = build_df('left')
        valid_right = build_df('right')
        self.valid_df = pd.concat([valid_left, valid_right], ignore_index=True)

        # ------------------------
        # 建立 train_df（有去除 noise）
        # ------------------------
        train_df = self.valid_df.copy()

        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th09.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th09_holdout.csv')

        noise_df_left = noise_df[
            (noise_df.target == 'left_neural_foraminal_narrowing') |
            (noise_df.target == 'left_subarticular_stenosis')
        ]
        noise_df_right = noise_df[
            (noise_df.target == 'right_neural_foraminal_narrowing') |
            (noise_df.target == 'right_subarticular_stenosis')
        ]
        noise_study_levels = set(noise_df_left.study_level) | set(noise_df_right.study_level)

        self.train_df = train_df[~train_df.study_level.isin(noise_study_levels)].reset_index(drop=True)

class rsna_axial_ss_nfn_x2_y6_center_pad0_with_valid(rsna_axial_ss_nfn_crop_base):
    def __init__(self, fold=0):
        super().__init__()
        image_width_ratio = 2
        self.box_crop_y_ratio = 6
        center_pad_ratio = 0
        self.fold = fold
        # self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_7/axial_classification.csv'
        self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_holdout_7/axial_classification_holdout.csv'

        def process_df(df, side):
            df['level'] = df.pred_level.map({
                1: 'l1_l2',
                2: 'l2_l3',
                3: 'l3_l4',
                4: 'l4_l5',
                5: 'l5_s1',
            })
            df['study_level'] = df.study_id.astype(str) + '_' + df.level.str.replace('/', '_').str.lower()
            df['left_right'] = side

            if side == 'left':
                df['x_min'] = (df.x_max + df.x_min) / 2
                del df['x_max']
                cols = [
                    'left_neural_foraminal_narrowing_normal',
                    'left_neural_foraminal_narrowing_moderate',
                    'left_neural_foraminal_narrowing_severe',
                    'left_subarticular_stenosis_normal',
                    'left_subarticular_stenosis_moderate',
                    'left_subarticular_stenosis_severe',
                ]
                for c in cols:
                    df[c.replace('left_', '')] = df[c].values
                df['x_max'] = df['x_min'] + df['image_width'] / image_width_ratio
                if center_pad_ratio != 0:
                    df['x_min'] -= df['image_width'] / center_pad_ratio
            else:
                df['x_max'] = (df.x_max + df.x_min) / 2
                del df['x_min']
                cols = [
                    'right_neural_foraminal_narrowing_normal',
                    'right_neural_foraminal_narrowing_moderate',
                    'right_neural_foraminal_narrowing_severe',
                    'right_subarticular_stenosis_normal',
                    'right_subarticular_stenosis_moderate',
                    'right_subarticular_stenosis_severe',
                ]
                for c in cols:
                    df[c.replace('right_', '')] = df[c].values
                df['x_min'] = df['x_max'] - df['image_width'] / image_width_ratio
                if center_pad_ratio != 0:
                    df['x_max'] += df['image_width'] / center_pad_ratio
            return df

        # ------ valid_df（不去除 noisy） ------
        valid_left = pd.read_csv(self.train_df_path)
        valid_left = process_df(valid_left, side='left')

        valid_right = pd.read_csv(self.train_df_path)
        valid_right = process_df(valid_right, side='right')

        self.valid_df = pd.concat([valid_left, valid_right], ignore_index=True)

        # ------ train_df（去除 noisy） ------
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th09.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th09_holdout.csv')

        noise_df_left = noise_df[
            (noise_df.target == 'left_neural_foraminal_narrowing') |
            (noise_df.target == 'left_subarticular_stenosis')
        ]
        noise_df_right = noise_df[
            (noise_df.target == 'right_neural_foraminal_narrowing') |
            (noise_df.target == 'right_subarticular_stenosis')
        ]
        noise_study_levels = set(noise_df_left.study_level) | set(noise_df_right.study_level)

        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noise_study_levels)].reset_index(drop=True)

class rsna_axial_ss_nfn_x2_y8_center_pad10_with_valid(rsna_axial_ss_nfn_crop_base):
    def __init__(self, fold=0):
        super().__init__()
        image_width_ratio = 2
        self.box_crop_y_ratio = 8
        center_pad_ratio = 10
        self.fold = fold
        # self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_7/axial_classification.csv'
        self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_holdout_7/axial_classification_holdout.csv'

        # ----- 共用欄位處理 -----
        def process_df(df, side):
            df['level'] = df.pred_level.map({
                1: 'l1_l2',
                2: 'l2_l3',
                3: 'l3_l4',
                4: 'l4_l5',
                5: 'l5_s1',
            })
            df['study_level'] = df.study_id.astype(str) + '_' + df.level.str.replace('/', '_').str.lower()
            df['left_right'] = side

            if side == 'left':
                df['x_min'] = (df.x_max + df.x_min) / 2
                del df['x_max']
                cols = [
                    'left_neural_foraminal_narrowing_normal',
                    'left_neural_foraminal_narrowing_moderate',
                    'left_neural_foraminal_narrowing_severe',
                    'left_subarticular_stenosis_normal',
                    'left_subarticular_stenosis_moderate',
                    'left_subarticular_stenosis_severe',
                ]
                for c in cols:
                    df[c.replace('left_', '')] = df[c].values
                df['x_max'] = df['x_min'] + df['image_width'] / image_width_ratio
                if center_pad_ratio != 0:
                    df['x_min'] -= df['image_width'] / center_pad_ratio
            else:
                df['x_max'] = (df.x_max + df.x_min) / 2
                del df['x_min']
                cols = [
                    'right_neural_foraminal_narrowing_normal',
                    'right_neural_foraminal_narrowing_moderate',
                    'right_neural_foraminal_narrowing_severe',
                    'right_subarticular_stenosis_normal',
                    'right_subarticular_stenosis_moderate',
                    'right_subarticular_stenosis_severe',
                ]
                for c in cols:
                    df[c.replace('right_', '')] = df[c].values
                df['x_min'] = df['x_max'] - df['image_width'] / image_width_ratio
                if center_pad_ratio != 0:
                    df['x_max'] += df['image_width'] / center_pad_ratio
            return df

        # ----- 建立 valid_df（保留所有資料） -----
        valid_left = pd.read_csv(self.train_df_path)
        valid_left = process_df(valid_left, side='left')

        valid_right = pd.read_csv(self.train_df_path)
        valid_right = process_df(valid_right, side='right')

        self.valid_df = pd.concat([valid_left, valid_right], ignore_index=True)

        # ----- 建立 train_df（去除 noisy 資料） -----
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th09.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th09_holdout.csv')

        noise_df_left = noise_df[
            (noise_df.target == 'left_neural_foraminal_narrowing') |
            (noise_df.target == 'left_subarticular_stenosis')
        ]
        noise_df_right = noise_df[
            (noise_df.target == 'right_neural_foraminal_narrowing') |
            (noise_df.target == 'right_subarticular_stenosis')
        ]
        noise_study_levels = set(noise_df_left.study_level) | set(noise_df_right.study_level)

        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noise_study_levels)].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# reduce_noise axial ResNet50V2(stage 2)
# train(by clean data) vaild(by original data)

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes]
        targets: [batch_size]  (不是 one-hot, 而是 int label, e.g. 0,1,2)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = softmax probability of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes]   # raw outputs
        targets: [batch_size, num_classes]  # one-hot or multi-label (0/1 per class)
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):

        # ---------------------------
        # ⭐ 這兩行是關鍵：處理 one-hot or multi-hot
        # ---------------------------
        if targets.ndim > 1:
            targets = torch.argmax(targets, dim=1)

        targets = targets.long()     # 保證 cross entropy 能用
        # ---------------------------

        # print(">>> alpha device:", self.alpha.device,
        #     "| targets device:", targets.device,
        #     "| targets dtype:", targets.dtype,
        #     "| targets shape:", targets.shape)

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            alpha_factor = self.alpha[targets]
            focal_loss = alpha_factor * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()




class Baseline_ResNet50V2:
    def __init__(self):
        self.memo = ''
        # self.gpu = 'small'
        self.gpu = 'v100'
        self.compe = 'rsna'
        self.batch_size = 16
        self.grad_accumulations = 1
        self.lr = 0.0001
        self.epochs = 20
        self.resume = False
        self.seed = 2023
        self.tta = 1
        self.model_name = 'convnext_small.fb_in22k_ft_in1k_384'
        # self.model_name = 'resnet50'
        self.num_classes = 1
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        # self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        # self.criterion = torch.nn.BCELoss()
        # self.transform = medical_v1
        self.transform = kuma_aug
        self.image_size = 384
        self.label_features = ['target']
        self.metric = roc_auc_score # AUC().torch # MultiAP().torch # MultiAUC().torch
        self.fp16 = True
        # self.optimizer = 'adam'
        self.optimizer = 'adamw'
        self.scheduler = 'CosineAnnealingWarmRestarts'
        self.eta_min = 5e-7
        self.train_by_all_data = False
        self.early_stop_patience = 1000
        self.inference = False
        self.predict_valid = True
        self.predict_test = False
        self.logit_to = None
        self.pretrained_path = None
        self.sync_batchnorm = True
        # self.sync_batchnorm = False
        self.warmup_epochs = -1
        self.finetune_transform = base_aug_v1
        self.mixup = False
        self.arcface = False
        self.box_crop = None
        self.predicted_mask_crop = None
        self.pad_square = False
        self.resume_epoch = 0
        self.t_max=30
        self.save_top_k = 1
        self.meta_cols = []
        self.output_features = False
        self.force_use_model_path_config_when_inf = None
        self.reset_classifier_when_inf = False
        self.upsample = None
        self.in_chans = 3
        self.add_imsizes_when_inference = [(0, 0)]
        self.inf_fp16 = False
        self.distill = False
        self.reload_dataloaders_every_n_epochs = 0
        self.tranform_dataset_version = None
        self.no_trained_model_when_inf = False
        self.normalize_horiz_orientation = False
        self.upsample_batch_pos_n = None
        self.cut_200 = False
        self.affine_for_gbr = False
        self.half_dark = False
        self.crop_by_left_right_line_text = False
        self.use_wandb = True
        self.use_last_ckpt_when_inference = True
        self.inference_only = False
        self.valid_df = None
        self.valid_df_path = None
        self.ema = False
        self.awp = False
        self.save_every_epoch_val_preds = False

class rsna_v1_ResNet50V2(Baseline_ResNet50V2):
    def __init__(self):
        super().__init__()
        self.compe = 'rsna_2024'
        self.predict_valid = True
        self.predict_test = False
        self.predict_train = False
        # self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.transform = medical_v3  # 定義在：src/utils/augmentations/augmentation.py
        self.batch_size = 8
        self.lr = 1e-5
        self.grad_accumulations = 2
        self.p_rand_order_v1 = 0

class rsna_axial_ss_nfn_ResNet50V2(rsna_v1_ResNet50V2):
    """
    Multi-task version of axial spinal model:
      - Neural Foraminal Narrowing
      - Subarticular Stenosis
    Each: Normal, Moderate, Severe → 3 classes
    Total outputs: 6 logits
    """
    def __init__(self, fold=0):
        super().__init__()
        self.is_axial_ss_nfn = True   # <--- 加這行

        # --------------------------
        # Dataset label config
        # --------------------------
        self.fold = fold
        self.label_features = [
            'neural_foraminal_narrowing_normal',
            'neural_foraminal_narrowing_moderate',
            'neural_foraminal_narrowing_severe',
            'subarticular_stenosis_normal',
            'subarticular_stenosis_moderate',
            'subarticular_stenosis_severe',
        ]
        self.num_classes = 6
        self.task = "multiclass"

        # --------------------------
        # Hyperparameters
        # --------------------------
        self.lr = 1e-4
        self.batch_size = 8
        self.image_size = 224
        self.epochs = 20
        self.transform = medical_v4

        # --------------------------
        # Multi-task loss
        # --------------------------
        self.alpha_nfn = torch.tensor([0.04, 0.33, 0.63]).float()
        self.alpha_ss = torch.tensor([0.05, 0.30, 0.65]).float()

        self.criterion_nfn = MultiClassFocalLoss(gamma=2.0, alpha=self.alpha_nfn)
        self.criterion_ss = MultiClassFocalLoss(gamma=2.0, alpha=self.alpha_ss)

        # --------------------------
        # Backbone Model
        # --------------------------
        base = ResNet50V2FPN(num_classes=self.num_classes, pretrained=True)

        # --------------------------
        # Wrap backbone into multi-task LightningModule
        # --------------------------
        self.model = AxialSSNFNWrapper(
            base_model=base,
            lr=self.lr,
            criterion_nfn=self.criterion_nfn,
            criterion_ss=self.criterion_ss,
        )

        # --------------------------
        # Other configs
        # --------------------------
        self.box_crop = True
        self.box_crop_x_ratio = 1
        self.box_crop_y_ratio = 2
        self.center_pad_ratio = 0
        self.image_width_ratio = 1

        self.grad_accumulations = 2
        self.memo = ""
        self.metric = None

        # --------------------------
        # Build dataframe
        # --------------------------
        self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_holdout_7/axial_classification_holdout.csv'
        self._build_dataframes()

    # ------------------------------------------------------
    # REMOVE calculate_loss: wrapper handles training loss.
    # If present, it may override wrapper logic accidentally.
    # ------------------------------------------------------

    def _build_dataframes(self):
        # ----- 共用欄位處理 -----
        def process_df(df, side):
            df['level'] = df.pred_level.map({
                1: 'l1_l2',
                2: 'l2_l3',
                3: 'l3_l4',
                4: 'l4_l5',
                5: 'l5_s1',
            })
            df['study_level'] = df.study_id.astype(str) + '_' + df.level.str.replace('/', '_').str.lower()
            df['left_right'] = side

            if side == 'left':
                df['x_min'] = (df.x_max + df.x_min) / 2
                del df['x_max']
                for c in [
                    'left_neural_foraminal_narrowing_normal',
                    'left_neural_foraminal_narrowing_moderate',
                    'left_neural_foraminal_narrowing_severe',
                    'left_subarticular_stenosis_normal',
                    'left_subarticular_stenosis_moderate',
                    'left_subarticular_stenosis_severe',
                ]:
                    df[c.replace('left_', '')] = df[c].values
                df['x_max'] = df['x_min'] + df['image_width'] / self.image_width_ratio
                if self.center_pad_ratio != 0:
                    df['x_min'] -= df['image_width'] / self.center_pad_ratio
            else:
                df['x_max'] = (df.x_max + df.x_min) / 2
                del df['x_min']
                for c in [
                    'right_neural_foraminal_narrowing_normal',
                    'right_neural_foraminal_narrowing_moderate',
                    'right_neural_foraminal_narrowing_severe',
                    'right_subarticular_stenosis_normal',
                    'right_subarticular_stenosis_moderate',
                    'right_subarticular_stenosis_severe',
                ]:
                    df[c.replace('right_', '')] = df[c].values
                df['x_min'] = df['x_max'] - df['image_width'] / self.image_width_ratio
                if self.center_pad_ratio != 0:
                    df['x_max'] += df['image_width'] / self.center_pad_ratio
            return df

        # ----- 建立 valid_df（保留所有資料） -----
        valid_left = pd.read_csv(self.train_df_path)
        valid_left = process_df(valid_left, side='left')

        valid_right = pd.read_csv(self.train_df_path)
        valid_right = process_df(valid_right, side='right')

        self.valid_df = pd.concat([valid_left, valid_right], ignore_index=True)

        # ----- 建立 train_df（去除 noisy 資料） -----
        train_df = self.valid_df.copy()

        noise_df = pd.read_csv(
            f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th09_holdout.csv'
        )
        noise_df_left = noise_df[
            (noise_df.target == 'left_neural_foraminal_narrowing') |
            (noise_df.target == 'left_subarticular_stenosis')
        ]
        noise_df_right = noise_df[
            (noise_df.target == 'right_neural_foraminal_narrowing') |
            (noise_df.target == 'right_subarticular_stenosis')
        ]
        noise_study_levels = set(noise_df_left.study_level) | set(noise_df_right.study_level)

        self.train_df = train_df[~train_df.study_level.isin(noise_study_levels)].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# classification for axial (stage 1)
class rsna_axial_spinal_crop_base(rsna_v1):
    def __init__(self, fold=0):
        super().__init__()
        self.fold = fold  # 我加
        # self.train_df_path = 'input/axial_classification.csv'
        # self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_7/axial_classification.csv'
        self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_holdout_7/axial_classification_holdout.csv'
        self.train_df = pd.read_csv(self.train_df_path)

        cols = []
        label_features = [
            'spinal_canal_stenosis',
        ]
        for col in label_features:
            cols.append(f'{col}_normal')
            cols.append(f'{col}_moderate')
            cols.append(f'{col}_severe')

        self.label_features = cols
        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.image_size = 384
        self.drop_rate = 0.0
        self.drop_path_rate = 0.0
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes,
            drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)
        self.metric = None
        self.memo = ''
        self.batch_size = 8
        self.grad_accumulations = 2
        self.crop_by_xy = False
        self.rsna_2024_multi_image = False
        self.rsna_random_sample = False
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 7
        self.transform = medical_v3
        self.box_crop = True
        self.box_crop_x_ratio = 2
        self.box_crop_y_ratio = 6

class rsna_axial_spinal_dis3_crop_x05_y6(rsna_axial_spinal_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 0.5
        self.box_crop_y_ratio = 6

class rsna_axial_spinal_dis3_crop_x1_y2(rsna_axial_spinal_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 1
        self.box_crop_y_ratio = 2

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# reduce_noise axial (stage 1)
# train(by clean data) vaild(by original data)
class rsna_axial_spinal_dis3_crop_x05_y6_with_valid(rsna_axial_spinal_dis3_crop_x05_y6):
    def __init__(self, fold=0):
        super().__init__()

        self.fold = fold
        # self.train_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/axial_classification.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/axial_classification_holdout_7/axial_classification_holdout.csv'

        # 讀取原始資料
        valid_df = pd.read_csv(self.train_df_path)
        valid_df['level'] = valid_df.pred_level.map({
            1: 'l1_l2',
            2: 'l2_l3',
            3: 'l3_l4',
            4: 'l4_l5',
            5: 'l5_s1',
        })
        valid_df['study_level'] = valid_df.study_id.astype(str) + '_' + valid_df.level.str.replace('/', '_').str.lower()
        valid_df['left_right'] = 'center'  # 中央對稱特徵

        self.valid_df = valid_df.copy()  # 保留全部資料給 valid_df

        # 建立 train_df：過濾 noisy 標籤資料
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th09.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th09_holdout.csv')
        noise_df = noise_df[noise_df.target == 'spinal_canal_stenosis']
        noisy_study_levels = set(noise_df.study_level)

        self.train_df = valid_df[~valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

        # 指定分類欄位（這裡無需重新命名）
        cols = [
            'spinal_canal_stenosis_normal',
            'spinal_canal_stenosis_moderate',
            'spinal_canal_stenosis_severe',
        ]

class rsna_axial_spinal_dis3_crop_x1_y2_with_valid(rsna_axial_spinal_dis3_crop_x1_y2):
    def __init__(self, fold=0):
        super().__init__()

        self.fold = fold
        # self.train_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/axial_classification.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/axial_classification_holdout_7/axial_classification_holdout.csv'

        # 讀取完整資料
        valid_df = pd.read_csv(self.train_df_path)
        valid_df['level'] = valid_df.pred_level.map({
            1: 'l1_l2',
            2: 'l2_l3',
            3: 'l3_l4',
            4: 'l4_l5',
            5: 'l5_s1',
        })
        valid_df['study_level'] = valid_df.study_id.astype(str) + '_' + valid_df.level.str.replace('/', '_').str.lower()
        valid_df['left_right'] = 'center'

        # 保留全部資料給 valid_df
        self.valid_df = valid_df.copy()

        # 載入 noisy 標記，並過濾 train_df
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th09.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th09_holdout.csv')
        noise_df = noise_df[noise_df.target == 'spinal_canal_stenosis']
        noisy_study_levels = set(noise_df.study_level)

        self.train_df = valid_df[~valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

        # 指定欄位（實際用途可能在 downstream 過程）
        self.cols = [
            'spinal_canal_stenosis_normal',
            'spinal_canal_stenosis_moderate',
            'spinal_canal_stenosis_severe',
        ]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# reduce_noise axial ResNet50V2(stage 1)
# train(by clean data) vaild(by original data)
class rsna_axial_spinal_ResNet50V2(rsna_v1_ResNet50V2):
    def __init__(self, fold=0):
        super().__init__()
        self.fold = fold  # 我加
        # self.train_df_path = 'input/axial_classification.csv'
        # self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_7/axial_classification.csv'
        self.train_df_path = '/kaggle/working/duplicate/csv_train/axial_classification_holdout_7/axial_classification_holdout.csv'
        self.train_df = pd.read_csv(self.train_df_path)

        cols = []
        label_features = [
            'spinal_canal_stenosis',
        ]
        for col in label_features:
            cols.append(f'{col}_normal')
            cols.append(f'{col}_moderate')
            cols.append(f'{col}_severe')

        self.label_features = cols
        self.num_classes = len(self.label_features)
        self.task = "multiclass"
        
        # self.model_name = 'convnext_small.in12k_ft_in1k_384'
        # self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes,
        #     drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)
        self.model = ResNet50V2FPN(num_classes=self.num_classes, pretrained=True)
        
        self.image_size = 224  # 384
        self.batch_size = 8
        self.grad_accumulations = 2
        self.lr = 1e-4  # 5.5e-5
        self.epochs = 20  # 10
        self.transform = medical_v3
        alpha = torch.tensor([0.032, 0.375, 0.593])   # Normal, Moderate, Severe -> axial_spinal
        self.criterion = MultiClassFocalLoss(gamma=2.0, alpha=alpha)

        self.drop_rate = 0.2  # 0.1
        self.drop_path_rate = 0.0
        
        self.metric = None
        self.memo = ''
        self.crop_by_xy = False
        self.rsna_2024_multi_image = False
        self.rsna_random_sample = False
        self.rsna_2024_agg_val = False
        
        self.box_crop = True
        self.box_crop_x_ratio = 1  # 2
        self.box_crop_y_ratio = 2  # 6

        self._build_dataframes_center()

    def _build_dataframes_center(self):
        valid_df = pd.read_csv(self.train_df_path)
        valid_df['level'] = valid_df.pred_level.map({
            1: 'l1_l2',
            2: 'l2_l3',
            3: 'l3_l4',
            4: 'l4_l5',
            5: 'l5_s1',
        })
        valid_df['study_level'] = valid_df.study_id.astype(str) + '_' + valid_df.level.str.replace('/', '_').str.lower()
        valid_df['left_right'] = 'center'  # 中央對稱

        # valid 資料：全部保留
        self.valid_df = valid_df.copy()

        # train 資料：過濾 noisy
        noise_df = pd.read_csv(
            f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th09_holdout.csv'
        )
        noise_df = noise_df[noise_df.target == 'spinal_canal_stenosis']
        noisy_study_levels = set(noise_df.study_level)
        self.train_df = valid_df[~valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

        cols = [
            'spinal_canal_stenosis_normal',
            'spinal_canal_stenosis_moderate',
            'spinal_canal_stenosis_severe',
        ]

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class rsna_v1(Baseline):
    def __init__(self):
        super().__init__()
        self.compe = 'rsna_2024'
        self.predict_valid = True
        self.predict_test = False
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.transform = medical_v3  # 定義在：src/utils/augmentations/augmentation.py
        self.batch_size = 8
        self.lr = 1e-5
        self.grad_accumulations = 2
        self.p_rand_order_v1 = 0

# classification for sagittal (stage 1)
class rsna_saggital_spinal_crop_base(rsna_v1):
    def __init__(self, fold=0):  # 加上fold參數
        super().__init__()
        self.fold = fold  # 儲存fold參數
        # self.train_df_path = 'input/sagittal_spinal_range2_rolling5.csv'
        # self.train_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_spinal_range2_rolling5.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_spinal_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        dfs = []
        col = 'spinal_canal_stenosis'
        for level, idf in self.train_df.groupby('level'):
            idf[f'{col}_normal'] = 0
            idf[f'{col}_moderate'] = 0
            idf[f'{col}_severe'] = 0
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Normal/Mild', f'{col}_normal'] = 1
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Moderate', f'{col}_moderate'] = 1
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Severe', f'{col}_severe'] = 1
            idf = idf[~idf[col+'_'+level.replace('/', '_').lower()].isnull()]
            dfs.append(idf)
        self.train_df = pd.concat(dfs)            
        self.label_features = [
            'spinal_canal_stenosis_normal',
            'spinal_canal_stenosis_moderate',
            'spinal_canal_stenosis_severe',
        ]
        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.drop_rate = 0.0
        self.drop_path_rate = 0.0
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=1, drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)  # 自動只用 timm 下載的對應模型權重
        self.model = RSNA2ndModel(base_model=base_model, num_classes=len(self.label_features), pool='avg', swin=False)

        self.metric = None
        self.memo = ''
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 4
        self.image_size = 128
        self.batch_size = 16
        self.grad_accumulations = 1
        self.use_sagittal_mil_dataset = True
        self.box_crop = True
        self.box_crop_x_ratio = 1
        self.box_crop_y_ratio = 0.5
        self.predict_train = False

class rsna_saggital_mil_spinal_crop_x03_y05(rsna_saggital_spinal_crop_base):
    def __init__(self, fold=0):
        super().__init__()
        # self.fold = fold  # 我加
        self.box_crop_x_ratio = 0.3
        self.box_crop_y_ratio = 0.5

class rsna_saggital_mil_spinal_crop_x03_y07(rsna_saggital_spinal_crop_base):
    def __init__(self, fold=0):
        super().__init__()
        # self.fold = fold  # 我加
        self.box_crop_x_ratio = 0.3
        self.box_crop_y_ratio = 0.7

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# reduce_noise sagittal (stage 1)
# train(by clean data) vaild(by original data)
class rsna_saggital_mil_spinal_crop_x03_y05_with_valid(rsna_saggital_mil_spinal_crop_x03_y05):
    def __init__(self, fold=0):
        super().__init__()

        self.fold = fold
        self.label_features = [
            'spinal_canal_stenosis_normal',
            'spinal_canal_stenosis_moderate',
            'spinal_canal_stenosis_severe',
        ]

        # === 建立 valid_df（保留所有樣本） ===
        self.valid_df = self.train_df.copy()  # 原始資料應由父類別加載
        self.valid_df['study_level'] = self.valid_df.study_id.astype(str) + '_' + self.valid_df.level.str.replace('/', '_').str.lower()
        self.valid_df['left_right'] = 'center'

        # 這一行是關鍵：把 valid 的 key 再保險一次統一為小寫底線
        self.valid_df['study_level'] = self.valid_df['study_level'].astype(str).str.replace('/', '_').str.lower()

        # 去除 target 欄位為空的行
        for col in self.label_features:
            self.valid_df = self.valid_df[~self.valid_df[col].isnull()]

        # === 建立 train_df（移除 noisy 樣本） ===
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th08.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th08_holdout.csv')
        noise_df = noise_df[noise_df.target == 'spinal_canal_stenosis']
        noisy_study_levels = set(noise_df.study_level)

        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

class rsna_saggital_mil_spinal_crop_x03_y07_with_valid(rsna_saggital_mil_spinal_crop_x03_y07):
    def __init__(self, fold=0):
        super().__init__()

        self.fold = fold

        self.label_features = [
            'spinal_canal_stenosis_normal',
            'spinal_canal_stenosis_moderate',
            'spinal_canal_stenosis_severe',
        ]

        # === 建立 valid_df（保留全部資料） ===
        self.valid_df = self.train_df.copy()  # 繼承的 self.train_df 為原始資料
        self.valid_df['study_level'] = self.valid_df.study_id.astype(str) + '_' + self.valid_df.level.str.replace('/', '_').str.lower()
        self.valid_df['left_right'] = 'center'

        # 這一行是關鍵：把 valid 的 key 再保險一次統一為小寫底線
        self.valid_df['study_level'] = self.valid_df['study_level'].astype(str).str.replace('/', '_').str.lower()
        
        for col in self.label_features:
            self.valid_df = self.valid_df[~self.valid_df[col].isnull()]

        # === 建立 train_df（排除 noisy 樣本） ===
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th08.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th08_holdout.csv')
        noise_df = noise_df[noise_df.target == 'spinal_canal_stenosis']
        noisy_study_levels = set(noise_df.study_level)

        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# reduce_noise sagittal ResNet50V2(stage 1)
# train(by clean data) vaild(by original data)
class rsna_v1_ResNet50V2(Baseline_ResNet50V2):
    def __init__(self):
        super().__init__()
        self.compe = 'rsna_2024'
        self.predict_valid = True
        self.predict_test = False
        self.predict_train = False
        # self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.transform = medical_v3  # 定義在：src/utils/augmentations/augmentation.py
        self.batch_size = 8
        self.lr = 1e-5
        self.grad_accumulations = 2
        self.p_rand_order_v1 = 0

class rsna_saggital_mil_spinal_ResNet50V2(rsna_v1_ResNet50V2):
    def __init__(self, fold=0):  # 加上fold參數
        super().__init__()
        self.fold = fold  # 儲存fold參數
        # self.train_df_path = 'input/sagittal_spinal_range2_rolling5.csv'
        # self.train_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_spinal_range2_rolling5.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_spinal_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        
        dfs = []
        col = 'spinal_canal_stenosis'
        for level, idf in self.train_df.groupby('level'):
            idf[f'{col}_normal'] = 0
            idf[f'{col}_moderate'] = 0
            idf[f'{col}_severe'] = 0
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Normal/Mild', f'{col}_normal'] = 1
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Moderate', f'{col}_moderate'] = 1
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Severe', f'{col}_severe'] = 1
            idf = idf[~idf[col+'_'+level.replace('/', '_').lower()].isnull()]
            dfs.append(idf)
        self.train_df = pd.concat(dfs)            
        
        self.label_features = [
            'spinal_canal_stenosis_normal',
            'spinal_canal_stenosis_moderate',
            'spinal_canal_stenosis_severe',
        ]
        self.num_classes = len(self.label_features)
        self.task = "multiclass"
        # self.model_name = 'convnext_small.in12k_ft_in1k_384'
        # base_model = timm.create_model(self.model_name, pretrained=True, num_classes=1, drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)  # 自動只用 timm 下載的對應模型權重
        # self.model = RSNA2ndModel(base_model=base_model, num_classes=len(self.label_features), pool='avg', swin=False)
        # self.model = ResNet50V2FPN(num_classes=self.num_classes, pretrained=True)
        # base_model = ResNet50V2FPN(num_classes=self.num_classes, pretrained=True)
        # self.model = SagittalMILModel(base_model, num_classes=self.num_classes, pooling="attention")
        self.model = ResNet50V2FPN(num_classes=self.num_classes, pretrained=True)
        
        self.metric = None
        self.memo = ''
        self.lr = 1e-4  # 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 20  # 10  # 4
        self.image_size = 228  # 128
        self.batch_size = 16
        self.grad_accumulations = 1
        alpha = torch.tensor([0.031, 0.371, 0.598])   # Normal, Moderate, Severe -> axial_spinal -> fix
        self.criterion = MultiClassFocalLoss(gamma=2.0, alpha=alpha)

        self.drop_rate = 0.2  # 0.0
        self.drop_path_rate = 0.0
        
        self.use_sagittal_mil_dataset = False  # True
        self.box_crop = True
        self.box_crop_x_ratio = 1  # 1
        self.box_crop_y_ratio = 1  # 0.5
        self.predict_train = False

        self._build_dataframes_center()

    def _build_dataframes_center(self):
        """建立 sagittal spinal 的 train_df 與 valid_df"""
        valid_df = self.train_df.copy()

        # 統一 study_level key
        valid_df['study_level'] = valid_df.study_id.astype(str) + '_' + valid_df.level.str.replace('/', '_').str.lower()
        valid_df['study_level'] = valid_df['study_level'].astype(str).str.replace('/', '_').str.lower()
        valid_df['left_right'] = 'center'

        # 去除 target 欄位為空的行
        for col in self.label_features:
            valid_df = valid_df[~valid_df[col].isnull()]

        # === 建立 valid_df（保留所有樣本） ===
        self.valid_df = valid_df.copy()

        # === 建立 train_df（移除 noisy 樣本） ===
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th08_holdout.csv')
        noise_df = noise_df[noise_df.target == 'spinal_canal_stenosis']
        noisy_study_levels = set(noise_df.study_level)
        self.train_df = valid_df[~valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# classification for sagittal (stage 2)
class rsna_saggital_mil_ss_crop_base(rsna_v1):
    def __init__(self):
        super().__init__()
        # self.train_df_path = 'input/sagittal_right_ss_range2_rolling5.csv'
        # self.train_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_right_ss_range2_rolling5.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_ss_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['subarticular_stenosis_normal'] = self.train_df['right_subarticular_stenosis_normal'].values
        self.train_df['subarticular_stenosis_moderate'] = self.train_df['right_subarticular_stenosis_moderate'].values
        self.train_df['subarticular_stenosis_severe'] = self.train_df['right_subarticular_stenosis_severe'].values
        # self.add_df_path = 'input/sagittal_left_ss_range2_rolling5.csv'
        # self.add_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_left_ss_range2_rolling5.csv' 
        self.add_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_ss_range2_rolling5.csv' 
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['subarticular_stenosis_normal'] = self.add_df['left_subarticular_stenosis_normal'].values
        self.add_df['subarticular_stenosis_moderate'] = self.add_df['left_subarticular_stenosis_moderate'].values
        self.add_df['subarticular_stenosis_severe'] = self.add_df['left_subarticular_stenosis_severe'].values
        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'subarticular_stenosis_normal',
            'subarticular_stenosis_moderate',
            'subarticular_stenosis_severe',
        ]
        l = len(self.train_df)
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
        print(l, len(self.train_df))

        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'

        # self.metric = MultiAUC(label_features=self.label_features).torch
        self.metric = None
        self.memo = ''
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 6
        self.batch_size = 16
        self.grad_accumulations = 1
        self.use_sagittal_mil_dataset = True
        self.drop_rate = 0.0
        self.drop_path_rate = 0.0

        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=1, drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)
        self.model = RSNA2ndModel(base_model=base_model, num_classes=len(self.label_features), pool='avg', swin=False)
        
        self.box_crop = True
        self.box_crop_x_ratio = 0.4
        self.box_crop_y_ratio = 0.2
        self.predict_train = False

class rsna_saggital_mil_ss_crop_x03_y05_96(rsna_saggital_mil_ss_crop_base):
    def __init__(self, fold=0):
        super().__init__()
        # self.fold = fold  # 我加
        self.box_crop_x_ratio = 0.3
        self.box_crop_y_ratio = 0.5
        self.image_size = 96

class rsna_saggital_mil_ss_crop_x03_y07_96(rsna_saggital_mil_ss_crop_base):
    def __init__(self, fold=0):
        super().__init__()
        # self.fold = fold  # 我加
        self.box_crop_x_ratio = 0.3
        self.box_crop_y_ratio = 0.7
        self.image_size = 96

class rsna_saggital_mil_ss_crop_x03_y2_96(rsna_saggital_mil_ss_crop_base):
    def __init__(self, fold=0):
        super().__init__()
        # self.fold = fold  # 我加
        self.box_crop_x_ratio = 0.3
        self.box_crop_y_ratio = 2
        self.image_size = 96

class rsna_saggital_mil_ss_crop_x1_y07_96(rsna_saggital_mil_ss_crop_base):
    def __init__(self, fold=0):
        super().__init__()
        # self.fold = fold  # 我加
        self.box_crop_x_ratio = 1
        self.box_crop_y_ratio = 0.7
        self.image_size = 96

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# reduce_noise sagittal (stage 2)
# train(by clean data) vaild(by original data)
class rsna_saggital_mil_ss_crop_x03_y05_96_with_valid(rsna_saggital_mil_ss_crop_x03_y05_96):
    def __init__(self, fold=0):
        super().__init__()

        self.fold = fold

        # === 讀取右側資料 ===
        # right_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_right_ss_range2_rolling5.csv'
        right_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_ss_range2_rolling5.csv'
        right_df = pd.read_csv(right_df_path)
        right_df['left_right'] = 'right'
        right_df['subarticular_stenosis_normal'] = right_df['right_subarticular_stenosis_normal']
        right_df['subarticular_stenosis_moderate'] = right_df['right_subarticular_stenosis_moderate']
        right_df['subarticular_stenosis_severe'] = right_df['right_subarticular_stenosis_severe']
        right_df['study_level'] = right_df.study_id.astype(str) + '_' + right_df.level.str.replace('/', '_').str.lower()

        # === 讀取左側資料 ===
        # left_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_left_ss_range2_rolling5.csv'
        left_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_ss_range2_rolling5.csv'
        left_df = pd.read_csv(left_df_path)
        left_df['left_right'] = 'left'
        left_df['subarticular_stenosis_normal'] = left_df['left_subarticular_stenosis_normal']
        left_df['subarticular_stenosis_moderate'] = left_df['left_subarticular_stenosis_moderate']
        left_df['subarticular_stenosis_severe'] = left_df['left_subarticular_stenosis_severe']
        left_df['study_level'] = left_df.study_id.astype(str) + '_' + left_df.level.str.replace('/', '_').str.lower()

        # === 合併作為 valid_df（保留 noisy） ===
        self.valid_df = pd.concat([right_df, left_df], ignore_index=True)

        # === 過濾 noisy 標記，建立 train_df ===
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th08.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th08_holdout.csv')
        noise_df = noise_df[noise_df.target.isin(['right_subarticular_stenosis', 'left_subarticular_stenosis'])]
        noisy_study_levels = set(noise_df.study_level)

        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

        # === 去除標籤為空的樣本 ===
        self.label_features = [
            'subarticular_stenosis_normal',
            'subarticular_stenosis_moderate',
            'subarticular_stenosis_severe',
        ]
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
            self.valid_df = self.valid_df[~self.valid_df[col].isnull()]

class rsna_saggital_mil_ss_crop_x03_y07_96_with_valid(rsna_saggital_mil_ss_crop_x03_y07_96):
    def __init__(self, fold=0):
        super().__init__()

        self.fold = fold

        # === 讀取右側資料 ===
        # right_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_right_ss_range2_rolling5.csv'
        right_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_ss_range2_rolling5.csv'
        right_df = pd.read_csv(right_df_path)
        right_df['left_right'] = 'right'
        right_df['subarticular_stenosis_normal'] = right_df['right_subarticular_stenosis_normal']
        right_df['subarticular_stenosis_moderate'] = right_df['right_subarticular_stenosis_moderate']
        right_df['subarticular_stenosis_severe'] = right_df['right_subarticular_stenosis_severe']
        right_df['study_level'] = right_df.study_id.astype(str) + '_' + right_df.level.str.replace('/', '_').str.lower()

        # === 讀取左側資料 ===
        # left_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_left_ss_range2_rolling5.csv'
        left_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_ss_range2_rolling5.csv'
        left_df = pd.read_csv(left_df_path)
        left_df['left_right'] = 'left'
        left_df['subarticular_stenosis_normal'] = left_df['left_subarticular_stenosis_normal']
        left_df['subarticular_stenosis_moderate'] = left_df['left_subarticular_stenosis_moderate']
        left_df['subarticular_stenosis_severe'] = left_df['left_subarticular_stenosis_severe']
        left_df['study_level'] = left_df.study_id.astype(str) + '_' + left_df.level.str.replace('/', '_').str.lower()

        # === 建立 valid_df（完整資料） ===
        self.valid_df = pd.concat([right_df, left_df], ignore_index=True)

        # === 載入 noisy study_level 標記並建立 train_df ===
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th08.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th08_holdout.csv')
        noise_df = noise_df[noise_df.target.isin(['left_subarticular_stenosis', 'right_subarticular_stenosis'])]
        noisy_study_levels = set(noise_df.study_level)

        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

        # === 去除標籤為空的樣本 ===
        self.label_features = [
            'subarticular_stenosis_normal',
            'subarticular_stenosis_moderate',
            'subarticular_stenosis_severe',
        ]
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
            self.valid_df = self.valid_df[~self.valid_df[col].isnull()]

class rsna_saggital_mil_ss_crop_x03_y2_96_with_valid(rsna_saggital_mil_ss_crop_x03_y2_96):
    def __init__(self, fold=0):
        super().__init__()

        self.fold = fold

        # === 載入右側資料 ===
        # right_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_right_ss_range2_rolling5.csv'
        right_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_ss_range2_rolling5.csv'
        right_df = pd.read_csv(right_df_path)
        right_df['left_right'] = 'right'
        right_df['subarticular_stenosis_normal'] = right_df['right_subarticular_stenosis_normal']
        right_df['subarticular_stenosis_moderate'] = right_df['right_subarticular_stenosis_moderate']
        right_df['subarticular_stenosis_severe'] = right_df['right_subarticular_stenosis_severe']
        right_df['study_level'] = right_df.study_id.astype(str) + '_' + right_df.level.str.replace('/', '_').str.lower()

        # === 載入左側資料 ===
        # left_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_left_ss_range2_rolling5.csv'
        left_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_ss_range2_rolling5.csv'
        left_df = pd.read_csv(left_df_path)
        left_df['left_right'] = 'left'
        left_df['subarticular_stenosis_normal'] = left_df['left_subarticular_stenosis_normal']
        left_df['subarticular_stenosis_moderate'] = left_df['left_subarticular_stenosis_moderate']
        left_df['subarticular_stenosis_severe'] = left_df['left_subarticular_stenosis_severe']
        left_df['study_level'] = left_df.study_id.astype(str) + '_' + left_df.level.str.replace('/', '_').str.lower()

        # === 建立 valid_df（不去除 noisy 樣本） ===
        self.valid_df = pd.concat([right_df, left_df], ignore_index=True)

        # === 載入 noisy 標記並建立 train_df（移除 noisy） ===
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th08.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th08_holdout.csv')
        noise_df = noise_df[noise_df.target.isin(['right_subarticular_stenosis', 'left_subarticular_stenosis'])]
        noisy_study_levels = set(noise_df.study_level)

        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

        # === 過濾缺 label 的樣本 ===
        self.label_features = [
            'subarticular_stenosis_normal',
            'subarticular_stenosis_moderate',
            'subarticular_stenosis_severe',
        ]
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
            self.valid_df = self.valid_df[~self.valid_df[col].isnull()]

class rsna_saggital_mil_ss_crop_x1_y07_96_with_valid(rsna_saggital_mil_ss_crop_x1_y07_96):
    def __init__(self, fold=0):
        super().__init__()

        self.fold = fold

        # === 讀取右側資料 ===
        # right_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_right_ss_range2_rolling5.csv'
        right_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_ss_range2_rolling5.csv'
        right_df = pd.read_csv(right_df_path)
        right_df['left_right'] = 'right'
        right_df['subarticular_stenosis_normal'] = right_df['right_subarticular_stenosis_normal']
        right_df['subarticular_stenosis_moderate'] = right_df['right_subarticular_stenosis_moderate']
        right_df['subarticular_stenosis_severe'] = right_df['right_subarticular_stenosis_severe']
        right_df['study_level'] = right_df.study_id.astype(str) + '_' + right_df.level.str.replace('/', '_').str.lower()

        # === 讀取左側資料 ===
        # left_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_left_ss_range2_rolling5.csv'
        left_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_ss_range2_rolling5.csv'
        left_df = pd.read_csv(left_df_path)
        left_df['left_right'] = 'left'
        left_df['subarticular_stenosis_normal'] = left_df['left_subarticular_stenosis_normal']
        left_df['subarticular_stenosis_moderate'] = left_df['left_subarticular_stenosis_moderate']
        left_df['subarticular_stenosis_severe'] = left_df['left_subarticular_stenosis_severe']
        left_df['study_level'] = left_df.study_id.astype(str) + '_' + left_df.level.str.replace('/', '_').str.lower()

        # === 建立 valid_df（完整未過濾） ===
        self.valid_df = pd.concat([right_df, left_df], ignore_index=True)

        # === 載入 noisy 標記並建立 train_df ===
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th08.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th08_holdout.csv')
        noise_df = noise_df[noise_df.target.isin(['right_subarticular_stenosis', 'left_subarticular_stenosis'])]
        noisy_study_levels = set(noise_df.study_level)

        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

        # === 標籤欄位定義 + 過濾缺值樣本 ===
        self.label_features = [
            'subarticular_stenosis_normal',
            'subarticular_stenosis_moderate',
            'subarticular_stenosis_severe',
        ]
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
            self.valid_df = self.valid_df[~self.valid_df[col].isnull()]

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# reduce_noise sagittal ResNet50V2(stage 2)
# train(by clean data) vaild(by original data)
class rsna_saggital_mil_ss_ResNet50V2(rsna_v1_ResNet50V2):
    def __init__(self):
        super().__init__()
        # self.train_df_path = 'input/sagittal_right_ss_range2_rolling5.csv'
        # self.train_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_right_ss_range2_rolling5.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_ss_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['subarticular_stenosis_normal'] = self.train_df['right_subarticular_stenosis_normal'].values
        self.train_df['subarticular_stenosis_moderate'] = self.train_df['right_subarticular_stenosis_moderate'].values
        self.train_df['subarticular_stenosis_severe'] = self.train_df['right_subarticular_stenosis_severe'].values
        # self.add_df_path = 'input/sagittal_left_ss_range2_rolling5.csv'
        # self.add_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_left_ss_range2_rolling5.csv' 
        self.add_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_ss_range2_rolling5.csv'
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['subarticular_stenosis_normal'] = self.add_df['left_subarticular_stenosis_normal'].values
        self.add_df['subarticular_stenosis_moderate'] = self.add_df['left_subarticular_stenosis_moderate'].values
        self.add_df['subarticular_stenosis_severe'] = self.add_df['left_subarticular_stenosis_severe'].values
        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'subarticular_stenosis_normal',
            'subarticular_stenosis_moderate',
            'subarticular_stenosis_severe',
        ]
        l = len(self.train_df)
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
        print(l, len(self.train_df))

        self.num_classes = len(self.label_features)
        self.task = "multiclass"
        # self.model_name = 'convnext_small.in12k_ft_in1k_384'
        # base_model = timm.create_model(self.model_name, pretrained=True, num_classes=1, drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)
        # self.model = RSNA2ndModel(base_model=base_model, num_classes=len(self.label_features), pool='avg', swin=False)
        # self.model = ResNet50V2FPN(num_classes=self.num_classes, pretrained=True)
        # base_model = ResNet50V2FPN(num_classes=self.num_classes, pretrained=True)
        # self.model = SagittalMILModel(base_model, num_classes=self.num_classes, pooling="attention")
        self.model = ResNet50V2FPN(num_classes=self.num_classes, pretrained=True)

        # self.metric = MultiAUC(label_features=self.label_features).torch
        self.metric = None
        self.memo = ''
        self.lr = 1e-4  # 5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 20  # 10  # 6
        self.batch_size = 16
        self.grad_accumulations = 1
        self.use_sagittal_mil_dataset = False  # True

        alpha = torch.tensor([0.081, 0.303, 0.616])   # Normal, Moderate, Severe -> axial_spinal -> fix
        self.criterion = MultiClassFocalLoss(gamma=2.0, alpha=alpha)
        
        self.drop_rate = 0.2  # 0.0
        self.drop_path_rate = 0.0
        
        self.box_crop = True
        self.box_crop_x_ratio = 1  # 0.4
        self.box_crop_y_ratio = 1  # 0.2
        self.predict_train = False

        self._build_dataframes_bilateral()

    def _build_dataframes_bilateral(self):
        """建立 sagittal subarticular stenosis 的 train_df / valid_df"""

        # === 讀取右側資料 ===
        right_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_ss_range2_rolling5.csv'
        right_df = pd.read_csv(right_df_path)
        right_df['left_right'] = 'right'
        right_df['subarticular_stenosis_normal'] = right_df['right_subarticular_stenosis_normal']
        right_df['subarticular_stenosis_moderate'] = right_df['right_subarticular_stenosis_moderate']
        right_df['subarticular_stenosis_severe'] = right_df['right_subarticular_stenosis_severe']
        right_df['study_level'] = right_df.study_id.astype(str) + '_' + right_df.level.str.replace('/', '_').str.lower()

        # === 讀取左側資料 ===
        left_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_ss_range2_rolling5.csv'
        left_df = pd.read_csv(left_df_path)
        left_df['left_right'] = 'left'
        left_df['subarticular_stenosis_normal'] = left_df['left_subarticular_stenosis_normal']
        left_df['subarticular_stenosis_moderate'] = left_df['left_subarticular_stenosis_moderate']
        left_df['subarticular_stenosis_severe'] = left_df['left_subarticular_stenosis_severe']
        left_df['study_level'] = left_df.study_id.astype(str) + '_' + left_df.level.str.replace('/', '_').str.lower()

        # === 合併作為 valid_df（保留 noisy） ===
        self.valid_df = pd.concat([right_df, left_df], ignore_index=True)

        # === 過濾 noisy，建立 train_df ===
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th08_holdout.csv')
        noise_df = noise_df[noise_df.target.isin(['right_subarticular_stenosis', 'left_subarticular_stenosis'])]
        noisy_study_levels = set(noise_df.study_level)
        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

        # === 去除標籤為空的樣本 ===
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
            self.valid_df = self.valid_df[~self.valid_df[col].isnull()]



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# classification for sagittal (stage 3)
class rsna_saggital_mil_nfn_crop_base(rsna_v1):
    def __init__(self):
        super().__init__()
        # self.train_df_path = 'input/sagittal_right_nfn_range2_rolling5.csv'
        # self.train_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_right_nfn_range2_rolling5.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_nfn_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['neural_foraminal_narrowing_normal'] = self.train_df['right_neural_foraminal_narrowing_normal'].values
        self.train_df['neural_foraminal_narrowing_moderate'] = self.train_df['right_neural_foraminal_narrowing_moderate'].values
        self.train_df['neural_foraminal_narrowing_severe'] = self.train_df['right_neural_foraminal_narrowing_severe'].values
        # self.add_df_path = 'input/sagittal_left_nfn_range2_rolling5.csv'
        # self.add_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_left_nfn_range2_rolling5.csv'
        self.add_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_nfn_range2_rolling5.csv'
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['neural_foraminal_narrowing_normal'] = self.add_df['left_neural_foraminal_narrowing_normal'].values
        self.add_df['neural_foraminal_narrowing_moderate'] = self.add_df['left_neural_foraminal_narrowing_moderate'].values
        self.add_df['neural_foraminal_narrowing_severe'] = self.add_df['left_neural_foraminal_narrowing_severe'].values
        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'neural_foraminal_narrowing_normal',
            'neural_foraminal_narrowing_moderate',
            'neural_foraminal_narrowing_severe',
        ]
        l = len(self.train_df)
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
        print(l, len(self.train_df))

        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'

        self.metric = None
        self.memo = ''
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 6
        self.image_size = 160
        self.batch_size = 16
        self.grad_accumulations = 1
        self.use_sagittal_mil_dataset = True
        self.ch_3_crop = True
        self.drop_rate = 0.0
        self.drop_path_rate = 0.0

        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=1, drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)
        self.model = RSNA2ndModel(base_model=base_model, num_classes=len(self.label_features), pool='avg', swin=False)

        self.box_crop = True
        self.box_crop_x_ratio = 0.4
        self.box_crop_y_ratio = 0.2
        self.predict_train = False

class rsna_saggital_mil_nfn_crop_x07_y1_v2(rsna_saggital_mil_nfn_crop_base):
    def __init__(self, fold=0):
        super().__init__()
        # self.fold = fold  # 我加
        self.box_crop_x_ratio = 0.7
        self.box_crop_y_ratio = 1

class rsna_saggital_mil_nfn_crop_x15_y1_v2(rsna_saggital_mil_nfn_crop_base):
    def __init__(self, fold=0):
        super().__init__()
        # self.fold = fold  # 我加
        self.box_crop_x_ratio = 1.5
        self.box_crop_y_ratio = 1

class rsna_saggital_mil_nfn_crop_x03_y1_v2(rsna_saggital_mil_nfn_crop_base):
    def __init__(self, fold=0):
        super().__init__()
        # self.fold = fold  # 我加
        self.box_crop_x_ratio = 0.3
        self.box_crop_y_ratio = 1

class rsna_saggital_mil_nfn_crop_x05_y05_v2(rsna_saggital_mil_nfn_crop_base):
    def __init__(self, fold=0):
        super().__init__()
        # self.fold = fold  # 我加
        self.box_crop_x_ratio = 0.5
        self.box_crop_y_ratio = 0.5

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# reduce_noise sagittal (stage 3)
# train(by clean data) vaild(by original data)
class rsna_saggital_mil_nfn_crop_x07_y1_v2_with_valid(rsna_saggital_mil_nfn_crop_x07_y1_v2):
    def __init__(self, fold=0):
        super().__init__()

        self.fold = fold

        # === 讀取右側資料 ===
        # right_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_right_nfn_range2_rolling5.csv'
        right_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_nfn_range2_rolling5.csv'
        right_df = pd.read_csv(right_df_path)
        right_df['left_right'] = 'right'
        right_df['neural_foraminal_narrowing_normal'] = right_df['right_neural_foraminal_narrowing_normal']
        right_df['neural_foraminal_narrowing_moderate'] = right_df['right_neural_foraminal_narrowing_moderate']
        right_df['neural_foraminal_narrowing_severe'] = right_df['right_neural_foraminal_narrowing_severe']
        right_df['study_level'] = right_df.study_id.astype(str) + '_' + right_df.level.str.replace('/', '_').str.lower()

        # === 讀取左側資料 ===
        # left_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_left_nfn_range2_rolling5.csv'
        left_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_nfn_range2_rolling5.csv'
        left_df = pd.read_csv(left_df_path)
        left_df['left_right'] = 'left'
        left_df['neural_foraminal_narrowing_normal'] = left_df['left_neural_foraminal_narrowing_normal']
        left_df['neural_foraminal_narrowing_moderate'] = left_df['left_neural_foraminal_narrowing_moderate']
        left_df['neural_foraminal_narrowing_severe'] = left_df['left_neural_foraminal_narrowing_severe']
        left_df['study_level'] = left_df.study_id.astype(str) + '_' + left_df.level.str.replace('/', '_').str.lower()

        # === 建立 valid_df（保留全部資料） ===
        self.valid_df = pd.concat([right_df, left_df], ignore_index=True)

        # === 載入 noisy 標記資料並篩掉 noisy study_level ===
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th08.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th08_holdout.csv')
        noise_df = noise_df[noise_df.target.isin([
            'right_neural_foraminal_narrowing',
            'left_neural_foraminal_narrowing'
        ])]
        noisy_study_levels = set(noise_df.study_level)
        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

        # === 指定標籤欄位，並過濾 NaN 標籤樣本 ===
        self.label_features = [
            'neural_foraminal_narrowing_normal',
            'neural_foraminal_narrowing_moderate',
            'neural_foraminal_narrowing_severe',
        ]
        l = len(self.train_df)
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
            self.valid_df = self.valid_df[~self.valid_df[col].isnull()]
        print(f'Before label NaN filtering: {l}, After: {len(self.train_df)}')

class rsna_saggital_mil_nfn_crop_x15_y1_v2_with_valid(rsna_saggital_mil_nfn_crop_x15_y1_v2):
    def __init__(self, fold=0):
        super().__init__()

        self.fold = fold

        # === 右側資料 ===
        # right_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_right_nfn_range2_rolling5.csv'
        right_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_nfn_range2_rolling5.csv'
        right_df = pd.read_csv(right_df_path)
        right_df['left_right'] = 'right'
        right_df['neural_foraminal_narrowing_normal'] = right_df['right_neural_foraminal_narrowing_normal']
        right_df['neural_foraminal_narrowing_moderate'] = right_df['right_neural_foraminal_narrowing_moderate']
        right_df['neural_foraminal_narrowing_severe'] = right_df['right_neural_foraminal_narrowing_severe']
        right_df['study_level'] = right_df.study_id.astype(str) + '_' + right_df.level.str.replace('/', '_').str.lower()

        # === 左側資料 ===
        # left_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_left_nfn_range2_rolling5.csv'
        left_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_nfn_range2_rolling5.csv'
        left_df = pd.read_csv(left_df_path)
        left_df['left_right'] = 'left'
        left_df['neural_foraminal_narrowing_normal'] = left_df['left_neural_foraminal_narrowing_normal']
        left_df['neural_foraminal_narrowing_moderate'] = left_df['left_neural_foraminal_narrowing_moderate']
        left_df['neural_foraminal_narrowing_severe'] = left_df['left_neural_foraminal_narrowing_severe']
        left_df['study_level'] = left_df.study_id.astype(str) + '_' + left_df.level.str.replace('/', '_').str.lower()

        # === 合併所有資料作為 valid_df（完整資料）
        self.valid_df = pd.concat([right_df, left_df], ignore_index=True)

        # === 去除 noisy 資料後形成 train_df
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th08.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th08_holdout.csv')
        noise_df = noise_df[noise_df.target.isin([
            'right_neural_foraminal_narrowing',
            'left_neural_foraminal_narrowing'
        ])]
        noisy_study_levels = set(noise_df.study_level)
        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

        # === 指定標籤欄位，並移除 NaN 標籤樣本
        self.label_features = [
            'neural_foraminal_narrowing_normal',
            'neural_foraminal_narrowing_moderate',
            'neural_foraminal_narrowing_severe',
        ]
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
            self.valid_df = self.valid_df[~self.valid_df[col].isnull()]

        print(f"Train: {len(self.train_df)}, Valid: {len(self.valid_df)}")

class rsna_saggital_mil_nfn_crop_x03_y1_v2_with_valid(rsna_saggital_mil_nfn_crop_x03_y1_v2):
    def __init__(self, fold=0):
        super().__init__()

        self.fold = fold  # 儲存 fold 參數

        # === 載入右側資料 ===
        # right_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_right_nfn_range2_rolling5.csv'
        right_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_nfn_range2_rolling5.csv'
        right_df = pd.read_csv(right_df_path)
        right_df['left_right'] = 'right'
        right_df['neural_foraminal_narrowing_normal'] = right_df['right_neural_foraminal_narrowing_normal']
        right_df['neural_foraminal_narrowing_moderate'] = right_df['right_neural_foraminal_narrowing_moderate']
        right_df['neural_foraminal_narrowing_severe'] = right_df['right_neural_foraminal_narrowing_severe']
        right_df['study_level'] = right_df.study_id.astype(str) + '_' + right_df.level.str.replace('/', '_').str.lower()

        # === 載入左側資料 ===
        # left_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_left_nfn_range2_rolling5.csv'
        left_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_nfn_range2_rolling5.csv'
        left_df = pd.read_csv(left_df_path)
        left_df['left_right'] = 'left'
        left_df['neural_foraminal_narrowing_normal'] = left_df['left_neural_foraminal_narrowing_normal']
        left_df['neural_foraminal_narrowing_moderate'] = left_df['left_neural_foraminal_narrowing_moderate']
        left_df['neural_foraminal_narrowing_severe'] = left_df['left_neural_foraminal_narrowing_severe']
        left_df['study_level'] = left_df.study_id.astype(str) + '_' + left_df.level.str.replace('/', '_').str.lower()

        # === 合併作為 valid_df（完整資料，僅去除 NaN）===
        self.valid_df = pd.concat([right_df, left_df], ignore_index=True)

        # === 讀入 noisy 樣本資訊 ===
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th08.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th08_holdout.csv')
        noise_df = noise_df[noise_df.target.isin([
            'left_neural_foraminal_narrowing',
            'right_neural_foraminal_narrowing'
        ])]
        noisy_study_levels = set(noise_df.study_level)

        # === 去除 noisy 樣本，作為 train_df ===
        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

        # === 指定 label 欄位，並移除 NaN 標籤樣本 ===
        self.label_features = [
            'neural_foraminal_narrowing_normal',
            'neural_foraminal_narrowing_moderate',
            'neural_foraminal_narrowing_severe',
        ]
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
            self.valid_df = self.valid_df[~self.valid_df[col].isnull()]

        print(f"Train size (filtered): {len(self.train_df)}")
        print(f"Valid size (full set, clean labels): {len(self.valid_df)}")

class rsna_saggital_mil_nfn_crop_x05_y05_v2_with_valid(rsna_saggital_mil_nfn_crop_x05_y05_v2):
    def __init__(self, fold=0):
        super().__init__()

        self.fold = fold  # 儲存 fold

        # === 載入右側資料 ===
        # right_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_right_nfn_range2_rolling5.csv'
        right_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_nfn_range2_rolling5.csv'
        right_df = pd.read_csv(right_df_path)
        right_df['left_right'] = 'right'
        right_df['neural_foraminal_narrowing_normal'] = right_df['right_neural_foraminal_narrowing_normal']
        right_df['neural_foraminal_narrowing_moderate'] = right_df['right_neural_foraminal_narrowing_moderate']
        right_df['neural_foraminal_narrowing_severe'] = right_df['right_neural_foraminal_narrowing_severe']
        right_df['study_level'] = right_df.study_id.astype(str) + '_' + right_df.level.str.replace('/', '_').str.lower()

        # === 載入左側資料 ===
        # left_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_left_nfn_range2_rolling5.csv'
        left_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_nfn_range2_rolling5.csv'
        left_df = pd.read_csv(left_df_path)
        left_df['left_right'] = 'left'
        left_df['neural_foraminal_narrowing_normal'] = left_df['left_neural_foraminal_narrowing_normal']
        left_df['neural_foraminal_narrowing_moderate'] = left_df['left_neural_foraminal_narrowing_moderate']
        left_df['neural_foraminal_narrowing_severe'] = left_df['left_neural_foraminal_narrowing_severe']
        left_df['study_level'] = left_df.study_id.astype(str) + '_' + left_df.level.str.replace('/', '_').str.lower()

        # === 合併 valid_df，僅清除 NaN ===
        self.valid_df = pd.concat([right_df, left_df], ignore_index=True)

        # === 載入 noisy study_level 清單 ===
        # noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th08.csv')
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th08_holdout.csv')
        noise_df = noise_df[noise_df.target.isin([
            'right_neural_foraminal_narrowing',
            'left_neural_foraminal_narrowing'
        ])]
        noisy_study_levels = set(noise_df.study_level)

        # === 建立 train_df（排除 noisy）===
        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

        # === 移除 NaN labels ===
        self.label_features = [
            'neural_foraminal_narrowing_normal',
            'neural_foraminal_narrowing_moderate',
            'neural_foraminal_narrowing_severe',
        ]
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
            self.valid_df = self.valid_df[~self.valid_df[col].isnull()]

        print(f"Train set size (denoised): {len(self.train_df)}")
        print(f"Valid set size (full, clean labels): {len(self.valid_df)}")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# reduce_noise sagittal ResNet50V2(stage 3)
# train(by clean data) vaild(by original data)
class rsna_saggital_mil_nfn_ResNet50V2(rsna_v1_ResNet50V2):
    def __init__(self):
        super().__init__()
        # self.train_df_path = 'input/sagittal_right_nfn_range2_rolling5.csv'
        # self.train_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_right_nfn_range2_rolling5.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_nfn_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['neural_foraminal_narrowing_normal'] = self.train_df['right_neural_foraminal_narrowing_normal'].values
        self.train_df['neural_foraminal_narrowing_moderate'] = self.train_df['right_neural_foraminal_narrowing_moderate'].values
        self.train_df['neural_foraminal_narrowing_severe'] = self.train_df['right_neural_foraminal_narrowing_severe'].values
        # self.add_df_path = 'input/sagittal_left_nfn_range2_rolling5.csv'
        # self.add_df_path = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_left_nfn_range2_rolling5.csv'
        self.add_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_nfn_range2_rolling5.csv'
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['neural_foraminal_narrowing_normal'] = self.add_df['left_neural_foraminal_narrowing_normal'].values
        self.add_df['neural_foraminal_narrowing_moderate'] = self.add_df['left_neural_foraminal_narrowing_moderate'].values
        self.add_df['neural_foraminal_narrowing_severe'] = self.add_df['left_neural_foraminal_narrowing_severe'].values
        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'neural_foraminal_narrowing_normal',
            'neural_foraminal_narrowing_moderate',
            'neural_foraminal_narrowing_severe',
        ]
        l = len(self.train_df)
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
        print(l, len(self.train_df))

        self.num_classes = len(self.label_features)
        self.task = "multiclass"
        # self.model_name = 'convnext_small.in12k_ft_in1k_384'
        # base_model = timm.create_model(self.model_name, pretrained=True, num_classes=1, drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)
        # self.model = RSNA2ndModel(base_model=base_model, num_classes=len(self.label_features), pool='avg', swin=False)
        # self.model = ResNet50V2FPN(num_classes=self.num_classes, pretrained=True)
        self.model = ResNet50V2FPN(num_classes=self.num_classes, pretrained=True)

        self.metric = None
        self.memo = ''
        self.lr = 1e-4  # 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 20  # 10  # 6
        self.image_size = 228  # 160
        self.batch_size = 16
        self.grad_accumulations = 1
        self.use_sagittal_mil_dataset = False  # True
        self.ch_3_crop = True
        alpha = torch.tensor([0.040, 0.173, 0.787])   # Normal, Moderate, Severe -> axial_spinal -> fix
        self.criterion = MultiClassFocalLoss(gamma=2.0, alpha=alpha)
        
        self.drop_rate = 0.2  # 0.0
        self.drop_path_rate = 0.0

        self.box_crop = True
        self.box_crop_x_ratio = 1  # 0.4
        self.box_crop_y_ratio = 1  # 0.2
        self.predict_train = False
        self._build_dataframes_bilateral()

    def _build_dataframes_bilateral(self):
        """建立 sagittal NFN (neural foraminal narrowing) 的 train_df / valid_df"""

        # === 讀取右側資料 ===
        right_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_right_nfn_range2_rolling5.csv'
        right_df = pd.read_csv(right_df_path)
        right_df['left_right'] = 'right'
        right_df['neural_foraminal_narrowing_normal'] = right_df['right_neural_foraminal_narrowing_normal']
        right_df['neural_foraminal_narrowing_moderate'] = right_df['right_neural_foraminal_narrowing_moderate']
        right_df['neural_foraminal_narrowing_severe'] = right_df['right_neural_foraminal_narrowing_severe']
        right_df['study_level'] = right_df.study_id.astype(str) + '_' + right_df.level.str.replace('/', '_').str.lower()

        # === 讀取左側資料 ===
        left_df_path = f'{WORKING_DIR}/csv_train/sagittal_classification_holdout_8/sagittal_left_nfn_range2_rolling5.csv'
        left_df = pd.read_csv(left_df_path)
        left_df['left_right'] = 'left'
        left_df['neural_foraminal_narrowing_normal'] = left_df['left_neural_foraminal_narrowing_normal']
        left_df['neural_foraminal_narrowing_moderate'] = left_df['left_neural_foraminal_narrowing_moderate']
        left_df['neural_foraminal_narrowing_severe'] = left_df['left_neural_foraminal_narrowing_severe']
        left_df['study_level'] = left_df.study_id.astype(str) + '_' + left_df.level.str.replace('/', '_').str.lower()

        # === 建立 valid_df（保留所有資料） ===
        self.valid_df = pd.concat([right_df, left_df], ignore_index=True)

        # === 過濾 noisy 標記，建立 train_df ===
        noise_df = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_holdout_9/noisy_target_level_th08_holdout.csv')
        noise_df = noise_df[noise_df.target.isin([
            'right_neural_foraminal_narrowing',
            'left_neural_foraminal_narrowing'
        ])]
        noisy_study_levels = set(noise_df.study_level)
        self.train_df = self.valid_df[~self.valid_df.study_level.isin(noisy_study_levels)].reset_index(drop=True)

        # === 過濾 NaN 標籤樣本 ===
        l = len(self.train_df)
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
            self.valid_df = self.valid_df[~self.valid_df[col].isnull()]
        print(f'Before label NaN filtering: {l}, After: {len(self.train_df)}')



# rsna_axial_spinal_model.py
import math
import os
import copy
import time
from typing import Optional, Sequence, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import timm
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np


# -------------------------
# FocalLoss (Multiclass) with class weights
# -------------------------
class FocalLoss_axial_spinal(nn.Module):
    def __init__(self, alpha: Optional[Sequence[float]] = None,
                 gamma: float = 2.0, reduction: str = 'mean'):
        """
        alpha: None / float / list[float] (len=C). 類別權重，處理不平衡
        gamma: focusing parameter
        reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        if isinstance(alpha, (list, tuple, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([float(alpha)], dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, C]
        targets: [N] (long, e.g., 0..C-1)
        """
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)  # predicted prob of the true class

        if self.alpha is not None:
            if self.alpha.numel() == 1:
                at = self.alpha.to(logits.device)
            else:
                at = self.alpha.to(logits.device)[targets]
        else:
            at = 1.0

        loss = at * (1.0 - pt).pow(self.gamma) * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# -------------------------
# Metrics
# -------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average='macro')


def macro_auc_ovo(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    y_true: [N] int
    y_proba: [N, C] probabilities
    """
    # 若某些類別在 y_true 沒出現，roc_auc_score 可能會報錯，這裡保護一下
    labels = np.unique(y_true)
    if labels.size < 2:
        return float('nan')
    try:
        return roc_auc_score(y_true, y_proba, average='macro', multi_class='ovo')
    except Exception:
        return float('nan')


# -------------------------
# Baseline (簡化成訓練用基底)
# -------------------------
class Baseline_ResNet50V2_axial_spinal:
    def __init__(self):
        # 這些是常用設定，留著讓子類覆寫
        self.gpu = 'v100'
        self.batch_size = 16
        self.grad_accumulations = 1
        self.lr = 1e-4
        self.epochs = 20
        self.seed = 2023
        self.model_name = 'convnext_small.fb_in22k_ft_in1k_384'
        self.num_classes = 3
        self.fp16 = True
        self.optimizer_name = 'adamw'
        self.scheduler_name = 'CosineAnnealingWarmRestarts'
        self.eta_min = 5e-7
        self.t_max = 30
        self.use_wandb = True

        # 由子類建立：
        self.model: nn.Module = None
        self.criterion: nn.Module = None

        # 早停
        self.early_stop_patience = 5  # 連續 5 個 epoch val 沒進步就停
        self.early_stop_mode = 'max'  # 以 macro-F1 為準：越大越好
        self.best_ckpt_path = 'best_model.pth'

        # metric
        self.monitor_metric_name = 'macro_f1'  # or 'macro_auc'
        self.history: Dict[str, list] = {'train_loss': [], 'val_loss': [],
                                         'val_macro_f1': [], 'val_macro_auc': []}

    # ----- 組件 -----
    def build_model(self):
        self.model = timm.create_model(
            self.model_name, pretrained=True, num_classes=self.num_classes
        )

    def build_criterion(self, alpha=None, gamma=2.0):
        self.criterion = FocalLoss_axial_spinal(alpha=alpha, gamma=gamma, reduction='mean')

    def build_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr
        if self.optimizer_name.lower() == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr)
        elif self.optimizer_name.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer_name}')

    def build_scheduler(self, optimizer):
        if self.scheduler_name == 'CosineAnnealingWarmRestarts':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.t_max, eta_min=self.eta_min
            )
        elif self.scheduler_name == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.t_max, eta_min=self.eta_min
            )
        else:
            return None

    # ----- 訓練與驗證 -----
    @torch.no_grad()
    def _evaluate(self, loader, device) -> Dict[str, Any]:
        self.model.eval()
        loss_meter = 0.0
        n = 0

        all_probs = []
        all_preds = []
        all_tgts = []

        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            bs = images.size(0)
            loss_meter += loss.item() * bs
            n += bs

            all_probs.append(probs.detach().cpu())
            all_preds.append(preds.detach().cpu())
            all_tgts.append(targets.detach().cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_tgts = torch.cat(all_tgts).numpy()

        val_loss = loss_meter / max(1, n)
        val_f1 = macro_f1(all_tgts, all_preds)
        val_auc = macro_auc_ovo(all_tgts, all_probs)

        return {'val_loss': val_loss, 'val_macro_f1': val_f1, 'val_macro_auc': val_auc}

    def _improve(self, current: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        if self.early_stop_mode == 'max':
            return current > best
        else:
            return current < best

    def fit(self, train_loader, val_loader=None, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(device)
        optimizer = self.build_optimizer(self.lr)
        scheduler = self.build_scheduler(optimizer)
        scaler = GradScaler(enabled=self.fp16)

        best_metric = None
        best_state = None
        patience_left = self.early_stop_patience

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            start_t = time.time()

            running_loss = 0.0
            n = 0
            optimizer.zero_grad(set_to_none=True)

            for step, (images, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with autocast(enabled=self.fp16):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                    loss = loss / max(1, self.grad_accumulations)

                scaler.scale(loss).backward()

                if (step + 1) % self.grad_accumulations == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                bs = images.size(0)
                running_loss += loss.item() * bs * max(1, self.grad_accumulations)
                n += bs

            if scheduler is not None:
                # 使用 WarmRestarts 時，通常每個 epoch step 一次就好
                if hasattr(scheduler, 'step'):
                    scheduler.step(epoch - 1 + step / max(1, len(train_loader)))

            train_loss = running_loss / max(1, n)
            self.history['train_loss'].append(train_loss)

            # 驗證
            if val_loader is not None:
                eval_res = self._evaluate(val_loader, device)
                self.history['val_loss'].append(eval_res['val_loss'])
                self.history['val_macro_f1'].append(eval_res['val_macro_f1'])
                self.history['val_macro_auc'].append(eval_res['val_macro_auc'])

                # 監控指標
                monitor_value = eval_res['val_macro_f1'] if self.monitor_metric_name == 'macro_f1' else eval_res['val_macro_auc']

                improved = self._improve(monitor_value, best_metric)
                if improved:
                    best_metric = monitor_value
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_left = self.early_stop_patience
                    # 存檔（可選）
                    if self.best_ckpt_path:
                        torch.save({'model': best_state,
                                    'epoch': epoch,
                                    'monitor': monitor_value}, self.best_ckpt_path)
                else:
                    patience_left -= 1

                print(f"[Epoch {epoch:03d}/{self.epochs}] "
                      f"train_loss={train_loss:.4f} | "
                      f"val_loss={eval_res['val_loss']:.4f} | "
                      f"macroF1={eval_res['val_macro_f1']:.4f} | "
                      f"macroAUC={eval_res['val_macro_auc']:.4f} | "
                      f"best={best_metric if best_metric is not None else float('nan'):.4f} | "
                      f"patience_left={patience_left}")
            else:
                print(f"[Epoch {epoch:03d}/{self.epochs}] train_loss={train_loss:.4f}")

            # 早停
            if val_loader is not None and patience_left <= 0:
                print("Early stopping triggered.")
                break

            dur = time.time() - start_t
            # 你可以在這裡加上 wandb.log(...) 等

        # 載回最佳權重
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history


# -------------------------
# RSNA Axial Spinal Model
# -------------------------
class rsna_axial_spinal_ResNet50V2_axial_spinal(Baseline_ResNet50V2_axial_spinal):
    def __init__(self,
                 alpha: Optional[Sequence[float]] = None,
                 gamma: float = 2.0,
                 lr: float = 5.5e-5,
                 epochs: int = 20,
                 patience: int = 5,
                 model_name: str = 'convnext_small.fb_in22k_ft_in1k_384',
                 num_classes: int = 3,
                 monitor: str = 'macro_f1'):
        """
        alpha: 類別權重（建議依據樣本數倒數設定，如 [1.0, 3.0, 3.0]）
        gamma: focal loss gamma
        lr: 學習率
        epochs: 訓練回合數（你要求的 20）
        patience: 早停耐心（例如 5）
        monitor: 'macro_f1' or 'macro_auc'
        """
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.early_stop_patience = patience
        self.monitor_metric_name = monitor
        self.model_name = model_name
        self.num_classes = num_classes

        # 建模型 & loss
        self.build_model()
        self.build_criterion(alpha=alpha, gamma=gamma)

        # 其他建議設定
        self.grad_accumulations = 2 if lr <= 6e-5 else 1  # 小 batch 時稍微累積
        self.fp16 = True
        self.optimizer_name = 'adamw'
        self.scheduler_name = 'CosineAnnealingWarmRestarts'
        self.t_max = 30
        self.eta_min = 5e-7

    @staticmethod
    def suggest_alpha_from_counts(class_counts: Sequence[int]) -> Sequence[float]:
        """
        依據各類樣本數給出建議 alpha（倒數 + 正規化）
        e.g. counts=[n0,n1,n2] -> alpha \propto 1/count
        """
        counts = np.array(class_counts, dtype=float)
        counts[counts == 0] = counts[counts > 0].min()  # 避免除 0
        inv = 1.0 / counts
        alpha = inv / inv.mean()
        return alpha.tolist()


# 相容舊程式碼
# FocalLoss = FocalLoss_axial_spinal
# Baseline_ResNet50V2 = Baseline_ResNet50V2_axial_spinal
# rsna_axial_spinal_ResNet50V2 = rsna_axial_spinal_ResNet50V2_axial_spinal
