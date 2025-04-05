# Adapted from https://github.com/SCLBD/DeepfakeBench/blob/main/training/detectors/clip_detector.py

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig
import loralib as lora
import copy

logger = logging.getLogger(__name__)

seed = 1024
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@DETECTOR.register_module(module_name='clip')
class CLIPDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

        # ...
        # Model Architecture to be specified
        
    def build_backbone(self, config):
        _, backbone = get_clip_visual(model_name="") # Model name to be specified
        return backbone

        
    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        arr=[]
        for i in range(8):
            feat = self.backbone(data_dict['image'][:,i,:,:,:])['pooler_output']
            arr.append(feat)
        feat = torch.stack(arr, dim=1)
    
        # ...
        # Code to be added

        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        '''
        [16, 8, 3, 224, 224] -> video_mode is True, otherwise would be [16, 3, 224, 224]
        batch size 16
        each clip is 8 frames -> clip_size
        3 channels
        spatial dimensions 224 x 224 -> resolution
        '''
        features = self.features(data_dict)

        # ...
        # Forward Pass Code to be added

        pred = self.classifier(features) #[16, 2]
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict


def get_clip_visual(model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    
    # ...
    # Code to be added

    return processor, model.vision_model
