import torch
import torch.nn as nn

from vedacore.misc import registry
from ..builder import build_backbone, build_head, build_neck
from .base_detector import BaseDetector
import numpy as np

@registry.register_module('detector')
class SingleStageDetector(BaseDetector):

    def __init__(self, backbone, head, neck=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        if neck:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        self.head = build_head(head)

        self.init_weights()
        ## Customizing the code
        # self.encoding = nn.Sequential(
        #     nn.Conv1d()
        # )

    def init_weights(self):
        self.backbone.init_weights()
        if self.neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

        self.head.init_weights()

    def forward_impl(self, x):
        feats = self.backbone(x)
        if self.neck:
            pass_in_to_head = self.neck(feats) # FIXED # passed in to head would get tuple
        # print('>>>>> SINGLESTAGE_DETECTOR : ', type(pass_in_to_head))
        to_contrastive = pass_in_to_head[1]
        if isinstance(pass_in_to_head, list):
            # print('pass in to head', pass_in_to_head[0][0].shape, pass_in_to_head[0][1].shape, pass_in_to_head[0][2].shape, pass_in_to_head[0][3].shape, pass_in_to_head[0][4].shape)
            feats = self.head(pass_in_to_head[0])
        return feats, to_contrastive

    def forward(self, x, train=True):
        if train:
            self.train()
            feats = self.forward_impl(x)
        else:
            self.eval()
            with torch.no_grad():
                feats = self.forward_impl(x)
        return feats

    # def forward_smiliarity(self, x):
        
