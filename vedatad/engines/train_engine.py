from vedacore.misc import registry
from vedacore.optimizers import build_optimizer
from vedatad.criteria import build_criterion
from .base_engine import BaseEngine
import torch
import torch.nn.functional as f

@registry.register_module('engine')
class TrainEngine(BaseEngine):

    def __init__(self, model, criterion, optimizer):
        super().__init__(model) ## with given model -> make model (model is configuration file -> here: SingleStageDetector)
        self.criterion = build_criterion(criterion)
        self.optimizer = build_optimizer(self.model, optimizer)

    def extract_feats(self, img):
        # print('INTO SINGLE STAGE DETECTOR')
        feats = self.model(img, train=True) # GOING INTO forward of single stage detector
        # print('OUTOF SINGLE STAGE DETECTOR')
        return feats

    def forward(self, data):
        # print('TRAIN ENGINE EXEC - forward')
        data_origin = data[0]
        self.data_with_map = data[1]
        x1 = self.forward_impl(**data_origin)

        return x1

    def forward_impl(self,
                     imgs,
                     video_metas,
                     gt_segments,
                     gt_labels,
                     gt_segments_ignore=None): # FIXED

        COEFF=0.01
        # feats = self.extract_feats(imgs)
        feats, to_contrastive = self.extract_feats(imgs)
        # label
        # contrastive_feat = contrastive_processing(to_contrastive)
        losses = self.criterion.loss(feats, video_metas, gt_segments,
                                     gt_labels, gt_segments_ignore)
        tsm = torch.bmm(torch.transpose(to_contrastive, 1, 2), to_contrastive)
        tsm_normalized = (tsm - torch.mean(tsm, [1, 2])[:, None, None]) / torch.std(tsm, [1, 2])[:, None, None]
        # print("x2 print: ", to_contrastive.shape)
        contrastive_loss = torch.mean(tsm_normalized) - torch.mean(tsm)
        contrastive_loss = COEFF * contrastive_loss
        losses['loss'] = losses['loss'] - contrastive_loss
        return losses
        # return losses, feats
    
    # def contrastive_processing(self, feature_for_contrast):

