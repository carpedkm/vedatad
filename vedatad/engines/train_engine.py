from vedacore.misc import registry
from vedacore.optimizers import build_optimizer
from vedatad.criteria import build_criterion
from .base_engine import BaseEngine


@registry.register_module('engine')
class TrainEngine(BaseEngine):

    def __init__(self, model, criterion, optimizer):
        super().__init__(model) ## with given model -> make model (model is configuration file -> here: SingleStageDetector)
        self.criterion = build_criterion(criterion)
        self.optimizer = build_optimizer(self.model, optimizer)

    def extract_feats(self, img):
        feats = self.model(img, train=True)
        return feats

    def forward(self, data):
        print('TRAIN ENGINE EXEC - forward')
        x1 = self.forward_impl(**data)

        return x1

    def forward_impl(self,
                     imgs,
                     video_metas,
                     gt_segments,
                     gt_labels,
                     gt_segments_ignore=None):
        feats, x2 = self.extract_feats(imgs)
        losses = self.criterion.loss(feats, video_metas, gt_segments,
                                     gt_labels, gt_segments_ignore)

        print("x2 print: ", x2.shape)

        return losses
        # return losses, feats
