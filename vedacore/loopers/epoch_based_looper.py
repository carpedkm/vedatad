from .base_looper import BaseLooper
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class EpochBasedLooper(BaseLooper):

    def __init__(self, modes, dataloaders, engines, hook_pool, logger,
                 workdir):
        super().__init__(modes, dataloaders, engines, hook_pool, logger,
                         workdir) 
# INIT : cfg.modes, dataloaders, engines, hook_pool, logger, cfg.workdir
    def epoch_loop(self, mode):
        self.mode = mode
        dataloader = self.dataloaders[mode]
        engine = self.engines[mode]
        iter_cnt = 0
        loss_sum = 0
        for idx, data in enumerate(dataloader):
            # print('data input to engine')
            self.hook_pool.fire(f'before_{mode}_iter', self)
            result = engine(data)
            loss_sum += result['loss'] # for tensor board -> get average loss
            iter_cnt += 1
            self.cur_results[mode] = result
            if mode == BaseLooper.TRAIN:
                self._iter += 1
            self._inner_iter = idx + 1
            self.hook_pool.fire(f'after_{mode}_iter', self)
        return loss_sum / iter_cnt # return it

    def start(self, max_epochs): # max_epochs = cfg.max_epochs
        self.hook_pool.fire('before_run', self)
        while self.epoch < max_epochs:
            for mode in self.modes:
                mode = mode.lower()
                self.hook_pool.fire(f'before_{mode}_epoch', self)
                avg_loss = self.epoch_loop(mode)
                if mode == BaseLooper.TRAIN:
                    self._epoch += 1
                    writer.add_scalar("Loss/train", avg_loss, self.epoch)
                self.hook_pool.fire(f'after_{mode}_epoch', self)
            if len(self.modes) == 1 and self.modes[0] == EpochBasedLooper.VAL:
                break
        writer.flush()
