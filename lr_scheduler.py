import warnings

from torch.optim.lr_scheduler import ReduceLROnPlateau

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class VerboseToLogReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False, log_rec=None):
        super(VerboseToLogReduceLROnPlateau, self).__init__(optimizer, mode=mode, factor=factor, patience=patience,
                                                            threshold=threshold, threshold_mode=threshold_mode,
                                                            cooldown=cooldown, min_lr=min_lr, eps=eps, verbose=verbose)
        self.log_rec = log_rec

    # 加入 log 功能
    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    verbose_str = 'Epoch {}: reducing learning rate of group {} to {:.4e}.'.format(epoch_str, i, new_lr)
                    self.log_rec.write(verbose_str)
                    print(verbose_str)
