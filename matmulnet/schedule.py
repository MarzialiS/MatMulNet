"""Schedules: the learning rate (halved on plateaus of the validation loss) and epsilon."""

EPS_START, EPS_END = 1e-1, 1e-3
LR_START, LR_MIN, LR_FACTOR, LR_PATIENCE = 1e-4, 1e-6, 0.5, 15


def epsilon_at(epoch, max_epochs, eps_start=EPS_START, eps_end=EPS_END):
    """Exponential decay from eps_start (epoch 1) towards eps_end over max_epochs epochs."""
    return eps_start * (eps_end / eps_start) ** ((epoch - 1) / max_epochs)


class Plateau:
    """Same rule as torch.optim.lr_scheduler.ReduceLROnPlateau(mode="min", factor, patience,
    min_lr) with its default relative threshold 1e-4: the learning rate is multiplied by
    `factor` once the validation loss has failed to improve for more than `patience` epochs."""

    threshold, tol = 1e-4, 1e-8

    def __init__(self, lr=LR_START, factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN,
                 best=float("inf"), bad=0):
        self.lr, self.factor, self.patience, self.min_lr = lr, factor, patience, min_lr
        self.best, self.bad = best, bad

    def step(self, val_loss):
        if val_loss < self.best * (1.0 - self.threshold):
            self.best, self.bad = val_loss, 0
        else:
            self.bad += 1
        if self.bad > self.patience:
            new = max(self.lr * self.factor, self.min_lr)
            if self.lr - new > self.tol:
                self.lr = new
            self.bad = 0
        return self.lr
