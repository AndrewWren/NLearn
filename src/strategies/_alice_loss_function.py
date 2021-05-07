import torch
from src.ml_utilities import c

class MSE(torch.nn.MSELoss):
    def __init__(self, alice):
        super().__init__()
        self.alice = alice

    def __call__(self, predictions, targets, *args, **kwargs):
        return self.forward(predictions, targets)


class MSEBits(MSE):
    def __init__(self, alice, mu=1.):
        super().__init__(alice)
        self.loss_fn = (
            lambda x, y, z:
            torch.nn.functional.mse_loss(x, y)
            + (mu / 2)
            * torch.nn.functional.mse_loss(z, torch.sign(z))
        )  # for both the mse_loss functions this implies reduction='mean'

    def __call__(self, predictions, targets, outputs=0, *args, **kwargs):
        return self.loss_fn(predictions, targets, outputs)


class MSEAccidental(MSE):
    def __init__(self, alice, mu=1.):
        super().__init__(alice)
        self.loss_fn = lambda x, y: torch.sum(
                torch.nn.functional.mse_loss(x, y, reduction='none')
                + (mu / 2) * torch.square(x - torch.sign(x)),
            )

    def __call__(self, predictions, targets, *args, **kwargs):
        return self.loss_fn(predictions, targets)


class Huber(torch.nn.SmoothL1Loss):
    def __init__(self, alice, beta=1.):
        super().__init__(beta)
        self.alice = alice

    def __call__(self, predictions, targets, *args, **kwargs):
        return self.forward(predictions, targets)


class HuberBits(Huber):
    def __init__(self, alice, beta=1., beta_bits=1., mu=1.):
        super().__init__(alice, beta)
        self.loss_fn = (
            lambda x, y, z:
            torch.nn.functional.smooth_l1_loss(x, y, beta=beta)
            + (mu / 2)
            * torch.nn.functional.smooth_l1_loss(z, torch.sign(z),
                                                 beta=beta_bits)
        )  # for both the mse_loss functions this implies reduction='mean'

        def __call__(self, predictions, targets, outputs=0, *args, **kwargs):
            return self.loss_fn(predictions, targets, outputs)
