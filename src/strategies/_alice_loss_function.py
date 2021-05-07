import torch
from src.ml_utilities import c

class MSE(torch.nn.MSELoss):
    def __init__(self, alice):
        super().__init__()
        self.alice = alice

    def __call__(self, predictions, targets, dummy):
        return self.forward(predictions, targets)


class MSEBits(MSE):
    def __init__(self, alice, mu=1.):
        super().__init__(alice)
        self.loss_fn = lambda x, y, z=0: torch.mean(

                torch.nn.functional.mse_loss(
                    x.unsqueeze(-1),
                    y.unsqueeze(-1),
                    reduction='none'
                )
                + (mu / (2 * c.N_CODE))
                * torch.nn.functional.mse_loss(z, torch.sign(z),
                                               reduction='none')
            )

    def __call__(self, predictions, targets, outputs):
        return self.loss_fn(predictions, targets, outputs)


class Huber(torch.nn.SmoothL1Loss):
    def __init__(self, alice, beta=1.):
        super().__init__(beta)
        self.alice = alice


class HuberBits(Huber):
    def __init__(self, alice, beta=1., beta_bits=1., mu=1.):
        super().__init__(alice, beta)
        self.loss_fn = lambda x, y: torch.mean(
            torch.sum(
                torch.nn.functional.smooth_l1_loss(x, y, reduction='none',
                                                   beta=beta)
                + (mu / 2) * torch.nn.functional.smooth_l1_loss(
                    x, torch.sign(x), reduction='none', beta=beta_bits),
                dim=-1
            )
        )

    def forward(self, input, target):
        return self.loss_fn(input, target)
