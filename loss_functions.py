import torch


class MSEBitsLoss(torch.nn.MSELoss):
    def __init__(self, mu=1.):
        super().__init__()
        self.loss_fn = lambda x, y: torch.nn.MSELoss()(x, y) \
                                   + (mu / 2) * torch.mean(
            torch.square(x - torch.sign(x)))

    def forward(self, input, target):
        return self.loss_fn(input, target)
