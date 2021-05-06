import torch


class MSE(torch.nn.MSELoss):
    def __init__(self, alice):
        super().__init__()
        self.alice = alice


class MSEBits(MSE):
    def __init__(self, alice, mu=1.):
        super().__init__(alice)
        self.loss_fn = lambda x, y: torch.mean(
            torch.sum(
                torch.nn.functional.mse_loss(x, y, reduction='none')
                + (mu / 2) * torch.square(x - torch.sign(x)),
                dim=-1
            )
        )

    def forward(self, input, target):
        return self.loss_fn(input, target)
