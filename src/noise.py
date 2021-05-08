import torch
from src.lib.ml_utilities import c, h


class Noise:
    """

    :param code: torch.float32
    :return:
    """
    def __init__(self, prob):
        self.prob = prob

    def inject(self, codes):
        """

        :param codes: torch.float32, of size (net.size0, N_CODE) in practice
        but not forced to be
        :return: codes after Bernoulli noise applied
        """
        noise = torch.ones(codes.size()).to(c.DEVICE) * self.prob
        # as bernoulli generates 1s with prob of the argument:
        noise =  1 - 2 * torch.bernoulli(noise, generator=h.te_rng)
        return torch.mul(codes, noise)
