import torch
from src.lib.torch_bin_dec import dec_2_bin, initialise_bin_dec
from src.lib.ml_utilities import c, h


class AlicePlay:
    def __init__(self, alice):
        self.alice = alice

    def __call__(self):
        self.targets = self.alice.session.session_spec.spec.circle(
            self.alice.session.targets_t)


class QPerCode(AlicePlay):
    def __init__(self, alice):
        super().__init__(alice)
        self.input_width = 2
        self.output_width = 2 ** h.N_CODE
        initialise_bin_dec()

    def __call__(self):
        super().__call__()
        alice_outputs = self.alice.net(self.targets)
        return dec_2_bin(torch.argmax(alice_outputs, dim=-1))
