import torch
from src.lib.torch_bin_dec import bin_2_dec, dec_2_bin
from src.ml_utilities import c


class AlicePlay:
    def __init__(self, alice):
        self.alice = alice

    def __call__(self):
        self.targets = torch.flatten(self.alice.session.targets_t, start_dim=1)


class Basic(AlicePlay):
    def __init__(self, alice):
        super().__init__(alice)
        self.input_width = alice.session.tuple_specs.n_elements * 2
        self.output_width = c.N_CODE

    def __call__(self):
        super().__call__()
        alice_outputs = self.alice.net(self.targets)
        return torch.sign(alice_outputs)


class QPerCode(AlicePlay):
    def __init__(self, alice):
        super().__init__(alice)
        self.input_width = alice.session.tuple_specs.n_elements * 2
        self.output_width = 2 ** c.N_CODE

    def __call__(self):
        super().__call__()
        alice_outputs = self.alice.net(self.targets)
        return dec_2_bin(torch.argmax(alice_outputs, dim=-1))
