import torch
from src.lib.torch_bin_dec import dec_2_bin, initialise_bin_dec
from src.lib.ml_utilities import c, h


class AlicePlay:
    def __init__(self, alice):
        self.alice = alice

    def __call__(self):
        self.targets = torch.flatten(self.alice.session.targets_t, start_dim=1)


class Basic(AlicePlay):
    def __init__(self, alice):
        super().__init__(alice)
        self.input_width = alice.session.tuple_specs.n_elements * 2
        self.output_width = h.N_CODE

    def __call__(self):
        super().__call__()
        alice_outputs = self.alice.net(self.targets)
        return torch.sign(alice_outputs)


class QPerCode(AlicePlay):
    def __init__(self, alice):
        super().__init__(alice)
        self.input_width = alice.session.tuple_specs.n_elements * 2
        self.output_width = 2 ** h.N_CODE
        initialise_bin_dec()

    def __call__(self):
        super().__call__()
        alice_outputs = self.alice.net(self.targets)
        return dec_2_bin(torch.argmax(alice_outputs, dim=-1))
