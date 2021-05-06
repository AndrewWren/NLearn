import torch
from scr.ml_utilities import c, h, rng_c, to_array, \
    to_device_tensor, writer


class AlicePlay:
    def __init__(self, alice):
        self.alice = alice
        self.targets = self.alice.session.targets_t

    def __call__(self):
        self.targets = torch.flatten(self.alice.session.targets_t, start_dim=1)


class Basic(AlicePlay):
    def __init__(self, alice):
        super().__init__(alice)
        self.input_width = alice.session.tuple_specs.n_elements * 2
        self.output_width = c.N_CODE
        self.alice = alice

    def __call__(self):
        super().__call__()
        alice_outputs = self.alice.net(self.targets)
        return torch.sign(alice_outputs)


FromDecisions = Basic



