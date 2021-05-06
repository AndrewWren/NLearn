import torch
from scr.ml_utilities import c, h, rng_c, to_array, \
    to_device_tensor, writer


class AlicePlay:
    def __init__(self, alice):
        self.alice = alice
        self.targets = self.alice.run.game_origin.targets


class Basic(AlicePlay):
    def __init__(self, alice):
        super().__init__(alice)
        self.input_width = alice.run.tuple_specs.n_elements * 2
        self.output_width = c.N_CODE

    def __call__(self):
        targets = torch.flatten(self.targets, start_dim=1)
        alice_outputs = self.alice.net(targets)
        return torch.sign(alice_outputs)


FromDecisions = Basic



