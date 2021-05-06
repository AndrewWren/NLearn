import torch
from scr.ml_utilities import c, h, rng_c, to_array, \
    to_device_tensor, writer
from scr.nets import FFs


class AliceNet:
    def __init__(self, alice):
        self.alice = alice


class FFNet(AliceNet):
    def __init__(self, alice, layers, width):
        super().__init__(alice)
        self.net = FFs(
            input_width=self.alice.play.input_width,
            output_width=self.alice.play.output_width,
            layers=layers,
            width=width
        )
