import torch
from scr.ml_utilities import c, h, to_array, \
    to_device_tensor, writer
import scr.nets


class FFs(scr.nets.FFs):
    def __init__(self, alice, layers, width):
        super().__init__(
            input_width=alice.play.input_width,
            output_width=alice.play.output_width,
            layers=layers,
            width=width
        )
