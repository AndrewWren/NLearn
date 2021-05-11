#TODO Does this need to be separate from the corresponding alice file?
import src.lib.max_tempered_layers
from src.lib.ml_utilities import c
import src.nets


class FFs(src.nets.FFs):
    def __init__(self, alice, layers, width):
        super().__init__(
            input_width=alice.play.input_width,
            output_width=alice.play.output_width,
            layers=layers,
            width=width
        )


class MaxNet(src.lib.max_tempered_layers.Net):
    def __init__(self, alice, focus, layers, width, beta=0.2):
        super().__init__(
            alice.play.input_width,
            alice.play.output_width,
            focus,
            layers,
            width,
            beta)
        self.alice = alice

