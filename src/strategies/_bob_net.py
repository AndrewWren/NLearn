#TODO Does this need to be separate from the corresponding alice file?
import src.nets


class FFs(src.nets.FFs):
    def __init__(self, bob, layers, width):
        super().__init__(
            input_width=bob.play.input_width,
            output_width=bob.play.output_width,
            layers=layers,
            width=width
        )
