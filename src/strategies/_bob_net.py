"""
Note: do not make session, alice or bob an attribute of these nets -
otherwise will cause pickling problems
"""
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
