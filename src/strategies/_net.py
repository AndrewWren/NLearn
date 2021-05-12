"""
Note: do not make session, alice or bob an attribute of these nets -
otherwise will cause pickling problems
"""
import torch
import torch.nn as nn
import src.lib.max_tempered_layers


class FFs(torch.nn.Module):

    def __init__(self, agent, layers=10, width=50):
        super().__init__()
        self.input_width = agent.play.input_width
        self.output_width = agent.play.output_width
        self.layers = layers
        self.width = width
        ffs = [nn.Linear(self.input_width, self.width)]
        ffs += [nn.Linear(self.width, self.width) for _ in range(self.layers
                                                                 - 2)]
        ffs += [nn.Linear(self.width, self.output_width)]
        self.ffs = nn.ModuleList(ffs)

    def forward(self, x):
        for ff in self.ffs[: -1]:
            x = ff(x)
            nn.ReLU(inplace=True)(x)
        return self.ffs[-1](x)


class MaxNet(src.lib.max_tempered_layers.Net):
    def __init__(self, agent, focus, layers, width, beta=0.2,
                 bias_included=False):
        super().__init__(
            agent.play.input_width,
            agent.play.output_width,
            focus,
            layers,
            width,
            beta)

