import random
import torch
import torch.nn as nn
import scr.ml_utilities as mlu
from scr.ml_utilities import c, h
from scr.net_class import Net
from scr.tuple_and_code import Code, Domain, ElementSpec, TupleSpec


class NetA(Net):

    def __init__(self):
        super().__init__()

    def _build(self):
        ffs = [nn.Linear(self.n_tuples, c.N_CODE)]
        ffs += [nn.Linear(c.N_CODE, c.N_CODE)]
        self.ffs = nn.ModuleList(ffs)

    def forward(self, x):
        for ff in self.ffs[: -1]:
            x = ff(x)
            nn.ReLU(inplace=True)(x)
        return self.ffs[-1](x)

class NetR(NetA):
    pass
