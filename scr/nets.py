import random
import numpy as np
import torch
import torch.nn as nn
import scr.ml_utilities as mlu
from scr.ml_utilities import c, h, rng_c
from scr.net_class import Net
from scr.tuple_and_code import Code, Domain, ElementSpec, GameOrigins, \
    GameReports, TupleSpecs


def to_device_tensor(x):
    """
    Convert array to device tensor
    :param x: numpy array
    :return:  pytorch c.DEVICE tensor
    """
    return torch.FloatTensor(x).to(c.DEVICE)


class FFs(Net):

    def __init__(self, input_width, output_width, layers=10, width=50):
        super().__init__()
        self.layers = layers
        self.width = width
        self.input_width = input_width
        self.output_width = output_width
        self._build()

    def _build(self):
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


"""def argmax(ls):
    # from https://towardsdatascience.com/there-is-no-argmax-function-for-python-list-cd0659b05e49
    f = lambda i: ls[i]
    return max(range(len(ls)), key=f)
"""


class Nets:
    def __init__(self, tuple_specs: TupleSpecs):
        self.tuple_spec = tuple_specs
        self.alice = FFs(input_width=tuple_specs.n_elements,
                         output_width=c.N_CODE).to(c.DEVICE)
        self.bob = FFs(input_width=tuple_specs.n_elements + c.N_CODE,
                       output_width=h.N_SELECT).to(c.DEVICE)
        self.selections = tuple_specs.selections
        self.epsilon_slope = 1 / (h.EPSILON_ZERO - h.EPSILON_FLAT_END)

    def play(self, game_origins: GameOrigins):
        targets = game_origins.selections[np.arange(h.BATCHSIZE),
                                          game_origins.target_nos]
        targets = to_device_tensor(targets)
        codes = self.alice(targets)
        print(f'{game_origins.selections.shape=}')
        bob_input = np.concatenate([
            game_origins.selections.reshape((
                h.BATCHSIZE, h.N_SELECT * tuple_specs.n_elements)),
            np.repeat(codes, repeats=self.selections)
        ], axis=1
        )
        bob_q_estimates = self.bob(to_device_tensor(bob_input))
        bob_q_estimates_max = torch.max(bob_q_estimates, dim=1).indices
        # TODO What about the Warning at
        # https://pytorch.org/docs/stable/generated/torch.max.html?highlight
        # =max#torch.max ?

        #GOT TO HERE

        decisions = [self.eps_greedy(game_origins.iteration, argmax(estimate))
                     for estimate in bob_q_estimates_max]
        scores = [self.tuple_spec.score(ground=target, guess=decision)
                  for target, decision in zip(targets, decisions)]
        game_reports = [GameReports(game_origin, decision, score)
                        for game_origin, decision, score in
                        zip(batch, decisions, scores)]
        return game_reports

    def eps_greedy(self, iteration: int, greedy: int) -> int:
        indicator = h.rng.random()
        epsilon = self.epsilon_function(iteration)
        if indicator < epsilon:
            return h.rng.integers(h.N_SELECT)
        return greedy

    def train(self, batch: list[GameReports]):
        training_return = exec('self.train_with_' + c.TRAINING_METHOD + '('
                                                                        'batch)')
        return training_return

    def train_with_q(self, batch: list[GameReports]):
        batch = batch.to(c.DEVICE)

    def epsilon_function(self, iteration: int) -> float:
        if iteration >= h.EPSILON_ZERO:
            return 0.
        return 1. - max(iteration - h.EPSILON_FLAT_END, 0) * self.epsilon_slope
