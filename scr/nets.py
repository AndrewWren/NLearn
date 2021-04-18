import random
import torch
import torch.nn as nn
import scr.ml_utilities as mlu
from scr.ml_utilities import c, h
from scr.net_class import Net
from scr.tuple_and_code import Code, Domain, ElementSpec, GameOrigin, \
    GameReport, TupleSpec


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


def argmax(ls):
    # from https://towardsdatascience.com/there-is-no-argmax-function-for-python-list-cd0659b05e49
    f = lambda i: ls[i]
    return max(range(len(ls)), key=f)


class Nets:
    def __init__(self, tuple_spec: TupleSpec):
        self.tuple_spec = tuple_spec
        self.net_a = FFs(input_width=tuple_spec.n_elements,
                         output_width=c.N_CODE)
        self.net_ri = FFs(input_width=tuple_spec.n_elements,
                          output_width=c.N_CODE)
        self.net_rd = FFs(input_width=tuple_spec.select * c.N_CODE,
                          output_width=tuple_spec.select)

    def play(self, batch: list[GameOrigin]) -> list[GameReport]:
        targets = [game_origin.selection[game_origin.target_no]
                   for game_origin in batch]
        targets = torch.FloatTensor(targets).to(c.DEVICE)
        signals = self.net_a(targets)  #TODO Need to factor in signals
        interpretations = torch.stack(
            [torch.cat([self.net_ri(element.to(c.DEVICE))
            for element in game_origin.selection])
            for game_origin in batch]
        )  #TODO Could this be made faster by bigger batching?
        estimates = self.net_rd(interpretations).cpu().numpy()
        decisions = [eps_greedy(argmax(estimate)) for estimate in estimates]
        scores = [self.tuple_spec.score(ground=target, guess=decision)
                  for target, decision in zip(targets, decisions)]
        game_reports = [GameReport(game_origin, decision, score)
        for game_origin, decision, score in zip(batch, decisions, scores)]
        return game_reports

    def eps_greedy(self, greedy):
        indicator = random.random()
        epsilon = epsilon_function(iteration)  #TODO Where should epsilon come from?
        if indicator <= epsilon:
            return random.randrange(self.select)
        return greedy

    def train(self, batch: list[GameReport]):
        training_return = exec('self.train_with_' + c.TRAINING_METHOD + '('
                                                                'batch)')
        return training_return

    def train_with_q(self, batch: list[GameReport]):
        batch = batch.to(c.DEVICE)

