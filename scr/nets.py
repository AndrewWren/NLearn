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
        self.tuple_specs = tuple_specs
        self.alice = FFs(input_width=tuple_specs.n_elements,
                         output_width=c.N_CODE).to(c.DEVICE)
        self.alice_optimizer = self.optimizer('ALICE')
        self.alice_loss_function = self.loss_function('ALICE')
        self.bob = FFs(input_width=h.N_SELECT * tuple_specs.n_elements +
                                   c.N_CODE,
                       output_width=h.N_SELECT).to(c.DEVICE)
        if h.BOB_OPTIMIZER == 'Same':
            bob_optimizer_label = 'ALICE'
        else:
            bob_optimizer_label = 'BOB'
        self.bob_optimizer = self.optimizer(bob_optimizer_label)
        if h.BOB_LOSS_FUNCTION == 'Same':
            bob_loss_function_label = 'ALICE'
        else:
            bob_loss_function_label = 'BOB'
        self.bob_loss_function = self.loss_function(bob_loss_function_label)
        self.selections = tuple_specs.selections
        # print(f'{self.selections=}')
        self.epsilon_slope = 1 / (h.EPSILON_ZERO - h.EPSILON_FLAT_END)

    def optimizer(self, net_label):
        values = eval('h.' + net_label + '_OPTIMIZER')
        return eval(
            'torch.optim.' + values[0]
            + '(self.alice.parameters(), *' + str(values[1]) + ')'
        )

    def loss_function(self, net_label):
        values = eval('h.' + net_label + '_LOSS_FUNCTION')
        return eval(
            'torch.nn.' + values[0]
            + 'Loss(*' + str(values[1]) + ')'
        )

    def play(self, game_origins: GameOrigins):
        """
        """
        """Forward passes
        """
        selections = to_device_tensor(game_origins.selections)
        targets = game_origins.selections[np.arange(h.BATCHSIZE),
                                          game_origins.target_nos]
        targets = to_device_tensor(targets)
        alice_outputs = self.alice(targets)
        alice_qs = torch.sum(torch.abs(alice_outputs), dim=1)
        codes = torch.sign(alice_outputs)
        """print(f'{game_origins.selections.shape=}')
        print(f'{targets.shape=}')
        print(f'{codes.shape=}')
        """
        bob_input = torch.cat([
            selections.reshape((
                h.BATCHSIZE,
                h.N_SELECT * self.tuple_specs.n_elements)),
            codes], 1
        )
        # print(f'{bob_input.shape=}')
        bob_q_estimates = self.bob(bob_input)
        bob_q_estimates_max = torch.argmax(bob_q_estimates, dim=1).long()
        # TODO What about the Warning at
        # https://pytorch.org/docs/stable/generated/torch.max.html?highlight
        # =max#torch.max ?  and see also torch.amax

        # GOT TO HERE

        decision_nos = self.eps_greedy(game_origins.iteration,
                                       bob_q_estimates_max)
        decisions = selections[list(range(h.BATCHSIZE)), decision_nos]
        decision_qs = bob_q_estimates[list(range(h.BATCHSIZE)), decision_nos]
        rewards = self.tuple_specs.rewards(grounds=targets, guesses=decisions)
        print(f'{targets.size()=}')
        print(f'{decision_qs.size()=}')
        print(f'{rewards.size()=}')
        game_reports = GameReports(game_origins, decision_qs, rewards)
        """Backward passes
        """
        alice_losses = self.alice_loss_function(alice_qs, rewards)
        self.alice_optimizer.zero_grad()
        alice_losses.backward()
        self.alice_optimizer.step()
        bob_losses = self.bob_loss_function(decision_qs, rewards)
        self.bob_optimizer.zero_grad()
        bob_losses.backward()
        self.bob_optimizer.step()
        return game_reports

    def eps_greedy(self, iteration, greedy_indices):
        """

        :param iteration: int
        :param greedy: torch.float32, size (h.BATCHSIZE, h.N_SELECT)
        :return: torch.int64, size (h.BATCHSIZE)
        """
        if iteration >= h.EPSILON_ZERO:
            return greedy_indices
        epsilon = self.epsilon_function(iteration)
        indicator = torch.empty(h.BATCHSIZE)
        indicator.uniform_(generator=h.t_rng)
        chooser = (indicator > epsilon).long()
        # print(f'{chooser=}')
        random_indices = torch.empty(h.BATCHSIZE)
        random_indices.random_(h.N_SELECT, generator=h.t_rng).long()
        for_choice = torch.dstack((random_indices, greedy_indices))[0]
        """print(f'{for_choice=}')
        print(f'{chooser.shape=}')
        print(f'{for_choice.shape=}')
        """
        return for_choice[list(range(h.BATCHSIZE)), chooser].long()

    """def train(self, batch: list[GameReports]):
        training_return = exec('self.train_with_' + c.TRAINING_METHOD + '('
                                                                        'batch)')
        return training_return

    def train_with_q(self, batch: list[GameReports]):
        batch = batch.to(c.DEVICE)
    """

    def epsilon_function(self, iteration):
        """

        :param iteration: int
        :return: torch.float32, size = (h.BATCHSIZE)
        """
        single_epsilon = torch.FloatTensor([
            1.
            - max(iteration - h.EPSILON_FLAT_END, 0) * self.epsilon_slope
        ])
        return single_epsilon.repeat(h.BATCHSIZE)
