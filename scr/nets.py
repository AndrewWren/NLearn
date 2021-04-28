from collections import namedtuple
import random
import numpy as np
import torch
import torch.nn as nn
import scr.ml_utilities as mlu
from scr.ml_utilities import c, h, rng_c, writer
from scr.net_class import Net
from scr.tuple_and_code import Domain, ElementSpec, GameOrigins, \
    GameReports, NiceCode, ReplayBuffer, TupleSpecs


def to_device_tensor(x):
    """
    Convert array to device tensor
    :param x: numpy array
    :return:  pytorch c.DEVICE tensor
    """
    return torch.FloatTensor(x).to(c.DEVICE)


def to_array(x):
    """
    Convert device tensor to array
    :param x: pytorch c.DEVICE tensor
    :return: numpy array
    """
    return x.cpu().detach().numpy()


LossInfo = namedtuple('LossInfo', 'bob_loss iteration alice_loss')


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


class Nets:
    def __init__(self, tuple_specs: TupleSpecs):
        self.tuple_specs = tuple_specs
        self.set_output_widths()
        self.alice = FFs(
            input_width=tuple_specs.n_elements,
            output_width=self.alice_output_width,
            layers=h.ALICE_LAYERS,
            width=h.ALICE_WIDTH
        ).to(c.DEVICE)
        self.alice_optimizer = self.optimizer('ALICE', 'alice')
        self.alice_loss_function = self.loss_function('ALICE')
        self.bob = FFs(
            input_width=h.N_SELECT * tuple_specs.n_elements +
                                   c.N_CODE,
            output_width=self.bob_output_width,
            layers=h.BOB_LAYERS,
            width=h.BOB_WIDTH
        ).to(c.DEVICE)
        if h.BOB_OPTIMIZER == 'Same':
            bob_optimizer_label = 'ALICE'
        else:
            bob_optimizer_label = 'BOB'
        self.bob_optimizer = self.optimizer(bob_optimizer_label, 'bob')
        if h.BOB_LOSS_FUNCTION == 'Same':
            bob_loss_function_label = 'ALICE'
        else:
            bob_loss_function_label = 'BOB'
        self.bob_loss_function = self.loss_function(bob_loss_function_label)
        self.selections = tuple_specs.selections
        self.epsilon_slope = (1 - h.EPSILON_MIN) / (
                h.EPSILON_MIN_POINT - h.EPSILON_ONE_END)
        self.size0 = None

    def optimizer(self, h_label, parameter_label):
        values = eval('h.' + h_label.upper() + '_OPTIMIZER')
        return eval(
            'torch.optim.' + values[0]
            + '(self.' + parameter_label.lower() + '.parameters(), **' + str(
                values[1]) + ')'
        )

    def loss_function(self, h_label):
        values = eval('h.' + h_label.upper() + '_LOSS_FUNCTION')
        return eval(
            'torch.nn.' + values[0]
            + 'Loss(**' + str(values[1]) + ')'
        )

    @torch.no_grad()
    def play(self, game_origins: GameOrigins) -> GameReports:
        """
        Forward passes
        """
        self.set_size0(h.GAMESIZE)
        self.epsilon = self.epsilon_function(game_origins.iteration)
        targets = game_origins.selections[np.arange(self.size0),
                                          game_origins.target_nos]
        targets = to_device_tensor(targets)
        greedy_codes = self.alice_play(targets)
        codes = self.alice_eps_greedy(greedy_codes)
        selections = to_device_tensor(game_origins.selections)
        bob_q_estimates_argmax = self.bob_play(selections, codes)
        decision_nos = self.bob_eps_greedy(bob_q_estimates_argmax)
        decisions = self.gatherer(selections, decision_nos, 'Decisions')
        rewards = self.tuple_specs.rewards(grounds=targets, guesses=decisions)
        return GameReports(game_origins, to_array(codes),
                           to_array(decision_nos), to_array(rewards))
        # Returns iteration target_nos selections decisions rewards
        # Don't return alice_qs decision_qs

    def train(self, current_iteration, buffer):
        """
        Calculating Q values on the current alice and bob nets.  Training
        through backward pass
        :param current_iteration: int
        :param buffer: ReplayBuffer
        :return (alice_loss.item(), bob_loss.item()): (float, float)
        """
        game_reports = buffer.sample()
        self.size0 = h.BATCHSIZE
        self.epsilon = self.epsilon_function(current_iteration)

        # Alice
        targets = game_reports.selections[np.arange(self.size0),
                                          game_reports.target_nos]
        alice_loss = self.alice_train(
            to_device_tensor(targets),
            to_device_tensor(game_reports.rewards),
            to_device_tensor(game_reports.codes)
        )
        self.alice_optimizer.zero_grad()
        alice_loss.backward()
        self.alice_optimizer.step()

        # Bob
        selections = to_device_tensor(game_reports.selections)
        codes = to_device_tensor(game_reports.codes)
        decision_nos = to_device_tensor(game_reports.decision_nos).long()
        rewards = to_device_tensor(game_reports.rewards)
        bob_loss = self.bob_train(selections, codes, decision_nos, rewards)
        self.bob_optimizer.zero_grad()
        bob_loss.backward()
        self.bob_optimizer.step()

        #logging of various sorts
        writer.add_scalars(
            'Sqrt losses',
            {f'Alice sqrt loss_{h.hp_run}': torch.sqrt(alice_loss),
             f'Bob sqrt loss_{h.hp_run}': torch.sqrt(bob_loss)
             },
            global_step=current_iteration
        )
        if (current_iteration == 0) or (current_iteration % 10000 == 0):
            mlu.log('Codes=')
            [mlu.log(NiceCode(code)) for code in codes]

        return alice_loss.item(), bob_loss.item()

    def alice_eps_greedy(self, greedy_codes):
        """

        :param greedy_codes: torch.float32, size (h.GAMESIZE or h.BATCHSIZE,
        c.N_CODE)
        :return: torch.int64, size (h.GAMESIZE or h.BATCHSIZE respectively,
        c.N_CODE)
        """
        indicator = torch.empty(self.size0).to(c.DEVICE)
        indicator.uniform_()
        chooser = (indicator >= self.epsilon).long()
        random_codes = torch.empty(self.size0, c.N_CODE).to(
            c.DEVICE)
        random_codes.random_(to=2).long()
        random_codes = 2 * random_codes - 1
        for_choice = torch.stack((random_codes, greedy_codes), dim=1)
        temp = self.gatherer(for_choice, chooser, 'Alice').long()
        return temp

    def bob_eps_greedy(self, greedy_indices):
        """

        :param iteration: int
        :param greedy_indices: torch.float32, size (h.GAMESIZE or h.BATCHSIZE)
        :return: torch.int64, size (h.GAMESIZE or h.BATCHSIZE respectively)
        """
        indicator = torch.empty(self.size0).to(c.DEVICE)
        indicator.uniform_()
        chooser = (indicator >= self.epsilon).long()
        random_indices = torch.empty(self.size0).to(c.DEVICE)
        random_indices.random_(to=h.N_SELECT).long()
        for_choice = torch.dstack((random_indices, greedy_indices))[0]
        return for_choice[list(range(self.size0)), chooser].long()

    def epsilon_function(self, iteration):
        """

        :param iteration: int
        :return: torch.float32, size = (h.BATCHSIZE)
        """
        if iteration >= h.EPSILON_MIN_POINT:
            single_epsilon = h.EPSILON_MIN * torch.ones(1).to(c.DEVICE)
        else:
            single_epsilon = torch.FloatTensor([
                1.
                - max(iteration - h.EPSILON_ONE_END, 0) * self.epsilon_slope
            ]).to(c.DEVICE)
        return single_epsilon.repeat(self.size0)

    def gatherer(self, input, indices, context):
        if (context == 'Alice') or (context == 'Decisions'):
            indices = indices.unsqueeze(1).repeat(1, input.size()[2]
                                                  ).unsqueeze(1)
        elif context == 'Decision_Qs':
            indices = indices.unsqueeze(1)
        else:
            exit(f'Invalid {context=}')
        gathered = torch.gather(input, 1, indices).squeeze()
        if self.size0 == 1:
            return gathered.unsqueeze(0)
        return gathered

    def set_size0(self, size0: int):
        self.size0 = size0

    def set_output_widths(self):
        if h.ALICE_STRATEGY == 'one_per_bit':
            self.alice_output_width = c.N_CODE
        elif h.ALICE_STRATEGY == 'one_per_code':
            self.alice_output_width = 2 ** c.N_CODE
        if h.BOB_STRATEGY == 'one_per_bit':
            self.bob_output_width = h.N_SELECT

    def alice_play(self, targets):
        """

        :param targets:
        :return greedy_codes: tensor, size=(h.GAMESIZE, c.N_CODE)
        """
        return eval(f'self.alice_play_{h.ALICE_STRATEGY}(targets)')

    def alice_play_one_per_bit(self, targets):
        alice_outputs = self.alice(targets)
        return torch.sign(alice_outputs)

    def alice_play_one_per_code(self, targets):
        alice_outputs = self.alice(targets)
        code_nos = torch.argmax(alice_outputs, dim=1).long()
        # https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits#63546308
        mask = 2 ** torch.arange(c.N_CODE - 1, -1, -1).to(c.DEVICE)
        return 2 * code_nos.unsqueeze(-1).bitwise_and(mask).ne(
            0).float() - 1

    def bob_play(self, selections, codes):
        return eval(f'self.bob_play_{h.BOB_STRATEGY}('
                    f'selections, codes)')

    def bob_play_one_per_bit(self, selections, codes):
        bob_input = torch.cat([
                    selections.reshape((
                        h.GAMESIZE,
                        h.N_SELECT * self.tuple_specs.n_elements)),
                    codes], 1
                )
        bob_q_estimates = self.bob(bob_input)
        return torch.argmax(bob_q_estimates, dim=1).long()
        # TODO What about the Warning at
        # https://pytorch.org/docs/stable/generated/torch.max.html?highlight
        # =max#torch.max ?  and see also torch.amax  Seems OK from testing.

    def alice_train(self, targets, rewards, codes):
        return eval(
            f'self.alice_train_{h.ALICE_STRATEGY}(targets, rewards, codes)')

    def alice_train_one_per_bit(self, targets, rewards, codes):
        alice_outputs = self.alice(targets)
        alice_qs = torch.einsum('ij, ij -> i', alice_outputs, codes)
        alice_loss = self.alice_loss_function(alice_qs, rewards)
        return alice_loss

    def alice_train_one_per_code(self, targets, rewards, codes):
        alice_outputs = self.alice(targets)
        # https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits#63546308
        mask = 2 ** torch.arange(c.N_CODE - 1, -1, -1).to(c.DEVICE)
        codes_nos = torch.sum(mask * codes, -1).long()
        alice_qs = alice_outputs[torch.arange(self.size0), codes_nos]
        alice_loss = self.alice_loss_function(alice_qs, rewards)
        return alice_loss

    def bob_train(self, selections, codes, decision_nos, rewards):
        return eval(
            f'self.bob_train_{h.BOB_STRATEGY}(selections, codes, decision_nos,'
            f' rewards)')

    def bob_train_one_per_bit(self, selections, codes, decision_nos, rewards):
        bob_input = torch.cat([
            selections.reshape((
                h.BATCHSIZE,
                h.N_SELECT * self.tuple_specs.n_elements)), codes], 1
        )
        bob_q_estimates = self.bob(bob_input)
        decision_qs = self.gatherer(bob_q_estimates, decision_nos,
                                    'Decision_Qs')
        return self.bob_loss_function(decision_qs, rewards)


