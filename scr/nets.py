from collections import  namedtuple
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


"""def argmax(ls):
    # from https://towardsdatascience.com/there-is-no-argmax-function-for-python-list-cd0659b05e49
    f = lambda i: ls[i]
    return max(range(len(ls)), key=f)
"""


class Nets:
    def __init__(self, tuple_specs: TupleSpecs):
        self.tuple_specs = tuple_specs
        self.alice = FFs(input_width=tuple_specs.n_elements,
                         output_width=c.N_CODE, layers=2).to(c.DEVICE)
        self.alice_optimizer = self.optimizer('ALICE')
        self.alice_loss_function = self.loss_function('ALICE')
        self.bob = FFs(input_width=h.N_SELECT * tuple_specs.n_elements +
                                   c.N_CODE,
                       output_width=h.N_SELECT, layers=2).to(c.DEVICE)
        if h.BOB_OPTIMIZER == 'Same':
            bob_optimizer_label = 'ALICE'
        else:
            bob_optimizer_label = 'BOB'
        #print(f'{bob_optimizer_label=}')
        self.bob_optimizer = self.optimizer(bob_optimizer_label)
        if h.BOB_LOSS_FUNCTION == 'Same':
            bob_loss_function_label = 'ALICE'
        else:
            bob_loss_function_label = 'BOB'
        self.bob_loss_function = self.loss_function(bob_loss_function_label)
        self.selections = tuple_specs.selections
        # print(f'{self.selections=}')
        self.epsilon_slope = (1 - h.EPSILON_MIN) / (
                h.EPSILON_MIN_POINT - h.EPSILON_ONE_END)
        self.size0 = None

    def optimizer(self, net_label):
        values = eval('h.' + net_label + '_OPTIMIZER')
        #print(f'For {net_label}, {values=}')
        return eval(
            'torch.optim.' + values[0]
            + '(self.alice.parameters(), **' + str(values[1]) + ')'
        )

    def loss_function(self, net_label):
        values = eval('h.' + net_label + '_LOSS_FUNCTION')
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
        #print(f'{targets.shape=}')
        targets = to_device_tensor(targets)
        alice_outputs = self.alice(targets)
        ####alice_qs = torch.sum(torch.abs(alice_outputs), dim=1)
        greedy_codes = torch.sign(alice_outputs).detach()
        codes = self.alice_eps_greedy(greedy_codes)
        ######
        """print(f'{game_origins.selections.shape=}')
        print(f'{targets.shape=}')
        print(f'{codes.shape=}')
        """
        selections = to_device_tensor(game_origins.selections).detach()
        """print(f'{greedy_codes.size()=}')
        print(f'{codes.size()=}')
        print(f'{selections.size()=}')
        """
        bob_input = torch.cat([
            selections.reshape((
                h.GAMESIZE,
                h.N_SELECT * self.tuple_specs.n_elements)),
            codes], 1
        )
        #bob_input.requires_grad = True
        #bob_input.retain_grad()
        # print(f'{bob_input.shape=}')
        bob_q_estimates = self.bob(bob_input)
        #bob_q_estimates.retain_grad()
        bob_q_estimates_argmax = torch.argmax(bob_q_estimates, dim=1).long()
        #bob_q_estimates_argmax.requires_grad = True
        #bob_q_estimates_argmax.retain_grad()
        # TODO What about the Warning at
        # https://pytorch.org/docs/stable/generated/torch.max.html?highlight
        # =max#torch.max ?  and see also torch.amax  Seems OK from testing.
        decision_nos = self.bob_eps_greedy(bob_q_estimates_argmax).detach()
        #decision_nos.retain_grad()
        decisions = self.gatherer(selections, decision_nos, 'Decisions')
        #print(f'{decisions=}')
        #print(f'{decisions.size()=}')
        #print(f'{decisions.size()=}')
        ####decision_qs = self.gatherer(bob_q_estimates, decision_nos, 'Decision_Qs')
        #decision_qs.retain_grad()
        #print(f'{decision_qs.size()=}')
        rewards = self.tuple_specs.rewards(grounds=targets, guesses=decisions)
        """print(f'{targets.size()=}')
        print(f'{decision_qs.size()=}')
        print(f'{rewards.size()=}')
        """
        return GameReports(game_origins, codes, decisions, rewards)
        # Returns iteration target_nos selections decisions rewards
        # Don't return alice_qs decision_qs

    def train(self, current_iteration: int, buffer: ReplayBuffer): #TODO train
        """
        Backward passes
        """
        game_reports = buffer.sample()
        self.size0 = h.BATCHSIZE
        self.epsilon = self.epsilon_function(current_iteration)
        targets = game_reports.selections[np.arange(self.size0),
                                          game_reports.target_nos]
        targets = to_device_tensor(targets)
        alice_outputs = self.alice(targets)
        alice_qs = torch.sum(torch.abs(alice_outputs), dim=1)
        alice_loss = self.alice_loss_function(alice_qs,
                                              torch.tensor(
                                                  game_reports.rewards))
        #TODO Consider if should generate new codes????
        #greedy_codes = torch.sign(alice_outputs).detach() #TODO or do this
        # and next line after the back-optimization?
        #codes = self.alice_eps_greedy(greedy_codes)
        self.alice_optimizer.zero_grad()
        alice_loss.backward()
        self.alice_optimizer.step()
        selections = torch.FloatTensor(game_reports.selections)
        codes = torch.FloatTensor(game_reports.codes)
        bob_input = torch.cat([
            selections.reshape((
                h.BATCHSIZE,
                h.N_SELECT * self.tuple_specs.n_elements)), codes], 1
        )
        bob_q_estimates = self.bob(bob_input)
        # bob_q_estimates.retain_grad()
        bob_q_estimates_argmax = torch.argmax(bob_q_estimates, dim=1).long()
        # bob_q_estimates_argmax.requires_grad = True
        # bob_q_estimates_argmax.retain_grad()
        # TODO What about the Warning at
        # https://pytorch.org/docs/stable/generated/torch.max.html?highlight
        # =max#torch.max ?  and see also torch.amax  Seems OK from testing.
        decision_nos = self.bob_eps_greedy(bob_q_estimates_argmax).detach()
        decision_qs = self.gatherer(bob_q_estimates, decision_nos,
                                    'Decision_Qs')
        rewards = torch.FloatTensor(game_reports.rewards)
        bob_loss = self.bob_loss_function(decision_qs, rewards)
        self.bob_optimizer.zero_grad()
        bob_loss.backward()
        self.bob_optimizer.step()
        writer.add_scalars(
            'Sqrt losses',
            {'Alice sqrt loss': torch.sqrt(alice_loss),
             'Bob sqrt loss': torch.sqrt(bob_loss)
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
        #print(f'{greedy_codes.size()=}')
        #print(f'{self.size0=}')
        indicator = torch.empty(self.size0)
        indicator.uniform_()
        chooser = (indicator >= self.epsilon).long()
        #chooser = chooser.unsqueeze(-1).unsqueeze(-1).repeat(1, c.N_CODE, 2)
        #print(f'{chooser=}')
        #print(f'{chooser.size()=}')
        random_codes = torch.empty(self.size0, c.N_CODE)
        random_codes.random_(to=2).long()
        random_codes = 2 * random_codes - 1
        for_choice = torch.stack((random_codes, greedy_codes), dim=1)
        #print(f'{for_choice.size()=}')
        """print(f'{greedy_codes.size()=}')
        print(f'{random_codes=}')
        print(f'{for_choice.size()=}')
        print(f'{chooser.size()=}')
        print(f'{for_choice=}')
        print(f'{chooser.shape=}')
        print(f'{for_choice.shape=}')
        """
        temp = self.gatherer(for_choice, chooser, 'Alice').long()
        #print(f'{temp=}')
        #print(f'{temp.size()=}')
        return temp

    def bob_eps_greedy(self, greedy_indices):
        """

        :param iteration: int
        :param greedy_indices: torch.float32, size (h.GAMESIZE or h.BATCHSIZE)
        :return: torch.int64, size (h.GAMESIZE or h.BATCHSIZE respectively)
        """
        indicator = torch.empty(self.size0)
        indicator.uniform_()
        #print(f'{indicator=}')
        chooser = (indicator >= self.epsilon).long()
        #print(f'{chooser=}')
        random_indices = torch.empty(self.size0)
        random_indices.random_(h.N_SELECT).long()
        for_choice = torch.dstack((random_indices, greedy_indices))[0]
        """print(f'{for_choice=}')
        print(f'{chooser.shape=}')
        print(f'{for_choice.shape=}')
        """
        return for_choice[list(range(self.size0)), chooser].long()

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
        if iteration >= h.EPSILON_MIN_POINT:
            single_epsilon = h.EPSILON_MIN * torch.ones(1)
        else:
            single_epsilon = torch.FloatTensor([
                1.
                - max(iteration - h.EPSILON_ONE_END, 0) * self.epsilon_slope
            ])
        return single_epsilon.repeat(self.size0)

    def gatherer(self, input, indices, context):
        #print(f'{context=}: {input.size()=}, {indices.size()=}')
        if (context == 'Alice') or (context == 'Decisions'):
            indices = indices.unsqueeze(1).repeat(1,input.size()[2]
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
