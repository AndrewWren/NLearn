from collections import namedtuple
import copy
import numpy as np
import torch
import src.books as books
import src.lib.ml_utilities as mlu
from src.lib.ml_utilities import c, h, to_array, to_device_tensor, writer
from src.game_set_up import GameOrigins, \
    GameReports, SessionSpec
from src.noise import Noise
import src.strategies._alice_play, src.strategies._alice_train
import src.strategies._bob_play, src.strategies._bob_train
import src.strategies._net, src.strategies._loss_function


LossInfo = namedtuple('LossInfo', 'bob_loss iteration alice_loss')


class Agent:
    def __init__(self, session, name):
        """

        :param session: Session
        :param name: str - must be the same as the name created  #TODO make
        this intrinsic to the code?
        """
        self.prefix = f'src.strategies._{name.lower()}_'
        self.session = session
        self.name = name.upper()
        self.play = self.use_key('play')
        self.train = self.use_key('train')
        self.net = self.use_key('net', nameless=True).to(c.DEVICE)  # Note the
        # net class uses the .play
        self.double_copy_period = None
        self.training_net = self.net
        try:
            double_copy_period = h[f'{self.name}_DOUBLE']
            if double_copy_period:  # None is the correct setting to not establish a
                # separate training net
                self.double_copy_period = double_copy_period
                self.training_net = copy.deepcopy(self.net)
        except:
            pass
        self.optimizer = self.get_optimizer()
        #with AvoidDeprecationWarning():
        self.loss_function = self.use_key('loss_function', nameless=True)

    def use_key(self, label, nameless=False):
        item = h[f'{self.name}_{label.upper()}']
        if '(' in item:
            item = item.replace('(', '(self, ', 1)
        else:
            item += '(self)'
        if nameless:
            return eval(f'src.strategies._{label.lower()}.{item}')
        else:
            return eval(f'{self.prefix}{label.lower()}.{item}')

    def get_optimizer(self):
        """
        optimizer has to be done differently as don't have
        _<agent>_optimizer files
        """
        item = h[f'{self.name}_OPTIMIZER']
        first_arg = 'self.training_net.parameters()'
        if '(' in item:
            item = item.replace('(', f'({first_arg}, ', 1)
        else:
            item += f'({first_arg})'
        return eval(f'torch.optim.{item}')


class Session:
    # file name --- and for related classes
    def __init__(self, session_spec: SessionSpec):
        self.session_spec = session_spec
        self.targets_t = None
        self.decisions = None
        self.game_reports = GameReports(GameOrigins(None, None, None), None,
                                        None, None)
        self.current_iteration = None
        self.alice = Agent(self, 'alice')
        self.bob = Agent(self, 'bob')
        self.selections = session_spec.selections
        self.n_select = h.N_SELECT
        self.epsilon_slope = (1 - h.EPSILON_MIN) / (
                h.EPSILON_MIN_POINT - h.EPSILON_ONE_END)
        self.size0 = None
        self.last_alice_loss = None
        self.noise = Noise(h.NOISE)
        self.noise_end = h.N_ITERATIONS - 1.1 * h.BUFFER_CAPACITY / h.GAMESIZE

    @torch.no_grad()
    def play(self, game_origins: GameOrigins):
        """
        Forward passes
        """
        self.set_size0(h.GAMESIZE)
        self.epsilon = self.epsilon_function(game_origins.iteration)
        targets = game_origins.selections[np.arange(self.size0),
                                          game_origins.target_nos]
        self.targets_t = to_device_tensor(targets).long()
        greedy_codes = self.alice.play()
        self.codes, chooser_a = self.alice_eps_greedy(greedy_codes)
        if (game_origins.iteration >= h.NOISE_START) and (
                game_origins.iteration < self.noise_end):
            self.codes = self.noise.inject(self.codes)
        self.selections = to_device_tensor(game_origins.selections)
        bob_q_estimates_argmax = self.bob.play()
        decision_nos, chooser_b = self.bob_eps_greedy(bob_q_estimates_argmax)
        decisions = self.selections[torch.arange(self.size0), decision_nos]
        # self.gatherer(self.selections, decision_nos, 'Decisions')
        rewards = self.session_spec.rewards(grounds=targets,
                                            guesses=to_array(
                                                decisions).astype(int))
        non_random_rewards = rewards[chooser_a & chooser_b]
        if non_random_rewards.shape[0]:
            non_random_rewards_0 = non_random_rewards
        else:
            non_random_rewards_0 = np.zeros(1)
        writer.add_scalars(
            'Rewards',
            {f'Mean reward_{h.hp_run}': np.mean(non_random_rewards_0),
             f'SD reward_{h.hp_run}': np.std(non_random_rewards_0)},
            global_step=game_origins.iteration
        )
        game_reports = GameReports(game_origins, to_array(self.codes),
                                        to_array(decision_nos), rewards)
        return non_random_rewards, game_reports

    def train(self, current_iteration, buffer):
        """
        Calculating Q values on the current alice and bob sessopm.  Training
        through backward pass
        :param current_iteration: int
        :param buffer: ReplayBuffer
        :return (alice_loss.item(), bob_loss.item()): (float, float)
        """
        self.current_iteration = current_iteration
        self.game_reports = buffer.sample()
        self.size0 = h.BATCHSIZE
        self.epsilon = self.epsilon_function(current_iteration)

        # Alice
        if current_iteration <= h.ALICE_LAST_TRAINING:
            self.targets_t = to_device_tensor(
                self.game_reports.selections[
                    np.arange(self.size0),
                    self.game_reports.target_nos
                ]
            ).long()
            self.decisions = to_device_tensor(self.game_reports.selections)[
                    torch.arange(self.size0), self.game_reports.decision_nos]
            self.codes = to_device_tensor(self.game_reports.codes)
            self.rewards = to_device_tensor(self.game_reports.rewards)
            alice_loss = self.alice.train()
            self.alice.optimizer.zero_grad()
            alice_loss.backward()
            self.alice.optimizer.step()
            self.last_alice_loss = alice_loss
            if self.alice.double_copy_period is not None:
                if current_iteration % self.alice.double_copy_period == 0:
                    self.alice.net = copy.deepcopy(self.alice.training_net)
        else:
            alice_loss = self.last_alice_loss

        # Bob
        # Next three lines are playing it safe to avoid any gradient info from
        # Alice - probably unnecessary
        self.codes = to_device_tensor(self.game_reports.codes)
        self.rewards = to_device_tensor(self.game_reports.rewards)
        self.selections = to_device_tensor(self.game_reports.selections) #
        self.decision_nos = to_device_tensor(
            self.game_reports.decision_nos).long()
        bob_loss = self.bob.train()
        self.bob.optimizer.zero_grad()
        bob_loss.backward()
        self.bob.optimizer.step()

        # logging of various sorts
        writer.add_scalars(
            'Sqrt losses',
            {f'Alice sqrt loss_{h.hp_run}': torch.sqrt(alice_loss),
             f'Bob sqrt loss_{h.hp_run}': torch.sqrt(bob_loss)
             },
            global_step=current_iteration
        )
        if current_iteration % c.CODE_BOOK_PERIOD == 0:
            mlu.log(f'Iteration={current_iteration:>10} training nets give:',
                    backspaces=20)
            mlu.log(f'{alice_loss.item()=}\t{bob_loss.item()=}')
            books.code_decode_book(self.alice, self.bob)
        elif current_iteration % 1000 == 0:
            print('\b' * 20 + f'Iteration={current_iteration:>10}', end='')

        return alice_loss.item(), bob_loss.item()

    def alice_eps_greedy(self, greedy_codes):
        """

        :param greedy_codes: torch.float32, size (h.GAMESIZE or h.BATCHSIZE,
        h.N_CODE)
        :return: torch.int64, size (h.GAMESIZE or h.BATCHSIZE respectively,
        h.N_CODE)
        """
        indicator = torch.empty(self.size0).to(c.DEVICE)
        indicator.uniform_()
        chooser = (indicator >= self.epsilon).long()
        random_codes = torch.empty(self.size0, h.N_CODE).to(
            c.DEVICE)
        random_codes.random_(to=2).long()
        random_codes = 2 * random_codes - 1
        for_choice = torch.stack((random_codes, greedy_codes), dim=1)
        temp = self.gatherer(for_choice, chooser, 'Alice').long()
        return temp, to_array(chooser).astype(bool)

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
        random_indices.random_(to=h.N_SELECT)
        for_choice = torch.dstack((random_indices, greedy_indices))[0]
        temp0 = for_choice[list(range(self.size0)), chooser].long()
        return temp0, to_array(chooser).astype(bool)

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
        if context == 'Alice':
            indices = indices.unsqueeze(1).repeat(1, input.size()[2]
                                                  ).unsqueeze(1)
        elif context == 'Decisions':
            indices = indices.unsqueeze(1).repeat(1, input.size()[-1]
                                                  ).unsqueeze(1)
            #.unsqueeze(1) #TODO This is the line to sort out!!
        elif context == 'Decision_Qs':
            indices = indices.unsqueeze(1)
        else:
            exit(f'Invalid {context=}')
        gathered = torch.gather(input, 1, indices).squeeze()
        if self.size0 == 1:
            return gathered.unsqueeze(0)
        if context == 'Decisions':
            return gathered.unsqueeze(1)
        return gathered

    def set_size0(self, size0: int):
        self.size0 = size0
