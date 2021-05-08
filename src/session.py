from collections import namedtuple
import copy
import numpy as np
import torch
import src.books as books
import src.ml_utilities as mlu
from src.ml_utilities import c, h, to_array, to_device_tensor, writer
from src.game_set_up import GameOrigins, \
    GameReports, NiceCode, TupleSpecs
from src.nets import FFs
from src.noise import Noise
import src.strategies._alice_play, src.strategies._alice_train, \
    src.strategies._alice_net, src.strategies._alice_loss_function
#import src.strategies._bob_play, src.strategies._bob_train,
# src.strategies._bob_net, src.strategies._bob_loss_function


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
        self.net = self.use_key('net')  # Note the net class uses the .play
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
        self.loss_function = self.use_key('loss_function')

    def use_key(self, label):
        item = h[f'{self.name}_{label.upper()}']
        if '(' in item:
            item = item.replace('(', '(self, ', 1)
        else:
            item += '(self)'
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
    def __init__(self, tuple_specs: TupleSpecs):
        self.tuple_specs = tuple_specs
        self.targets_t = None
        self.decisions = None
        self.game_reports = GameReports(GameOrigins(None, None, None), None,
                                        None, None)
        self.current_iteration = None
        self.alice = Agent(self, 'alice')
        self.set_widths()
        self.bob = FFs(
            input_width=self.bob_input_width,
            output_width=self.bob_output_width,
            layers=h.BOB_LAYERS,
            width=h.BOB_WIDTH
        ).to(c.DEVICE)
        self.bob_optimizer = self.optimizer('BOB', 'bob')
        if h.BOB_LOSS_FUNCTION == 'Same':
            bob_loss_function_label = 'ALICE'
        else:
            bob_loss_function_label = 'BOB'
        self.bob_loss_function = self.loss_function(bob_loss_function_label)
        self.selections = tuple_specs.selections
        self.epsilon_slope = (1 - h.EPSILON_MIN) / (
                h.EPSILON_MIN_POINT - h.EPSILON_ONE_END)
        self.size0 = None
        self.last_alice_loss = None
        self.noise = Noise(h.NOISE)
        self.noise_end = h.N_ITERATIONS - 1.1 * h.BUFFER_CAPACITY / h.GAMESIZE

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
            values[0] + 'Loss(**' + str(values[1]) + ')'
        )

    @torch.no_grad()
    def play(self, game_origins: GameOrigins):
        """
        Forward passes
        """
        self.set_size0(h.GAMESIZE)
        self.epsilon = self.epsilon_function(game_origins.iteration)
        targets = game_origins.selections[np.arange(self.size0),
                                          game_origins.target_nos]
        self.targets_t = to_device_tensor(targets)
        greedy_codes = self.alice.play()
        codes, chooser_a = self.alice_eps_greedy(greedy_codes)
        if (game_origins.iteration >= h.NOISE_START) and (
                game_origins.iteration < self.noise_end):
            codes = self.noise.inject(codes)
        selections = to_device_tensor(game_origins.selections)
        bob_q_estimates_argmax = self.bob_play(selections, codes)
        decision_nos, chooser_b = self.bob_eps_greedy(bob_q_estimates_argmax)
        decisions = self.gatherer(selections, decision_nos, 'Decisions')
        rewards = self.tuple_specs.rewards(grounds=targets,
                                           guesses=to_array(decisions))
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
        game_reports = GameReports(game_origins, to_array(codes),
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
            )
            self.decisions = torch.flatten(
                to_device_tensor(self.game_reports.selections)[
                    torch.arange(self.size0), self.game_reports.decision_nos],
                start_dim=1
            )
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
        selections = to_device_tensor(self.game_reports.selections)
        codes = to_device_tensor(self.game_reports.codes)
        decision_nos = to_device_tensor(self.game_reports.decision_nos).long()
        bob_loss = self.bob_train(selections, codes, decision_nos,
                                  self.game_reports.rewards)
        self.bob_optimizer.zero_grad()
        bob_loss.backward()
        self.bob_optimizer.step()

        # logging of various sorts
        writer.add_scalars(
            'Sqrt losses',
            {f'Alice sqrt loss_{h.hp_run}': torch.sqrt(alice_loss),
             f'Bob sqrt loss_{h.hp_run}': torch.sqrt(bob_loss)
             },
            global_step=current_iteration
        )
        if current_iteration % 10000 == 0:  #10000
            mlu.log(f'Iteration={current_iteration:>10} training nets give:',
                    backspaces=20)
            mlu.log(f'{alice_loss.item()=}\t{bob_loss.item()=}')
            books.code_decode_book(
                self.alice,
                self.bob,
                c.TUPLE_SPEC[0][0],
                h.N_SELECT,
            )
        elif current_iteration % 1000 == 0:
            print('\b' * 20 + f'Iteration={current_iteration:>10}', end='')

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
                                                  ).unsqueeze(1).unsqueeze(1)
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

    def set_widths(self):
        """if (h.ALICE_STRATEGY == 'circular') or (h.ALICE_STRATEGY ==
                                                'from_decisions'):
            self.alice_output_width = c.N_CODE
        """
        if h.BOB_STRATEGY == 'circular':
            self.bob_input_width = (h.N_SELECT * self.tuple_specs.n_elements
                                    * 2 + c.N_CODE)
            self.bob_output_width = h.N_SELECT
        elif h.BOB_STRATEGY == 'circular_vocab':
            self.bob_input_width = self.tuple_specs.n_elements * 2 + c.N_CODE
            self.bob_output_width = 1

    """def alice_play_circular(self, targets):
        targets = torch.flatten(targets, start_dim=1)
        alice_outputs = self.alice(targets)
        return torch.sign(alice_outputs)
    
    def alice_play_from_decisions(self, targets):
        return self.alice_play_circular(targets)
    """
    def bob_play(self, selections, codes):
        return eval(f'self.bob_play_{h.BOB_STRATEGY}('
                    f'selections, codes)')

    def bob_play_circular(self, selections, codes):
        bob_input = torch.cat(
            [torch.flatten(selections, start_dim=1), codes], 1
        )
        bob_q_estimates = self.bob(bob_input)
        return torch.argmax(bob_q_estimates, dim=1).long()
        # TODO What about the Warning at
        # https://pytorch.org/docs/stable/generated/torch.max.html?highlight
        # =max#torch.max ?  and see also torch.amax  Seems OK from testing.

    def bob_play_circular_vocab(self, selections, codes):
        selections = torch.transpose(selections, 0, 1)
        bob_q_estimates = list()
        for selection in selections:  # TODO Could do all in a single net sessopm
            bob_input = torch.cat(
                [torch.flatten(selection, start_dim=1), codes], 1
            )
            bob_q_estimates.append(self.bob(bob_input))
        temp1 = torch.reshape(torch.stack(bob_q_estimates, dim=1),
                              (self.size0, h.N_SELECT))
        result = torch.argmax(temp1, dim=1)
        # TODO What about the Warning at
        # https://pytorch.org/docs/stable/generated/torch.max.html?highlight
        # =max#torch.max ?  and see also torch.amax  Seems OK from testing.
        return result

    """def alice_train(self, targets, rewards, codes, decisions):
        

        :param targets: numpy array
        :param rewards: numpy array
        :param codes: pytorch tensor
        :param decisions: pytorch tensor
        :return:
        
        return eval(
            f'self.alice_train_{h.ALICE_STRATEGY}(targets, rewards, codes,'
            f' decisions)')

    def alice_train_circular(self, targets, rewards, codes, decisions):
        targets = torch.flatten(targets, start_dim=1)
        alice_outputs = self.alice(targets)
        alice_qs = torch.einsum('bj, bj -> b', alice_outputs, codes)
        alice_loss = self.alice_loss_function(alice_qs, rewards)
        return alice_loss

    def alice_train_from_decisions(self, targets, rewards, codes, decisions):
        targets = torch.flatten(targets, start_dim=1)
        alice_codes_from_targets = torch.sign(self.alice(targets))
        with torch.no_grad():
            decisions = torch.flatten(decisions, start_dim=1)
            alice_codes_from_decisions = torch.sign(self.alice(decisions))
        closeness = torch.einsum('ij, ij -> i', alice_codes_from_targets,
                                 alice_codes_from_decisions) / c.N_CODE
        if self.current_iteration < h.ALICE_PROXIMITY_BONUS:
            return self.alice_loss_function(closeness, rewards)
        bonus_prop = min(1, (self.current_iteration -
                              h.ALICE_PROXIMITY_BONUS) /
                         h.ALICE_PROXIMITY_SLOPE_LENGTH)
        closeness_bonus = (closeness == 1.).float()
        rewards_bonus = (rewards == 1.).float()
        return self.alice_loss_function(closeness + closeness_bonus,
                                        rewards + rewards_bonus)
    
    #TODO Should the code actually transmitted be used at all?
    """

    def bob_train(self, selections, codes, decision_nos, rewards):
        """

        :param selections: pytorch tensor
        :param codes: pytorch tensor
        :param decision_nos: pytorch tensor
        :param rewards: numpy array
        :return:
        """
        return eval(
            f'self.bob_train_{h.BOB_STRATEGY}(selections, codes, decision_nos,'
            f' rewards)')

    def bob_train_circular(self, selections, codes, decision_nos, rewards):
        """
        As bob_train
        """
        bob_input = torch.cat(
            [torch.flatten(selections, start_dim=1), codes], 1
        )
        bob_q_estimates = self.bob(bob_input)
        decision_qs = self.gatherer(bob_q_estimates, decision_nos,
                                    'Decision_Qs')
        return self.bob_loss_function(decision_qs, to_device_tensor(rewards))

    def bob_train_circular_vocab(self, selections, codes, decision_nos,
                                 rewards):
        """
        As bob_train
        """
        bob_decisions = torch.flatten(
            selections[torch.arange(self.size0), decision_nos],
            start_dim=1
        )
        bob_input = torch.cat([bob_decisions, codes], dim=1)
        bob_decisions_q_estimates = self.bob(bob_input).reshape((self.size0,))
        return self.bob_loss_function(bob_decisions_q_estimates,
                                      to_device_tensor(rewards)
        )
