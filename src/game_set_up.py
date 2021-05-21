import math
from collections import deque, namedtuple
import numpy as np
import src.lib.ml_utilities as mlu
from src.lib.ml_utilities import c, h


class NiceCode:
    def __init__(self, value: list):
        self.tuple = self.binary_list(value)

    def to_bit(self, b):
        if b > 0:
            return 1
        elif b < 0:
            return 0
        return 2

    def binary_list(self, bb):
        return [self.to_bit(b) for b in bb]

    def raw(self):
        return np.array([ b * 2 - 1 for b in self.tuple])

    def __str__(self):
        b_list = [str(int(b)) for b in self.tuple]
        return ''.join(b_list)

    def __repr__(self):
        return 'Code(' + self.__str__() + ')'

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


class Basic:
    def __init__(self):
        self.modulus = h.N_NUMBERS
        self.n_select = h.N_SELECT
        self.circular_map = lambda x: np.transpose((np.cos(x), np.sin(x)))
        self.domain = self.circular_map(
            np.arange(self.modulus) * 2 * np.pi / self.modulus
        )
        self.reward_type = h.REWARD_TYPE
        if self.reward_type == 'Square distance':
            distinct_random_square_distances = np.sum(
                np.square(self.domain[1:, :] - self.domain[0, :]), axis=-1
            )
            mean_random_sq_distance = np.mean(
                distinct_random_square_distances) * (1 - 1 / self.n_select)
            self.var_random_guessing = np.mean(
                np.square(distinct_random_square_distances)) * (1 - 1 /
                                                               self.n_select) \
                                       - mean_random_sq_distance ** 2
            self.factor = 1 / mean_random_sq_distance
        elif self.reward_type == 'Near misses only':
            self.factor = self.modulus / ((self.modulus - 5) * 6 + 2 * 5 + 2
                                          * 4 + 1 * 0)
            self.var_random_guessing = ((self.modulus - 5) * 36 + 2 * 25 + 2
                                          * 16 + 1 * 0) / self.modulus \
                                       - 1 / (self.factor ** 2)
            mlu.log(f'0 away is reward of 1')
            mlu.log(f'1 away is reward of {1 - 4 * self.factor}')
            mlu.log(f'2 away is reward of {1 - 5 * self.factor}')
        elif self.reward_type == 'Exact only':
            self.factor = self.modulus / ((self.modulus - 1) * 1)
            self.var_random_guessing = ((self.modulus - 1) * 1) / self.modulus\
                                       - 1 / (self.factor ** 2)
            mlu.log(f'0 away is reward of 1')
            mlu.log(f'1 or more away is reward of {1 - 1 * self.factor}')
        else:
            exit(f'Invalid REWARD_TYPE {self.reward_type}')
        self.domain_t = mlu.to_device_tensor(self.domain)
        try:
            shuffle_flag = h.SHUFFLE
        except:
            shuffle_flag = False
        self.numbers = np.arange(h.N_NUMBERS)
        if shuffle_flag:
            h.ne_rng.shuffle(self.numbers)
        self.numbers_t = mlu.to_device_tensor(self.numbers).long()

    def rewards(self, grounds, guesses):
        """
        Returns the square of the Euclidean distance between each of the
        points represented by the final dimension.
        :param grounds: np.array of size (self.size0, 2)
        :param guesses: np.array of size (self.size0, 2)
        :return: float
        """
        if self.reward_type == 'Square distance':
            grounds = self.circle(grounds)
            guesses = self.circle(guesses)
            return 1 - np.sum(
                np.square(guesses - grounds),
                axis=-1
            ) * self.factor
        elif self.reward_type == 'Near misses only':
            grounds = self.numbers[grounds]
            guesses = self.numbers[guesses]
            separation = guesses - grounds
            separation = separation % self.modulus
            separation = np.minimum(separation, self.modulus - separation)
            return 1 - (
                    6 - np.maximum(3 - separation, 0)
                    - 3 * (separation == 0)
            ) * self.factor
        elif self.reward_type == 'Exact only':
            grounds = self.numbers[grounds]
            guesses = self.numbers[guesses]
            separation = guesses - grounds
            separation = separation % self.modulus
            separation = np.minimum(separation, self.modulus - separation)
            return 1 - (1 - (separation == 0)) * self.factor

    def circle(self, numbers):
        return self.domain[self.numbers[numbers]]

    def circle_t(self, numbers):
        return self.domain_t[self.numbers_t[numbers.long()]]

    def __repr__(self):
        return f'Basic({self.modulus}, {self.n_select})'


GameOrigin = namedtuple('GameOrigin', 'iteration target_nos selections')
GameReport = namedtuple('GameReport', 'iteration target_no selection code '
                                      'decision_no reward')
GameOrigins = namedtuple('GameOrigins', 'iteration target_nos selections')


class GameReports:
    def __init__(self, game_origins, codes, decision_nos, rewards):
        self.iteration = game_origins.iteration
        self.target_nos = game_origins.target_nos
        self.selections = game_origins.selections
        self.codes = codes
        self.decision_nos = decision_nos
        self.rewards = rewards

    def game_report_list(self):
        zipee = [
            [self.iteration] * h.GAMESIZE,
            self.target_nos,
            self.selections,
            self.codes,
            self.decision_nos,
            self.rewards
        ]
        return [GameReport(*report) for report in zip(*zipee)]


class SessionSpec:
    def __init__(self):
        self.spec = eval(h.NUMBERS + '()')
        self.size0 = h.GAMESIZE
        self.selections = None

    def random(self):
        """

        :return: a np array of size (self.size0, h.N_SELECT,
        self.n_elements, 2)
        """
        return np.stack(
            [h.ne_rng.choice(self.spec.numbers, size=h.N_SELECT,
                              replace=False)
             for _ in range(self.size0)]
        )

    def iter(self):
        for iteration in range(1, h.N_ITERATIONS + 1):
            selections = self.random()
            target_nos = h.ne_rng.integers(h.N_SELECT, size=self.size0)
            game_origins = GameOrigins(iteration, target_nos, selections)
            yield game_origins

    def rewards(self, grounds, guesses):
        """
        :param grounds: numpy array, shape = (self.size0, self.n_elements)
        :param guesses: numpy array, shape = (self.size0, self.n_elements)
        :return:  numpy array, shape = (self.size0, )
        """
        return self.spec.rewards(grounds, guesses)

    def random_reward_sd(self):
        return math.sqrt(self.spec.var_random_guessing * (
                self.spec.factor ** 2))

    def __repr__(self):
        return f'SessionSpec({self.specs})'


# From github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/Chapter06/02_dqn_pong.py
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, game_reports):
        self.buffer.extend(game_reports.game_report_list())

    def sample(self):
        indices = np.random.choice(len(self.buffer), h.BATCHSIZE,
                                   replace=False)
        game_reports_list = list(map(list,
                                     zip(*[self.buffer[idx] for idx in
                                           indices])))
        iteration, target_nos, selections = game_reports_list[: 3]
        iteration = np.array(iteration)
        target_nos = np.array(target_nos)
        selections = np.stack(selections)
        game_origins = GameOrigins(iteration, target_nos, selections)
        codes, decision_nos, rewards = game_reports_list[3:]
        codes = np.vstack(codes)
        decision_nos = np.array(decision_nos)
        rewards = np.array(rewards)
        return GameReports(game_origins, codes, decision_nos, rewards)
