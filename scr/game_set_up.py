import math
from collections import deque, namedtuple
from collections.abc import Callable
import random
import numpy as np
import numpy.typing as npt
import torch

import config
from scr.ml_utilities import c, h, rng_c


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

    def __str__(self):
        b_list = [str(int(b)) for b in self.tuple]
        return ''.join(b_list)

    def __repr__(self):
        return 'Code(' + self.__str__() + ')'

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


class Domain:
    def __init__(self, domain_type: type, domain_range):
        self.type = domain_type
        self.range = domain_range
        if self.type == int:
            self.range = int(self.range)

    def __repr__(self):
        return f'Domain({self.__str__()})'

    def __str__(self):
        return f'({self.type}, {self.range})'


class ElementCircular:
    def __init__(self, modulus: int, n_select=None):
        self.modulus = modulus
        self.n_select = n_select or h.N_SELECT
        self.circular_map = lambda x: np.transpose((np.cos(x), np.sin(x)))
        self.domain = self.circular_map(
            np.arange(modulus) * 2 * np.pi / modulus
        )
        distinct_random_square_distances = np.sum(
            np.square(self.domain[1:, :] - self.domain[0, :]), axis=-1
        )
        self.mean_random_sq_distance = np.mean(
            distinct_random_square_distances) * (1 - 1 / self.n_select)
        self.var_random_sq_distance = np.mean(
            np.square(distinct_random_square_distances)) * (1 - 1 /
                                                           self.n_select) \
            - self.mean_random_sq_distance ** 2
        self.factor = 1 / self.mean_random_sq_distance
        pass

    def rewards(self, grounds, guesses):
        """
        Returns the square of the Euclidean distance between each of the
        points represented by the final dimension.
        :param grounds: np.array of size (self.size0, 2)
        :param guesses: np.array of size (self.size0, 2)
        :return: float
        """
        return 1 - self.factor * np.sum(
            np.square(guesses - grounds),
            axis=-1)

    def __repr__(self):
        return f'ElementCircular({self.modulus}, {self.n_select})'


GameOrigin = namedtuple('GameOrigin', 'iteration target_nos selections')

GameReport = namedtuple('GameReport', 'iteration target_no selection code '
                                      'decision_no rewards')

GameOrigins = namedtuple('GameOrigins', 'iteration target_nos selections')

GameReports = namedtuple('GameReports', 'gameorigins codes decision_nos '
                                        'rewards')


class GameReports(GameReports):
    def __init__(self, gameorigins, codes, decisions, rewards):
        super().__init__()
        self.iteration = self.gameorigins.iteration
        self.target_nos = self.gameorigins.target_nos
        self.selections = self.gameorigins.selections

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


class TupleSpecs:
    def __init__(self, specs: list = c.TUPLE_SPEC):
        spec_list = list()
        for spec in specs:
            if not isinstance(spec, ElementCircular):
                spec = ElementCircular(*spec)
            spec_list.append(spec)
        self.specs = tuple(spec_list)
        self.n_elements = len(spec_list)
        self.size0 = h.GAMESIZE
        self.selections = None

    def random(self):
        """

        :return: a np array of size (self.size0, h.N_SELECT,
        self.n_elements, 2)
        """
        randoms = np.stack(
            [[h.n_rng.choice(spec.domain, size=h.N_SELECT, replace=False)
              for spec in self.specs]
             for _ in range(self.size0)]
        )  # TODO in the choice consider setting shuffle=False
        return np.transpose(randoms, (0, 2, 1, 3))

    def iter(self):
        for iteration in range(1, h.N_ITERATIONS + 1):
            selections = self.random()
            target_nos = h.n_rng.integers(h.N_SELECT, size=self.size0)
            game_origins = GameOrigins(iteration, target_nos, selections)
            yield game_origins

    def rewards(self, grounds, guesses):
        """
        :param grounds: numpy array, shape = (self.size0, self.n_elements)
        :param guesses: numpy array, shape = (self.size0, self.n_elements)
        :return:  numpy array, shape = (self.size0, )
        """
        to_stack = tuple(spec.rewards(grounds[:, s, ...], guesses[:, s, ...])
                         for s, spec in enumerate(self.specs))
        if len(to_stack) > 1:
            rewards = np.dstack(to_stack)
            return np.sum(rewards, axis=-1)
        else:
            return to_stack[0]

    def random_reward_sd(self):
        return math.sqrt(sum([spec.var_random_sq_distance * (spec.factor ** 2)
                              for spec in self.specs]))

    def __repr__(self):
        return f'TupleSpecs({self.specs})'


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
