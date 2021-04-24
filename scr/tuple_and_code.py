from collections import namedtuple
from collections.abc import Callable
import random
import numpy as np
import numpy.typing as npt
import torch

import config
from scr.ml_utilities import c, h, rng_c


class Code:
    """Not yet used.  Consider if needed - and if so need to accommodate
    batches"""
    def __init__(self, value: list):
        if len(value) != config.N_CODE:
            exit(f'Defining Code instance with length {len(value)} != '
                 f'config.N_CODE')
        self.tuple = self.binary_list(value)

    def to_bit(self, b):
        if b > 0:
            return 1
        elif b < 0:
            return 0
        return random.randrange(2)

    def binary_list(self, bb):
        return [self.to_bit(b) for b in bb]

    def __str__(self):
        b_list = [str(int(b)) for b in self.tuple]
        return ''.join(b_list)

    def __repr__(self):
        return 'Code(' + self.__str__() + ')'


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


class ElementSpec:
    def __init__(
            self,
            domain: Domain or tuple,
            reward: tuple = (1, 0),
            fraction: float = 0.2,
            random: Callable = None
    ):
        if isinstance(domain, Domain):
            self.domain = domain
        else:
            self.domain = Domain(*domain)

        """Makes the reward an attribute, modifying it to get a zero mean 
        reward over h.N_SELECT
        """
        #print(reward, h.N_SELECT)
        self.mean_reward = (reward[0] + (h.N_SELECT - 1) * reward[1]) / \
                           h.N_SELECT
        self.right = reward[0] - self.mean_reward
        self.wrong = reward[1] - self.mean_reward

        self.fraction = fraction
        self.repetition = None
        self.n_repetitions = None
        self.stub_length = None

        default_random = None
        default_rewarder = None
        if (self.domain.type == int) and (self.domain.range != np.inf):
            default_random = self.default_random_selector_int_finite
            self.random_repr = 'Uniform'
        self.random = default_random
        if self.random is None:
            exit('Need a random for the ElementSpec initiation.')

    def default_random_selector_int_finite(self):
        return h.n_rng.integers(
            self.domain.range,
            size=(h.BATCHSIZE, 1)
        ) * (2 / self.domain.range) - 1

    def __repr__(self):
        return f'Element({self.domain}, {self.rewarder}, {self.random_repr})'

    def reward(self, ground: int, guess: int):
        if ground == guess:
            return self.rewarder.right
        else:
            return self.rewarder.wrong


def flatten_shuffle(ls):
    fl = [item for sublist in ls for item in sublist]
    random.shuffle(fl)
    return fl


def transpose_tuple(a):
    return tuple(map(tuple, zip(*a)))


GameOrigins = namedtuple('GameOrigins', 'iteration target_nos selections')


GameReports = namedtuple('GameReports', 'gameorigins decisions rewards')
class GameReports(GameReports):
    def __init__(self, gameorigins, decisions, rewards):
        super().__init__()
        self.iteration = self.gameorigins.iteration
        self.target_no = self.gameorigins.target_nos
        self.selection = self.gameorigins.selections


class TupleSpecs:
    def __init__(
            self,
            specs: list = c.TUPLE_SPEC,
            pool: int = 100
    ):
        self.pool = pool
        spec_list = list()
        for spec in specs:
            if not isinstance(spec, ElementSpec):
                spec = ElementSpec(*spec)
            spec_list.append(spec)
            spec.repetition = round(spec.fraction * self.pool)
            spec.n_repetitions = self.pool // spec.repetition
            spec.stub_length = self.pool % spec.repetition
        self.specs = tuple(spec_list)
        self.n_elements = len(spec_list)
        self.selections = None
        self.target_no = None
        
        """ Some attributes related to reward and its calculation
        """
        self.current_reward = None
        right = torch.FloatTensor([spec.right for spec in self.specs]).to(
            c.DEVICE)
        wrong = torch.FloatTensor([spec.wrong for spec in self.specs]).to(
            c.DEVICE)
        self.factor = right - wrong
        self.offset = (torch.sum(wrong)).repeat(h.BATCHSIZE)
        
        self.n_iterations = h.N_ITERATIONS

    def random(self):
        selections = list()
        for spec in self.specs:
            random_selections = np.concatenate(
                [np.repeat(spec.random(), spec.repetition, axis=1)
                 for _ in range(spec.n_repetitions)]
                + [np.repeat(spec.random(), spec.stub_length, axis=1)],
                axis=1
            )
            h.n_rng.shuffle(random_selections, axis=1)
            selections.append(random_selections[:,: h.N_SELECT])
        selections = np.stack(selections, axis=-1)  #TODO consider making
        # this a 2D rather than a 3D array to avoid later reshaping.
        # Means picking the target state requires a tiny bit of arithmetic.
        self.selections = selections
        #print(f'{self.selections.shape=}')
        return self.selections

    def iter(self):
        # print(f'{h.N_ITERATIONS=}')
        for iteration in range(h.N_ITERATIONS):
            self.random()
            self.target_nos = h.n_rng.integers(h.N_SELECT, size=h.BATCHSIZE)
            game_origins = GameOrigins(
                iteration, self.target_nos, self.selections
            )
            yield game_origins

    def __repr__(self):
        return f'TupleSpecs({self.specs})'

    def rewards(self, grounds, guesses):
        """
        :param grounds: torch.float32, size = (h.batchsize, self.n_elements)
        :param guesses: torch.float32, size = (h.batchsize, self.n_elements)
        :return: torch.float32, size = (h.batchsize, )
        """
        """print(f'{grounds.size()=}')
        print(f'{guesses.size()=}')
        print(f'{self.offset.size()=}')
        print(f'{(grounds == guesses).float()=}')
        """
        self.current_reward = torch.matmul((grounds == guesses).float(),
                                           self.factor)\
                              + self.offset
        return self.current_reward
