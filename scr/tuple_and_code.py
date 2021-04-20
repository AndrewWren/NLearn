from collections import namedtuple
from collections.abc import Callable
import random
import numpy as np
import config
from scr.ml_utilities import c, h, rng_c


class Code:
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


class Scorer:
    def __init__(self, right=1, wrong=0):
        self.right = right
        self.wrong = wrong
        if (right == 1) and (wrong == 0):
            self.repr = 'Default'
        else:
            self.repr = f'Special({self.right}, {self.wrong})'

    def __repr__(self):
        return self.repr


class ElementSpec:
    def __init__(
            self,
            domain: Domain or tuple,
            scorer: Scorer = None,
            fraction: float = 0.2,
            random: Callable = None
    ):
        if isinstance(domain, Domain):
            self.domain = domain
        else:
            self.domain = Domain(*domain)

        self.scorer = scorer or Scorer()

        self.fraction = fraction
        self.repetition = None
        self.n_repetitions = None
        self.stub_length = None

        default_random = None
        default_score = None
        if (self.domain.type == int) and (self.domain.range != np.inf):
            default_random = self.default_random_selector_int_finite
            self.random_repr = 'Uniform'
        self.random = default_random
        if self.random is None:
            exit('Need a random for the ElementSpec initiation.')

    def default_random_selector_int_finite(self):
        return h.rng.integers(self.domain.range, size=(h.BATCHSIZE, 1))

    def __repr__(self):
        return f'Element({self.domain}, {self.scorer}, {self.random_repr})'

    def score(self, ground: int, guess: int):
        if ground == guess:
            return self.scorer.right
        else:
            return self.scorer.wrong


def flatten_shuffle(ls):
    fl = [item for sublist in ls for item in sublist]
    random.shuffle(fl)
    return fl


def transpose_tuple(a):
    return tuple(map(tuple, zip(*a)))


GameOrigin = namedtuple('GameOrigin', 'iteration target_no selection')


GameReport = namedtuple('GameReport', 'gameorigin decision score')
class GameReport(GameReport):
    def __init__(self, gameorigin, decision, score):
        super().__init__()
        self.iteration = self.gameorigin.iteration
        self.target_no = self.gameorigin.target_no
        self.selection = self.gameorigin.selection


class TupleSpecs:
    def __init__(
            self,
            specs: list = c.TUPLE_SPEC,
            pool: int = 100,
            select: int = 10
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
        self.select = select
        self.selection = None
        self.target_no = None
        self.current_score = None
        self.n_iterations = h.N_ITERATIONS

    def random(self):
        selection = list()
        for spec in self.specs:
            random_selections = np.concatenate(
                [np.repeat(spec.random(), spec.repetition, axis=1)
                 for _ in range(spec.n_repetitions)]
                + [np.repeat(spec.random(), spec.stub_length, axis=1)],
                axis=1
            )
            h.rng.shuffle(random_selections, axis=1)
            selection.append(random_selections[:,: self.select])
        selection = np.concatenate(selection, axis=1)
        self.selection = selection
        return self.selection

    def iter(self):
        print(f'{h.N_ITERATIONS=}')
        for iteration in range(h.N_ITERATIONS):
            self.random()
            self.target_no = random.randrange(self.select)
            game_origin = GameOrigin(iteration, self.target_no, self.selection)
            yield game_origin

    def __repr__(self):
        return f'TupleSpecs({self.specs})'

    def score(self, grounds: tuple, guesses: tuple):
        score = 0
        for spec, ground, guess in zip(self.specs, grounds, guesses):
            if ground == guess:
                score += spec.scorer.right
            else:
                score += spec.scorer.wrong
        self.current_score = score
        return self.current_score
