from collections.abc import Callable
import random
import numpy as np

import config


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
            fraction: Float = 0.1,
            random: Callable = None
    ):
        if isinstance(domain, Domain):
            self.domain = domain
        else:
            self.domain = Domain(*domain)

        self.scorer = scorer or Scorer()

        self.fraction = fraction

        default_random = None
        default_score = None
        if (self.domain.type == int) and (self.domain.range != np.inf):
            default_random = self.default_random_selector_int_finite
            self.random_repr = 'Uniform'
        self.random = random or default_random
        if self.random is None:
            exit('Need a random for the ElementSpec initiation.')

    def default_random_selector_int_finite(self):
        return random.randrange(self.domain.range)

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
    return list(map(tuple, zip(*a)))


class TupleSpec:
    def __init__(
            self,
            element_specs: list,
            pool: int = 100,
            select: int = 10
    ):
        spec_list = list()
        for element_spec in element_specs:
            if isinstance(element_spec, ElementSpec):
                spec_list.append(element_spec)
            else:
                spec_list.append(ElementSpec(*element_spec))
        self.specs = tuple(spec_list)
        self.pool = pool
        self.select = select
        self.current = None
        self.pick_no = None
        self.current_score = None

    def random(self):
        selection = list()
        for spec in self.specs:
            reptition = round(spec.fraction * self.pool)
            n_repetitions = self.pool // reptition
            stub_length = self.pool % reptition
            random_selections = [[spec.random()] * reptition
                                 for _ in range(n_repetitions)]
            if stub_length > 0:
                random_selections += [spec.random()] * stub_length
            random_selections = flatten_shuffle(random_selections)[
                                : self.select]
            selection.append(random_selections)
        self.current = transpose_tuple(selection)
        return self.current

    def mix_and_pick(self):
        self.random()
        self.pick_no = random.randrange(self.select)
        pick_tuple = self.random()[pick_no]
        return self.pick_no, pick_tuple

    def __repr__(self):
        return f'TupleSpec({self.specs})'

    def score(self, grounds: tuple, guesses: tuple):
        #TODO Decide scoring method
        score = 0
        for spec, ground, guess in zip(self.specs, grounds, guesses):
            if ground == guess:
                score += spec.scorer.right
            else:
                score += spec.scorer.wrong
        self.current_score = score
        return self.current_score
