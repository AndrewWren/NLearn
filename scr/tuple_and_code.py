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
            random: Callable = None
    ):
        if isinstance(domain, Domain):
            self.domain = domain
        else:
            self.domain = Domain(*domain)

        self.scorer = scorer or Scorer()

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


class TupleSpec:
    def __init__(self, element_specs):
        spec_list = list()
        for element_spec in element_specs:
            if isinstance(element_spec, ElementSpec):
                spec_list.append(element_spec)
            else:
                spec_list.append(ElementSpec(*element_spec))
        self.specs = tuple(spec_list)

    def random(self):
        return tuple([spec.random() for spec in self.specs])

    def __repr__(self):
        return f'TupleSpec({self.specs})'

    def score(self, ground: tuple, guess: tuple):
        #TODO Decide scoring method and its implications for choosing
        # wrongs
        if ground == guess:
            return sum([spec.scorer.right in self.specs])
        else:
            return sum([spec.scorer.wrong in self.specs])
