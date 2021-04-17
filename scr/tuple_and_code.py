from collections.abc import Callable
import random
import numpy as np


class Code:
    def __init__(self, value: list):
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


class Score:
    def __init__(self, right=1, wrong=0):
        self.right = right
        self.wrong = wrong


class TupleSpec:
    def __init__(
            self,
            domain: Domain,
            random: Callable = None,
            score: Score = None
    ):
        self.domain = domain
        default_random = None
        default_score = None
        if (self.domain.type == int) and (self.domain.range != np.inf):
            default_random = self.default_random_selector_int_finite
        #default_score =
        self.random = random or default_random
        if self.random is None:
            exit('Need a random for the TupleSpec initiation.')
        self.score = score or Score()

    def default_random_selector_int_finite(self):
            return random.randrange(self.domain.range)
