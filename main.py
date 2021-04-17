import random
import scr.ml_utilities as mlu
from scr.ml_utilities import c
import scr.net_class
from scr.subject_tuple import Domain, TupleSpec


def define_tuple():
    domain = Domain(c.TYPE, c.RANGE)
    return TupleSpec(domain)


def train_a():
    pass


def train_r():
    pass


def test_ar():
    pass


def understand():
    pass


if __name__ == '__main__':
    random.seed(c.RANDOM_SEED)
    tuple_spec = define_tuple()
    print(tuple_spec.random())
