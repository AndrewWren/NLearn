import random
import scr.ml_utilities as mlu
from scr.ml_utilities import c
import scr.net_class
from scr.tuple_and_code import Code, Domain, ElementSpec, TupleSpec


def define_tuple():
    return TupleSpec(c.TUPLE_SPEC)


def train_a(tuple_spec: ElementSpec, code: Code):
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
    #print(tuple_spec.random())
    print(tuple_spec)
    code = Code([1, 2, -7.3])
    #print(code)
