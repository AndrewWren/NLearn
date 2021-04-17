import random
import scr.ml_utilities as mlu
from scr.ml_utilities import c, h
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

@mlu.over_hp
def run_tuples():
    for iteration, pick_no, pick_tuple, current_tuples in tuple_spec.iter():
        print(iteration, pick_no, pick_tuple)
        print(current_tuples)
    return [None], 0


if __name__ == '__main__':
    random.seed(c.RANDOM_SEED)
    tuple_spec = define_tuple()
    print(tuple_spec)
    run_tuples()

    exit()
    code = Code([1, 2, -7.3, -5, 4, 3, -20.22, 3.145, -2.2, 10.])
    print(code)
