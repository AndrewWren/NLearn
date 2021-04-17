import random
import scr.ml_utilities as mlu
from scr.ml_utilities import c, h
from scr.nets import NetA, NetR
from scr.tuple_and_code import Code, Domain, ElementSpec, TupleSpec


@mlu.over_hp
def train_ar(tuple_spec: TupleSpec):
    net_a = NetA()
    net_r = NetR()
    for iteration, pick_no, current_tuples in tuple_spec.iter():
        pick_tuple = current_tuples[pick_no]
        #TODO Buffer
        #TODO Train once get to iteration >= h.COLD
        continue
    return [None], 0


def test_ar():
    pass


def understand():
    pass


@mlu.over_hp
def run_tuples():
    for iteration, pick_no, current_tuples in tuple_spec.iter():
        pick_tuple = current_tuples[pick_no]
        print(iteration, pick_no, pick_tuple)
        print(current_tuples)
    return [None], 0


if __name__ == '__main__':
    random.seed(c.RANDOM_SEED)
    tuple_spec = TupleSpec()
    print(tuple_spec)
    run_tuples()
    #train_ar(tuple_spec)
    """run_tuples()

    code = Code([1, 2, -7.3, -5, 4, 3, -20.22, 3.145, -2.2, 10.])
    print(code)
    """
