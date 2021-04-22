import random
import numpy as np
import scr.ml_utilities as mlu
from scr.ml_utilities import c, h, rng_c
from scr.nets import FFs, Nets
from scr.tuple_and_code import Code, Domain, ElementSpec, TupleSpecs


@mlu.over_hp
def train_ar(tuple_spec: TupleSpecs):
    tuple_spec = TupleSpecs()
    nets = Nets(tuple_spec)
    buffer = list()
    for game_origin_for_buffer in tuple_spec.iter():
        game_report_for_buffer = nets.play(game_origin_for_buffer)
        buffer.append(game_report_for_buffer)
        if len(buffer) < h.BUFFER_LENGTH:
            continue
        buffer_entry_for_training = random.randrange(h.BUFFER_LENGTH)
        game_report_for_training = buffer.pop(buffer_entry_for_training)
        nets.train(game_report_for_training)
    return [None], 0
#target_tuple = current_tuples[target_no]


def test_ar():
    pass


def understand():
    pass


@mlu.over_hp
def run_tuples():
    tuple_spec = TupleSpecs()
    for iteration, pick_no, current_tuples in tuple_spec.iter():
        pick_tuple = current_tuples[np.arange(32), pick_no, :]
        print(iteration, pick_no)
        print(f'{pick_tuple.shape=}')
        print(f'{current_tuples.shape=}')
    return [None], 0


if __name__ == '__main__':
    random.seed(c.RANDOM_SEED)
    run_tuples()
    """code = Code([1, 2, -7.3, -5, 4, 3, -20.22, 3.145, -2.2, 10.])
    print(code)
    """
