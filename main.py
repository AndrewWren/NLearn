import random
import numpy as np
import torch
import scr.ml_utilities as mlu
from scr.ml_utilities import c, h, rng_c
from scr.nets import FFs, Nets
from scr.tuple_and_code import Code, Domain, ElementSpec, TupleSpecs


@mlu.over_hp
def train_ab():
    tuple_spec = TupleSpecs()
    nets = Nets(tuple_spec)
    buffer = list()
    for game_origin in tuple_spec.iter():
        if (iteration := game_origin.iteration) % 1000 == 0:
            print('\b' * 20 + f'Iteration={iteration:>10}', end='')
        game_report = nets.play(game_origin)
        buffer.append(game_report)
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
    #run_tuples()
    train_ab()
    """code = Code([1, 2, -7.3, -5, 4, 3, -20.22, 3.145, -2.2, 10.])
    print(code)
    """
    mlu.close_log()
