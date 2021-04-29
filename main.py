import random
import numpy as np
import torch
import scr.ml_utilities as mlu
from scr.ml_utilities import c, h, rng_c
from scr.nets import FFs, LossInfo, Nets
from scr.game_set_up import Domain, ElementCircular, NiceCode, ReplayBuffer,\
    TupleSpecs


@mlu.over_hp
def train_ab():
    tuple_specs = TupleSpecs()
    mlu.log(f'{tuple_specs.random_reward_sd()=}')
    nets = Nets(tuple_specs)
    buffer = ReplayBuffer(h.BUFFER_CAPACITY)
    best_bob_loss = LossInfo(np.inf, None, None)
    for game_origins in tuple_specs.iter():
        if (iteration := game_origins.iteration) % 1000 == 0:
            print('\b' * 20 + f'Iteration={iteration:>10}', end='')
        game_reports = nets.play(game_origins)
        buffer.append(game_reports)
        if iteration < h.START_TRAINING:
            continue
        alice_loss, bob_loss = nets.train(iteration, buffer)
        if (alice_loss == np.nan) or (bob_loss == np.nan):
            return [np.nan], 0
        if bob_loss < best_bob_loss.bob_loss:
            best_bob_loss = LossInfo(bob_loss, iteration, alice_loss)
    return [best_bob_loss], 0
#target_tuple = current_tuples[target_nos]


def test_ar():
    pass


def understand():
    pass


@mlu.over_hp
def run_tuples():
    tuple_specs = TupleSpecs()
    for iteration, pick_no, current_tuples in tuple_specs.iter():
        pick_tuple = current_tuples[np.arange(32), pick_no, :]
        print(iteration, pick_no)
        print(f'{pick_tuple.shape=}')
        print(f'{current_tuples.shape=}')
    return [None], 0


if __name__ == '__main__':
    #run_tuples()
    train_ab()
    """code = NiceCode([1, 2, -7.3, -5, 4, 3, -20.22, 3.145, -2.2, 10.])
    print(code)
    """
    mlu.close_log()
