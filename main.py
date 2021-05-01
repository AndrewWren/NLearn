import random
import numpy as np
import torch
import scr.ml_utilities as mlu
from scr.ml_utilities import c, h, rng_c
from scr.nets import LossInfo, Nets
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
        game_reports = nets.play(game_origins)
        buffer.append(game_reports)
        if game_origins.iteration < h.START_TRAINING:
            if game_origins.iteration % 1000 == 0:
                print('\b' * 20 + f'Iteration={game_origins.iteration:>10}',
                      end='')
            continue
        alice_loss, bob_loss = nets.train(game_origins.iteration, buffer)
        if (game_origins.iteration % 100000 == 0) or (
                game_origins.iteration == h.N_ITERATIONS):
            mlu.save_model(nets.alice, title='Alice', parameter_name='iter',
                   parameter=game_origins.iteration)
            mlu.save_model(nets.bob, title='Bob', parameter_name='iter',
                       parameter=game_origins.iteration)
        if (alice_loss == np.nan) or (bob_loss == np.nan):
            return [np.nan], 0
        if bob_loss < best_bob_loss.bob_loss:
            best_bob_loss = LossInfo(bob_loss, game_origins.iteration,
                                     alice_loss)
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
