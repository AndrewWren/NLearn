import collections
import os
import random
import numpy as np
import torch
import src.books as books
import src.ml_utilities as mlu
from src.ml_utilities import c, h, to_array, to_device_tensor, writer
from src.session import LossInfo, Session
from src.game_set_up import Domain, ElementCircular, NiceCode, ReplayBuffer,\
    TupleSpecs


@mlu.over_hp
def train_ab():
    for key, value in h.items():
        if ('BOB' in key) and (value == 'Same'):
            h[key] = h[key.replace('BOB', 'ALICE')]
    tuple_specs = TupleSpecs()
    mlu.log(f'{tuple_specs.random_reward_sd()=}')
    session = Session(tuple_specs)
    buffer = ReplayBuffer(h.BUFFER_CAPACITY)
    best_non_random_reward = - np.inf
    nrr_buffer = collections.deque(maxlen=c.SMOOTHING_LENGTH)
    best_nrr_iteration = None
    saved_alice_model_title = None
    saved_bob_model_title = None
    for game_origins in tuple_specs.iter():
        session.current_iteration = game_origins.iteration
        non_random_rewards, game_reports = session.play(game_origins)
        buffer.append(game_reports)
        nrr_buffer.append(non_random_rewards)
        if game_origins.iteration < h.START_TRAINING:
            if game_origins.iteration % 1000 == 0:
                print('\b' * 20 + f'Iteration={game_origins.iteration:>10}',
                      end='')
            continue
        alice_loss, bob_loss = session.train(game_origins.iteration, buffer)
        if (alice_loss == np.nan) or (bob_loss == np.nan):
            return [(best_non_random_reward, best_nrr_iteration),
                    (saved_alice_model_title, saved_bob_model_title),
                    f'nan error at iteration={game_origins.iteration}'], 0
        if (game_origins.iteration % c.SAVE_PERIOD == 0) or (
                game_origins.iteration == h.N_ITERATIONS):
            saved_alice_model_title = mlu.save_model(session.alice.net,
                                                     title='Alice',
                               parameter_name='iter',
                   parameter=game_origins.iteration)
            saved_bob_model_title = mlu.save_model(session.bob, title='Bob',
                                           parameter_name='iter',
                       parameter=game_origins.iteration)
        if (game_origins.iteration % c.SMOOTHING_LENGTH == 0) and \
            ((nrr_buffer_n := np.concatenate(nrr_buffer)).shape[0] >=
                                c.SMOOTHING_LENGTH):
                if ((smoothed_nrr := np.mean(nrr_buffer_n))
                        > best_non_random_reward):
                    best_non_random_reward = smoothed_nrr
                    best_nrr_iteration = game_origins.iteration
    return [(- best_non_random_reward, best_nrr_iteration),
            (saved_alice_model_title, saved_bob_model_title)], 0


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
    full_results = train_ab()
    for full_result in full_results:
        saved_alice_model_title, saved_bob_model_title = full_result[1]
        books.code_decode_book(
            mlu.load_model(saved_alice_model_title),
            mlu.load_model(saved_bob_model_title),
            16,
            16
        )  #TODO Automate those 16s

    #code_books_0()
    """code_book('21-05-03_20:36:57BST_NLearn_model_2_Alice_iter500000', 16,
              16, print_full_dict=True)
    """
    """code_decode_book('21-05-05_15:59:30BST_NLearn_model_1_Alice_iter150000',
                     '21-05-05_15:59:30BST_NLearn_model_1_Bob_iter150000',
                     16, 16)
    """
    mlu.close_log()
