import os
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


@torch.no_grad()
def code_book(model_file, modulus, n_select, print_list=False,
              print_dict=True):
    print(model_file)
    model_file = os.path.join(c.MODEL_FOLDER, model_file)
    model = torch.load(model_file)
    elt = ElementCircular(modulus, n_select)
    inputs = torch.tensor(elt.domain).to(c.DEVICE).float()
    outputs = model(inputs).squeeze()
    codes = torch.sign(outputs)
    code_dict = dict()
    for i, code in enumerate(codes):
        nice_code = NiceCode(code)
        if print_list:
            print(f'{i}\t{nice_code}')
        if print_dict:
            if nice_code in code_dict:
                code_dict[nice_code].append(i)
            else:
                code_dict[nice_code] = [i]
    if print_dict:
        print()
        for key in code_dict:
            print(f'{key}\t{code_dict[key]}')
    print()

if __name__ == '__main__':
    #run_tuples()
    #train_ab()
    for hp_run in range(1, 4 + 1):
        code_book(f'21-05-02_17:29:40BST_NLearn_model_'
                  f'{hp_run}_Alice_iter500000',
                  16, 5)
    mlu.close_log()
