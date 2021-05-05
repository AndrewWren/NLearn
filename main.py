import collections
import os
import random
import numpy as np
import torch
import scr.ml_utilities as mlu
from scr.ml_utilities import c, h, rng_c, to_array, to_device_tensor, writer
from scr.nets import LossInfo, Nets
from scr.game_set_up import Domain, ElementCircular, NiceCode, ReplayBuffer,\
    TupleSpecs


@mlu.over_hp
def train_ab():
    tuple_specs = TupleSpecs()
    mlu.log(f'{tuple_specs.random_reward_sd()=}')
    nets = Nets(tuple_specs)
    buffer = ReplayBuffer(h.BUFFER_CAPACITY)
    best_non_random_reward = -np.inf
    nrr_buffer = collections.deque(maxlen=c.SMOOTHING_LENGTH)
    best_nrr_iteration = None
    for game_origins in tuple_specs.iter():
        nets.current_iteration = game_origins.iteration
        non_random_rewards, game_reports = nets.play(game_origins)
        buffer.append(game_reports)
        nrr_buffer.append(non_random_rewards)
        if game_origins.iteration < h.START_TRAINING:
            if game_origins.iteration % 1000 == 0:
                print('\b' * 20 + f'Iteration={game_origins.iteration:>10}',
                      end='')
            continue
        alice_loss, bob_loss = nets.train(game_origins.iteration, buffer)
        if (alice_loss == np.nan) or (bob_loss == np.nan):
            return [(best_non_random_reward, best_nrr_iteration),
                    f'nan error at iteration={game_origins.iteration}'], 0
        if (game_origins.iteration % 100000 == 0) or (
                game_origins.iteration == h.N_ITERATIONS):
            mlu.save_model(nets.alice, title='Alice', parameter_name='iter',
                   parameter=game_origins.iteration)
            mlu.save_model(nets.bob, title='Bob', parameter_name='iter',
                       parameter=game_origins.iteration)
        if (game_origins.iteration % c.SMOOTHING_LENGTH == 0) and \
            ((nrr_buffer_n := np.concatenate(nrr_buffer)).shape[0] >=
                                c.SMOOTHING_LENGTH):
                if ((smoothed_nrr := np.mean(nrr_buffer_n))
                        > best_non_random_reward):
                    best_non_random_reward = smoothed_nrr
                    best_nrr_iteration = game_origins.iteration
    return [(- best_non_random_reward, best_nrr_iteration)], 0


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
              print_dict=True, print_full_dict=False):
    print(model_file)
    model_file = os.path.join(c.MODEL_FOLDER, model_file)
    model = torch.load(model_file)
    elt = ElementCircular(modulus, n_select)
    inputs = to_device_tensor(elt.domain)
    outputs = to_array(model(inputs).squeeze())
    codes = np.sign(outputs)
    return codes, print_book(codes, outputs, print_dict, print_full_dict,
                        print_list)


@torch.no_grad()
def code_decode_book(model_file_alice, model_file_bob, modulus, n_select):
    print(f'{model_file_alice=}\n{model_file_bob=}')
    _, code_dict = code_book(model_file_alice, modulus, n_select)
    model_file_bob = os.path.join(c.MODEL_FOLDER, model_file_bob)
    model_bob = torch.load(model_file_bob)
    elt = ElementCircular(modulus, n_select)
    domain = to_device_tensor(elt.domain)
    decode_dict = dict()
    for nice_code in code_dict.keys():
        code_repeated = to_device_tensor(nice_code.raw()).repeat(modulus, 1)
        bob_input = torch.cat(
            [domain, code_repeated], 1
        )
        bob_q_estimates = model_bob(bob_input).squeeze()
        decode_dict[nice_code] = torch.argmax(bob_q_estimates).item()
    print()
    for nice_code in decode_dict:
        print(f'{nice_code}\t{decode_dict[nice_code]}')
    print()


def print_book(codes, outputs=None, print_dict=True, print_full_dict=False,
               print_list=True):
    if outputs is None:
        outputs = np.empty(codes.shape)
    code_dict = dict()
    for i, (code, output) in enumerate(zip(codes, outputs)):
        nice_code = NiceCode(code)
        if print_list:
            print(f'{i}\t{nice_code}\t{output}')
        if print_full_dict:
            try:
                code_dict[nice_code].append((i, output))
            except:
                code_dict[nice_code] = [(i, output)]
        elif print_dict:
            try:
                code_dict[nice_code].append(i)
            except:
                code_dict[nice_code] = [i]
    if (print_dict or print_full_dict):
        print()
        for nice_code in code_dict:
            print(f'{nice_code}\t{code_dict[nice_code]}')
    print()
    return code_dict

def code_books_0():
    for hp_run in range(1, 4 + 1):
        code_book(f'21-05-02_17:29:40BST_NLearn_model_'
                  f'{hp_run}_Alice_iter500000',
                  16, 5)


if __name__ == '__main__':
    #run_tuples()
    train_ab()
    #code_books_0()
    """code_book('21-05-03_20:36:57BST_NLearn_model_2_Alice_iter500000', 16,
              16, print_full_dict=True)
    """
    """code_decode_book('21-05-04_16:11:54BST_NLearn_model_1_Alice_iter12500',
                     '21-05-04_16:11:54BST_NLearn_model_1_Bob_iter12500',
                     16, 16)
    """
    mlu.close_log()
