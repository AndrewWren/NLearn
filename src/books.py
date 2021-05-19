import numpy as np
import torch
import src.lib.ml_utilities as mlu
from src.lib.ml_utilities import h, to_array, to_device_tensor
from src.game_set_up import Basic, NiceCode


def log_book(codes, outputs=None, log_dict=True, log_full_dict=False,
             log_list=False):
    if outputs is None:
        outputs = np.empty(codes.shape)
    code_dict = dict()
    for i, (code, output) in enumerate(zip(codes, outputs)):
        nice_code = NiceCode(code)
        if log_list:
            mlu.log(f'{i}\t{nice_code}\t{output}')
        if log_full_dict:
            try:
                code_dict[nice_code].append((i, output))
            except:
                code_dict[nice_code] = [(i, output)]
        elif log_dict:
            try:
                code_dict[nice_code].append(i)
            except:
                code_dict[nice_code] = [i]
    if (log_dict or log_full_dict):
        for nice_code in code_dict:
            mlu.log(f'{nice_code}\t{code_dict[nice_code]}')
    return code_dict


@torch.no_grad()
def code_book(alice, print_list=False,
              print_dict=True, print_full_dict=False):
    alice.session.targets_t = alice.session.session_spec.spec.numbers_t
    outputs = to_array(alice.play().squeeze())
    codes = np.sign(outputs)
    return codes, log_book(codes, outputs, print_dict, print_full_dict,
                        print_list)


@torch.no_grad()
def code_decode_book(alice, bob):
    mlu.log()
    _, code_dict = code_book(alice)
    if (h.N_SELECT == h.N_NUMBERS) or (h.BOB_PLAY == 'CircularVocab'):
        domain = bob.session.session_spec.spec.numbers_t
        decode_dict = dict()
        bob.session.size0 = 1
        bob.session.n_select = h.N_NUMBERS
        for nice_code in code_dict.keys():
            bob.session.codes = to_device_tensor(nice_code.raw()).unsqueeze(0)

            bob.session.selections = domain.unsqueeze(0)
            #domain.unsqueeze(1).unsqueeze(0)
            decode_dict[nice_code] = bob.play().squeeze().item()
        bob.session.n_select = h.N_SELECT
        mlu.log()
        for nice_code in decode_dict:
            decode = decode_dict[nice_code]
            status = (decode in code_dict[nice_code])
            mlu.log(f'{nice_code}\t{decode}\t\t{status}')
    mlu.log()
    mlu.log(f'Number of codes used={len(code_dict)}')
    mlu.log()
