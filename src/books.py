import numpy as np
import torch
import src.ml_utilities as mlu
from src.ml_utilities import c, h, to_array, to_device_tensor, writer
from src.game_set_up import ElementCircular, NiceCode


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
def code_book(model, modulus, n_select, print_list=False,
              print_dict=True, print_full_dict=False):
    elt = ElementCircular(modulus, n_select)
    inputs = to_device_tensor(elt.domain)
    outputs = to_array(model(inputs).squeeze())
    codes = np.sign(outputs)
    return codes, log_book(codes, outputs, print_dict, print_full_dict,
                        print_list)


@torch.no_grad()  #TODO print to mlu
def code_decode_book(model_alice, model_bob, modulus, n_select):
    mlu.log()
    _, code_dict = code_book(model_alice, modulus, n_select)
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
    mlu.log()
    for nice_code in decode_dict:
        mlu.log(f'{nice_code}\t{decode_dict[nice_code]}')
    mlu.log()
    mlu.log()
