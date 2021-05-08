import torch
from src.lib.ml_utilities import c, h


DEVICE = c.DEVICE # Change what DEVICE is as needed

def initialise_bin_dec():
    global BIN_TEMPLATE, DEC_TEMPLATE
    BIN_TEMPLATE = torch.LongTensor([2 ** n for n in range(
        h.N_CODE - 1,  -1, -1)]).to(DEVICE)
    DEC_TEMPLATE = BIN_TEMPLATE / 2


def dec_2_bin(d_tensor):
    d_tensor = d_tensor.unsqueeze(-1).repeat(1, h.N_CODE)
    return 2 * torch.bitwise_and(BIN_TEMPLATE, d_tensor) / BIN_TEMPLATE \
           - 1


def bin_2_dec(b_tensor):
    return torch.sum(((b_tensor + 1) * DEC_TEMPLATE), dim=-1).long()
