import torch
from src.ml_utilities import c


BINARY_LENGTH = c.N_CODE  # Change what BINARY_LENGTH is as needed
DEVICE = c.DEVICE # Change what DEVICE is as needed

BIN_TEMPLATE = torch.LongTensor([2 ** n for n in range(BINARY_LENGTH - 1, -1,
                                                        -1)]).to(DEVICE)
DEC_TEMPLATE = BIN_TEMPLATE / 2


def dec_2_bin(d_tensor):
    d_tensor = d_tensor.unsqueeze(-1).repeat(1, BINARY_LENGTH)
    return 2 * torch.bitwise_and(BIN_TEMPLATE, d_tensor) / BIN_TEMPLATE \
           - 1


def bin_2_dec(b_tensor):
    return torch.sum(((b_tensor + 1) * DEC_TEMPLATE), dim=-1).long()
