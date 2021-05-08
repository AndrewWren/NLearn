import numpy as np


BINARY_LENGTH = h.N_CODE   # Change what BINARY_LENGTH is as needed

BIN_TEMPLATE = np.array([2 ** n for n in range(BINARY_LENGTH - 1, -1, -1)])
DEC_TEMPLATE = BIN_TEMPLATE / 2


def dec_2_bin(d_array):
    return 2 * (d_array & BIN_TEMPLATE) / BIN_TEMPLATE - 1


def bin_2_dec(b_array):
    return np.sum((b_array + 1) * DEC_TEMPLATE)
