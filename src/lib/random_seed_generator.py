# script to generate random seeds for use in config

import numpy as np


HIGH = 10 ** 6


rng = np.random.default_rng()
n = int(input('How many sets of four seeds?'))
seeds = rng.integers(HIGH, size=(n, 4))
print('[')
for i, seed in enumerate(seeds):
    if i > 0:
        print('')
    print(f'{tuple(seed)},', end='')
print('\b\n],')
