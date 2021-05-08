# script to generate random seeds for use in config

import numpy as np


HIGH = 10 ** 6


rng = np.random.default_rng()
n = int(input('How many sets of four seeds?'))
seeds = rng.integers(HIGH, size=(n, 4))
for i, seed in enumerate(seeds):
    print(f'\n{tuple(seed)},', end='')
print('\b')
