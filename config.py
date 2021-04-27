import torch


hyperparameters = {
    'N_ITERATIONS': 500000,
    'RANDOM_SEED': 42,
    'TORCH_RANDOM_SEED': 4242,
    'ALICE_LAYERS': 3,
    'ALICE_WIDTH': 100,
    'BOB_LAYERS': 3,
    'BOB_WIDTH': 100,
    'BATCHSIZE': 32,
    'GAMESIZE': 1,  #1
    'BUFFER_CAPACITY': 20000,
    'START_TRAINING': 20000,  #20000
    'N_SELECT': 10,
    'EPSILON_ONE_END': 100000,
    'EPSILON_MIN': 0.01,
    'EPSILON_MIN_POINT': 350000,
    'ALICE_STRATEGY': 'one_per_bit',
    'BOB_STRATEGY': 'one_per_bit',
    'ALICE_OPTIMIZER': ('SGD', '{"lr": 0.01}'),
    'BOB_OPTIMIZER': 'Same',
    'ALICE_LOSS_FUNCTION': ('MSE', {}),
    'BOB_LOSS_FUNCTION': 'Same'
}


RANDOM_SEED = 42
TUPLE_SPEC = (
    ((int, 100000), ),
    ((int, 50000), (10, -1), ),
)
N_CODE = 8
# TRAINING_METHOD = 'q'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FOLDER = 'models'
CONFIGS_FOLDER = 'configs'
LOGS_FOLDER = 'logs'
