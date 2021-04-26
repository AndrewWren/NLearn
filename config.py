import torch


hyperparameters = {
    'N_ITERATIONS': 1000000,  # 100000,
    'RANDOM_SEED': 42,
    'TORCH_RANDOM_SEED': 4242,
    'BATCHSIZE': 32,
    'GAMESIZE': 1,
    'BUFFER_CAPACITY': 10000,
    'N_SELECT': 10,
    'EPSILON_ONE_END': 100000,  # 1000
    'EPSILON_MIN': 0.01,
    'EPSILON_MIN_POINT': 900000,
    'ALICE_OPTIMIZER': ('SGD', '{"lr": 0.1}'),
    'BOB_OPTIMIZER': ('SGD', '{"lr": 10000100000000000000.}'),
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
