import torch


hyperparameters = {
    'N_ITERATIONS': 1000,  #000,
    'RANDOM_SEED': 42,
    'TORCH_RANDOM_SEED': 4242,
    'BATCHSIZE': 32,
    'N_SELECT': 10,
    'EPSILON_FLAT_END': 10,  # 1000
    'EPSILON_ZERO': 50,  # 10000
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
