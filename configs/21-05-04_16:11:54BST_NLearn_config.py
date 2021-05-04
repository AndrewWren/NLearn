import torch


hyperparameters = {
    'N_ITERATIONS': 12500,  # 5 * (10 ** 5),
    'RANDOM_SEED': 42,
    'TORCH_RANDOM_SEED': 4242,
    'ALICE_LAYERS': 3,
    'ALICE_WIDTH': 50,
    'BOB_LAYERS': 3,
    'BOB_WIDTH': 50,
    'BATCHSIZE': 32,
    'GAMESIZE': 32,
    'BUFFER_CAPACITY': 32 * 20000,
    'START_TRAINING': 2000, # 20000,
    'N_SELECT': 16,  # 5,
    'EPSILON_ONE_END': 2000,  #40000,
    'EPSILON_MIN':   0.5,  # to write up, [0.01, 0.1],
    'EPSILON_MIN_POINT': 10000,  # 3 * (10 ** 5),
    'ALICE_STRATEGY': 'from_decisions',
    'BOB_STRATEGY': 'circular_vocab',
    'ALICE_OPTIMIZER': [
                        ('SGD', '{"lr": 0.01}')
                        ],
    'BOB_OPTIMIZER': [
                        ('SGD', '{"lr": 0.01}')
                        ],
    'ALICE_LOSS_FUNCTION': ('MSEBits', {}), # ('MSE', {}),
    'BOB_LOSS_FUNCTION': 'Same',
    'ALICE_LAST_TRAINING': 3 * (10 ** 5),  # to write up [2 * (10 ** 5),
    # 3 * (10 ** 5)],
    'ALICE_PROXIMITY_BONUS': 10 ** 7
}


RANDOM_SEED = 42
TUPLE_SPEC = (
    (16,),
)
N_CODE = 8
SMOOTHING_LENGTH = 1000  # 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FOLDER = 'models'
CONFIGS_FOLDER = 'configs'
LOGS_FOLDER = 'logs'
