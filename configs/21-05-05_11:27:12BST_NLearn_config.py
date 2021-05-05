import torch


hyperparameters = {
    'N_ITERATIONS': 15 * (10 ** 4),
    'ITERATIONS_SEED': 100,
    'RANDOM_SEED': 42,
    'TORCH_RANDOM_SEED': 4242,
    'ALICE_LAYERS': 3,
    'ALICE_WIDTH': 50,
    'BOB_LAYERS': 3,
    'BOB_WIDTH': 50,
    'BATCHSIZE': 32,
    'GAMESIZE': 32,
    'BUFFER_CAPACITY': 32 * 20000,
    'START_TRAINING': 20000,
    'N_SELECT': 16,  # 5,
    'EPSILON_ONE_END': 40000,
    'EPSILON_MIN': 0.01,
    'EPSILON_MIN_POINT': 3 * (10 ** 5),
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
    'ALICE_PROXIMITY_BONUS': 30000,
    'ALICE_PROMIXITY_SLOPE_LENGTH': 10 ** 4,
    'ALICE_LAST_TRAINING': 100 * (10 ** 5)
}


RANDOM_SEED = 42
TUPLE_SPEC = (
    (16,),
)
N_CODE = 8
# TRAINING_METHOD = 'q'

SMOOTHING_LENGTH = 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FOLDER = 'models'
CONFIGS_FOLDER = 'configs'
LOGS_FOLDER = 'logs'
