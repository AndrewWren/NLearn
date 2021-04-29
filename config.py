import torch


hyperparameters = {
    'N_ITERATIONS': 10000000,
    'RANDOM_SEED': 42,
    'TORCH_RANDOM_SEED': 4242,
    'ALICE_LAYERS': 3,
    'ALICE_WIDTH': 50,
    'BOB_LAYERS': 3,
    'BOB_WIDTH': 50,
    'BATCHSIZE': 32,
    'GAMESIZE': 32,
    'BUFFER_CAPACITY': 32 * 20000,
    'START_TRAINING': 32 * 20000,
    'N_SELECT': 5,
    'EPSILON_ONE_END': 64 * 20000,
    'EPSILON_MIN': 0.01,
    'EPSILON_MIN_POINT': 6000000,
    'ALICE_STRATEGY': 'one_per_code',
    'BOB_STRATEGY': 'one_per_bit',
    'ALICE_OPTIMIZER': #[('SGD', '{"lr": 0.3}'),
    ('SGD', '{"lr": 0.1}'),
                     # ('SGD', '{"lr": 0.01}'), ('SGD', '{"lr": 0.001}')],
                        # ('SGD', '{"lr": 0.01}'),
    'BOB_OPTIMIZER': #[('SGD', '{"lr": 0.3}'), ('SGD', '{"lr": 0.1}'),
                      ('SGD', '{"lr": 0.01}'), #('SGD', '{"lr": 0.001}')],
    # #'Same',
    'ALICE_LOSS_FUNCTION': ('MSE', {}),
    'BOB_LOSS_FUNCTION': 'Same'
}


RANDOM_SEED = 42
TUPLE_SPEC = (
    ((int, 200), ),
    ((int, 200), (10, -1), ),
)
N_CODE = 8
# TRAINING_METHOD = 'q'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FOLDER = 'models'
CONFIGS_FOLDER = 'configs'
LOGS_FOLDER = 'logs'
