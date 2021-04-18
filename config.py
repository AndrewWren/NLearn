import torch


hyperparameters = {
    'N_ITERATIONS': 10,
    'BUFFER_LENGTH': 100
}


RANDOM_SEED = 42
TUPLE_SPEC = (
    ((int, 100000), ),
    ((int, 50000), (10, -1), ),
)
N_CODE = 10
TRAINING_METHOD = 'q'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FOLDER = 'models'
CONFIGS_FOLDER = 'configs'
LOGS_FOLDER = 'logs'
