import torch


hyperparameters = {
    'h1': None,
    'h2': None
}

RANDOM_SEED = 42
TYPE = int
RANGE = 100000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FOLDER = 'models'
CONFIGS_FOLDER = 'configs'
LOGS_FOLDER = 'logs'
