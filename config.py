
hyperparameters = {  #TODO enable dictionary-based choices for finer choosing
    'N_ITERATIONS':  5 * (10 ** 5),
    'RANDOM_SEEDS': (81821, 18755, 28414, 111000),  # order n, ne, t, te
                      # (42, 100, 4242, 4242)
    'ALICE_NET': 'FFs(3, 50)',
    'BOB_LAYERS': 3,
    'BOB_WIDTH': 50,
    'BATCHSIZE': 32,
    'GAMESIZE': 32,
    'BUFFER_CAPACITY': 32 * 20000,
    'START_TRAINING': 20000,
    'N_SELECT': 16,
    'EPSILON_ONE_END': 40000,
    'EPSILON_MIN': 0.01,
    'EPSILON_MIN_POINT': 3 * (10 ** 5),
    'ALICE_PLAY': 'Basic',
    'ALICE_TRAIN': 'Basic',  # 'FromDecisions',
    'BOB_STRATEGY': 'circular_vocab',
    'ALICE_OPTIMIZER': [
                        'SGD(lr=0.32)'
                        ],
    'BOB_OPTIMIZER': [
                        ('SGD', '{"lr": 0.01}')
                        ],
    'ALICE_LOSS_FUNCTION': 'Huber(beta=0.1)',
    'BOB_LOSS_FUNCTION': ('torch.nn.MSE', {}), # 'Same',
    'ALICE_PROXIMITY_BONUS': 30000 * (10 ** 3),
    'ALICE_PROXIMITY_SLOPE_LENGTH': 10000,
    'ALICE_LAST_TRAINING': 100 * (10 ** 5),
    'NOISE_START': 10 ** 8,
    'NOISE': 0.,
    'ALICE_DOUBLE': None
}


#  RANDOM_SEED = 42
TUPLE_SPEC = (
    (16,),
)
N_CODE = 8
# TRAINING_METHOD = 'q'

SMOOTHING_LENGTH = 10000
SAVE_PERIOD = 10 ** 5


DEVICE = None

MODEL_FOLDER = 'models'
CONFIGS_FOLDER = 'configs'
LOGS_FOLDER = 'logs'
