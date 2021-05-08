
hyperparameters = {  #TODO enable dictionary-based choices for finer choosing
    'N_ITERATIONS': 50 * (10 ** 3),   # 5 * (10 ** 5),
    'RANDOM_SEEDS': [
        (868291, 274344, 358840, 94453),
        (382832, 68444, 754888, 857796),
        (736520, 815195, 305871, 974216)
    ],
        #(81821, 18755, 28414, 111000),  # order n, ne, t, te
                      # (42, 100, 4242, 4242)
    'ALICE_NET': 'FFs(3, 50)',
    'BOB_LAYERS': 3,
    'BOB_WIDTH': 50,
    'BATCHSIZE': 32,
    'GAMESIZE': 32,
    'BUFFER_CAPACITY': 32 * 20000,
    'START_TRAINING': 20000,
    'N_SELECT': 16,
    'EPSILON_ONE_END': 2000,  #25000,  # 40000,
    'EPSILON_MIN': 0.01,
    'EPSILON_MIN_POINT': 40000,  #3 * (10 ** 5),
    'ALICE_PLAY': 'QPerCode',
    'ALICE_TRAIN': 'QPerCode',  # 'FromDecisions',
    'BOB_STRATEGY': 'circular_vocab',
    'ALICE_OPTIMIZER': [
                        'SGD(lr=0.01)'
                        ],
    'BOB_OPTIMIZER': [
                        ('SGD', '{"lr": 0.01}')
                        ],
    'ALICE_LOSS_FUNCTION': 'Huber(beta=0.5)',
    'BOB_LOSS_FUNCTION': ('torch.nn.MSE', {}), # 'Same',
    'ALICE_PROXIMITY_BONUS':  10 ** 8, # 30000 * (10 ** 3),
    'ALICE_PROXIMITY_SLOPE_LENGTH': 10000,
    'ALICE_LAST_TRAINING': 100 * (10 ** 5),
    'NOISE_START': 10 ** 8,
    'NOISE': 0.,
    'ALICE_DOUBLE': 500
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
