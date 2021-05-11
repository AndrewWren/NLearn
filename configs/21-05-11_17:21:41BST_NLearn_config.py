
hyperparameters = {  #TODO enable dictionary-based choices for finer choosing
    'N_ITERATIONS': 70 * (10 ** 3),   # 5 * (10 ** 5),
    'RANDOM_SEEDS': [
        (714844, 936892, 888616, 165835)  #,
        # (508585, 487266, 751926, 247136),
        # (843402, 443788, 742412, 270619),
        # (420915, 961830, 723900, 510954)
    ],
    'ALICE_NET': 'MaxNet("In", 3, 50)',  # 'FFs(3, 50)',
    'BOB_NET': 'FFs(3, 50)',
    'BATCHSIZE': 32,
    'GAMESIZE': 32,
    'BUFFER_CAPACITY': 32 * 20000,
    'START_TRAINING': 20000,
    'N_SELECT': 16,  # 256,  #16,
    'EPSILON_ONE_END': 2000,  #25000,  # 40000,
    'EPSILON_MIN': 0.0,
    'EPSILON_MIN_POINT': 20000,  #3 * (10 ** 5),
    'ALICE_PLAY': 'QPerCode',
    'ALICE_TRAIN': 'QPerCode',  # 'FromDecisions',
    'BOB_PLAY': 'CircularVocab',
    'BOB_TRAIN': 'CircularVocab',
    'ALICE_OPTIMIZER': 'SGD(lr=0.01)',
    'BOB_OPTIMIZER': 'SGD(lr=0.01)',
    'ALICE_LOSS_FUNCTION': 'Huber(beta=0.1)',
    'BOB_LOSS_FUNCTION': 'Huber(beta=0.1)',
    'ALICE_PROXIMITY_BONUS':  10 ** 8, # 30000 * (10 ** 3),
    'ALICE_PROXIMITY_SLOPE_LENGTH': 10000,
    'ALICE_LAST_TRAINING': 100 * (10 ** 5),
    'NOISE_START': 30000,
    'NOISE': 0.1,
    'ALICE_DOUBLE': None,
    'N_CODE': 8,
    'N_NUMBERS': 256
}


#  RANDOM_SEED = 42
"""TUPLE_SPEC = (
    (16,),
)
"""
SMOOTHING_LENGTH = 10000
SAVE_PERIOD = 10 ** 5


DEVICE = None

MODEL_FOLDER = 'models'
CONFIGS_FOLDER = 'configs'
LOGS_FOLDER = 'logs'
