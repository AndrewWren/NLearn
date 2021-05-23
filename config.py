
divisor = 1.

hyperparameters = {  #TODO enable dictionary-based choices for finer choosing
    'N_ITERATIONS': 1000 * (10 ** 3),  #int(70 * (10 ** 3) // divisor),
                  # 5 * (10 **
    # 5),
    'RANDOM_SEEDS': [
        # (789873, 935177, 972236, 435766),
        (532334, 809631, 735618, 545983),
        (321406, 416695, 885201, 467036)  #,
        # (911011, 571019, 667157, 225093),
        # (335581, 265392, 137411, 842014),
        # (307035, 405050, 968633, 690674),
        # (577683, 443890, 562139, 319257),
        # (625084, 419126, 762692, 952720)
    ],
    'ALICE_NET': 'FFs(3, 50)',
    'BOB_NET': 'FFs(3, 50)',
    'BATCHSIZE': 32,
    'GAMESIZE': 32,
    'BUFFER_CAPACITY': int(32 * 20000 // divisor),
    'START_TRAINING': int(20000 // divisor),
    'N_SELECT': 256,  #16,
    'EPSILON_ONE_END': 50000,  #int(2000 // divisor),  #25000,  # 40000,
    'EPSILON_MIN': 0.01,
    'EPSILON_MIN_POINT': 600000,  #int(20000 // divisor),  #3 * (10 **
                  # 5),
    'ALICE_PLAY': 'QPerCode',
    'ALICE_TRAIN': 'QPerCode',
    'BOB_PLAY': 'CircularVocab',
    'BOB_TRAIN': 'CircularVocab',
    'ALICE_OPTIMIZER': 'SGD(lr=0.01)',
    'BOB_OPTIMIZER': 'SGD(lr=0.01)',
    'ALICE_LOSS_FUNCTION': 'Huber(beta=0.1)',
    'BOB_LOSS_FUNCTION': 'Huber(beta=0.1)',
    'ALICE_PROXIMITY_BONUS':  10 ** 8, # 30000 * (10 ** 3),
    'ALICE_PROXIMITY_SLOPE_LENGTH': 10000,
    'ALICE_LAST_TRAINING': 100 * (10 ** 5),
    'NOISE_START': int(30000 // divisor),
    'NOISE': 0,  #0.1,
    'NOISE_END': 48000,  #None,
    'ALICE_DOUBLE': None,
    'N_CODE': 8,
    'NUMBERS': 'Basic',
    'N_NUMBERS': 256,  #2 ** 14,
    'SHUFFLE': True,
    'REWARD_TYPE': 'Exact only'  #'Near misses only'
}


#  RANDOM_SEED = 42
SMOOTHING_LENGTH = int(10000 // divisor)
SAVE_PERIOD = 10 ** 5   # Also saves on the last iteration
CODE_BOOK_PERIOD = int(10000 // divisor)


DEVICE = None

MODEL_FOLDER = 'models'
CONFIGS_FOLDER = 'configs'
LOGS_FOLDER = 'logs'
