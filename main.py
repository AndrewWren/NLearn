import collections
import numpy as np
import src.lib.ml_utilities as mlu
from src.lib.ml_utilities import c, h
from src.session import Session
from src.game_set_up import ReplayBuffer,\
    SessionSpec


@mlu.over_hp
def train_ab():
    for key, value in h.items():
        if ('BOB' in key) and (value == 'Same'):
            h[key] = h[key.replace('BOB', 'ALICE')]
    session_spec = SessionSpec()
    mlu.log(f'{session_spec.random_reward_sd()=}\n')
    session = Session(session_spec)
    buffer = ReplayBuffer(h.BUFFER_CAPACITY)
    best_non_random_reward = - np.inf
    nrr_buffer = collections.deque(maxlen=c.SMOOTHING_LENGTH)
    best_nrr_iteration = None
    saved_alice_model_title = None
    saved_bob_model_title = None
    for game_origins in session_spec.iter():
        session.current_iteration = game_origins.iteration
        non_random_rewards, game_reports = session.play(game_origins)
        buffer.append(game_reports)
        nrr_buffer.append(non_random_rewards)
        if game_origins.iteration < h.START_TRAINING:
            if game_origins.iteration % 1000 == 0:
                print('\b' * 20 + f'Iteration={game_origins.iteration:>10}',
                      end='')
            continue
        alice_loss, bob_loss = session.train(game_origins.iteration, buffer)
        if (alice_loss == np.nan) or (bob_loss == np.nan):
            return [(best_non_random_reward, best_nrr_iteration),
                    (saved_alice_model_title, saved_bob_model_title),
                    f'nan error at iteration={game_origins.iteration}'], 0
        if (game_origins.iteration % c.SAVE_PERIOD == 0) or (
                game_origins.iteration == h.N_ITERATIONS):
            saved_alice_model_title = mlu.save_model(
                session.alice.net,
                title='Alice',
                parameter_name='iter',
                parameter=game_origins.iteration
            )
            saved_bob_model_title = mlu.save_model(
                session.bob.net,
                title='Bob',
                parameter_name='iter',
                parameter=game_origins.iteration
            )
        if (game_origins.iteration % c.SMOOTHING_LENGTH == 0) and \
            ((nrr_buffer_n := np.concatenate(nrr_buffer)).shape[0] >=
                                c.SMOOTHING_LENGTH):
                if ((smoothed_nrr := np.mean(nrr_buffer_n))
                        > best_non_random_reward):
                    best_non_random_reward = smoothed_nrr
                    best_nrr_iteration = game_origins.iteration
    return [(- best_non_random_reward, best_nrr_iteration),
            (saved_alice_model_title, saved_bob_model_title)], 0


def test_ar():
    pass


def understand():
    pass


if __name__ == '__main__':
    full_results = train_ab()
    mlu.close_log()
