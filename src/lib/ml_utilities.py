import datetime
from itertools import product
import json
import numpy as np
import os
from os import path
import platform
import shutil
import sys
from time import perf_counter, strftime
from dateutil.tz import tzlocal
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import config as c


"""Note: DO NOT use h. hyperparameters as default arguments in any module --- 
as they are only evaluated when the function is first defined.
c. constants can be used as default arguments.
"""

c.DEVICE = c.DEVICE or torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
STD_NOW = datetime.datetime.now(tz=tzlocal()).strftime("%y-%m-%d_%H:%M:%S%Z")
log_lines = ['****WARNING NOT CLOSED - MAY BE DUE TO ERROR***\n'] * 2
writer = SummaryWriter()


# See https://exceptionshub.com/how-to-get-filename-of-the-__main__-module-in-python.html
def main_folder():
    return path.basename(
        path.dirname(path.abspath(sys.modules['__main__'].__file__)))


# timezone stuff from
# https://stackoverflow.com/questions/35057968/get-system-local-timezone-in-python
def time_stamp(title: str, folder: str = None,
               the_now: str = STD_NOW) -> str:
    """ Returns a string giving the filepath with title preceded by the
    the_now's date, time and timezone, and (if folder is not None) in the
    folder.

    However, if the_now == None, then use the selections date, time, timezone.
    """
    the_now = the_now or datetime.datetime.now(tz=tzlocal()).strftime(
        "%y-%m-%d_%H:%M:%S%Z")
    filename = the_now + '_' + title
    if folder == None:
        return filename
    else:
        return os.path.join(folder, filename)


PROJECT_NAME = main_folder()
LOG_FILENAME = time_stamp(PROJECT_NAME + '.log', c.LOGS_FOLDER)
LOG_WARNING_FILENAME = time_stamp(PROJECT_NAME + 'WARNING.log',
                                  c.LOGS_FOLDER)


def value_code(value, save=True, to_json=False):
    if isinstance(value, torch.device):
        value = 'torch.device("cuda" if torch.cuda.is_available() else ' \
                '"cpu")  # Used: \"' + str(value) + '\"'
        return value
    elif callable(value):
        value = value.__module__ + '.' + value.__name__
        return value
    elif not isinstance(value, str):
        return value
    if save and (not to_json):
        return '\'' + value + '\''
    else:
        return value


class Constants():
    def __init__(self):
        self.constants = {key: value for key, value in
                          vars(c).items() if key[0].isupper()}
        self.hyperparameters = c.hyperparameters
        """if 'CHRONOLOGY' not in self.constants.keys():
            self.constants['CHRONOLOGY'] = list()
        self.constants['CHRONOLOGY'].append(STD_NOW)
        """

    def print(self):
        for key, value in self.constants.items():
            print(f'{key} = {value_code(value, save=False)}')
        print('\nhyperparameters = {')
        for key, value in self.hyperparameters.items():
            print(f'\t{key}: {value_code(value, save=False)}, ')
        print('}\n')

    def log(self):
        for key, value in self.constants.items():
            log(f'{key} = {value_code(value)}')
        log('\nhyperparameters = {')
        last_key = len(self.hyperparameters) - 1
        for k, (key, value) in enumerate(self.hyperparameters.items()):
            if k < last_key:
                log(f"\t'{key}': {value_code(value)},")
            else:
                log(f"\t'{key}': {value_code(value)}")
        log('}\n')

    def __repr__(self, save=True):
        if save:
            repr_list = [f'{key} = {value_code(value)}' for key, value
                         in
                         self.constants.items()]
        else:
            repr_list = [f'{key} = {value}' for key, value in
                         self.constants.items()]
        repr = '\n'.join(repr_list)
        repr = 'Constants: {\n' + repr + '\n}'
        return repr

    def config_modules(self):
        return [name for name in dir(c) if name[0].islower()]

    def copy_config(self, title='config', folder=c.CONFIGS_FOLDER,
                    to_json=False,
                    with_now=True):
        title = PROJECT_NAME + '_' + title
        if with_now:
            title = time_stamp(title)
        if folder == None:
            filename = title
        else:
            filename = os.path.join(folder, title)
        if to_json:
            filename += '.json'
            output = {key: value_code(value, to_json=True)
                      for key, value in self.constants.items()}
            output = json.dumps(output, separators=(',\n', ': '))
            output = '{\n' + output[1: -1] + '\n}'
        else:
            filename += '.py'
            shutil.copy('config.py', filename)

    # TODO .load(self)


def save_log(entry: str):
    global log_lines
    log_lines.insert(-1, entry + '\n')
    with open(LOG_WARNING_FILENAME, 'w') as f:
        f.writelines(log_lines)


def log(entry='', backspaces=0):
    """

    :param entry: str or ready to be turned into str by str built-in
    :param backspaces: int, the number of backspaces to include when printing
    :return:
    """
    print('\b' * backspaces + str(entry))
    save_log(str(entry))


def close_log():
    global log_lines
    writer.close()
    log_lines = [f'The closed log for run {STD_NOW}\n\n'] \
                + log_lines[1: -1] \
                + [f'\nEnd closed log for run {STD_NOW}\n']
    print(f'The closed log for run {STD_NOW}\n\n')
    with open(LOG_FILENAME, 'w') as f:
        f.writelines(log_lines)
    os.remove(LOG_WARNING_FILENAME)


# TODO Consider integrating with Python logging module


# https://dev.to/0xbf/use-dot-syntax-to-access-dictionary-key-python-tips-10ec
class DictDot(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)


"""hp is the same as c.hyperparameters, but with any hyperparameters values 
which aren't lists changed into single element lists
"""
h = DictDot({key: None for key in c.hyperparameters})
hp = c.hyperparameters.copy()
bases = list()
for key, value in c.hyperparameters.items():
    if not isinstance(value, list):
        hp[key] = [value]
    else:
        bases.append(len(value))
keys, values = zip(*hp.items())
n_h = np.prod([len(value) for value in values])
last_key = len(keys) - 1


def set_numpy_rng(index, key):
    try:
        h[key] = np.random.default_rng(h.RANDOM_SEEDS[index])
    except:
        exit(f'Error setting or seeding numpy rng with {index=}')


def set_torch_rng(index, key):
    try:
        # See https://pytorch.org/docs/stable/generated/torch
        # .Generator.html on manual_seed()
        torch_seed_rng = np.random.default_rng(h.RANDOM_SEEDS[index])
        zeros_ones = [0] * 16 + [1] * 16
        torch_seed_rng.shuffle(zeros_ones)  # Note numpy shuffle is inplace
        torch_seed = sum([bit * (2 ** (32 - b))
                          for b, bit in enumerate(zeros_ones)])
        h[key] = torch.Generator(device=c.DEVICE).manual_seed(
            torch_seed)
    except:
        exit(f'Error setting or seeding torch rng with {index=}')


def set_rngs():
    set_numpy_rng(0, 'n_rng')
    set_numpy_rng(1, 'ne_rng')
    set_torch_rng(2, 't_rng')
    set_torch_rng(3, 'te_rng')


def set_and_log_h(keys, bundle, last_key=None, set_h=True):
    last_key = last_key or len(keys) - 1
    log('hyperparameters = {')
    for k, (key, value) in enumerate(zip(keys, bundle)):
        if set_h:
            h[key] = value
        if k < last_key:
            log(f"\t'{key}': {value_code(value)},")
        else:
            log(f"\t'{key}': {value_code(value)}")
    log('}\n')


def time_interval(time_period: float):
    return datetime.timedelta(seconds=round(time_period))


def best_result_format(best_result):
    if isinstance(best_result, float):
        # print('float')
        return f'{best_result:.03f}'
    elif isinstance(best_result, tuple):
        if len(best_result) == 1:
            return f'{best_result[0]:.03f}'
        # print('tuple')
        best_result_string = '('
        for elem in best_result:
            best_result_string += f'{best_result_format(elem)}, '
        return best_result_string[:-2] + ')'
    return f'{best_result}'


def log_intro_hp_run(n_h, hp_run, time_elapsed):
    to_log = f'>>>> {hp_run=} of {n_h}'
    if hp_run > 1:
        now = datetime.datetime.now(tz=tzlocal())
        time_estimate = time_elapsed * n_h / (hp_run - 1)
        end_estimate = now + datetime.timedelta(
            seconds=(time_estimate - time_elapsed))
        to_log += f', time elapsed' \
                  f' {time_interval(time_elapsed)} ' \
                  f'of' \
                  f' estimated {time_interval(time_estimate)}, '
        to_log += '\nimplying ending at '
        if platform.system() == 'Windows':
            to_log += end_estimate.strftime(
                "%H:%M:%S%Z on %A %#d %B %Y")
        else:
            to_log += end_estimate.strftime(
                "%H:%M:%S%Z on %A %-d %B %Y")
    log(to_log)


def log_end_run(n_h, last_key, time_elapsed, results, best_so_far):
    log('\n\n')
    log(f'Time taken over all {n_h} given sets of hyperparameters'
        f'={time_interval(time_elapsed)}, '
        f'averaging {time_interval(time_elapsed / n_h)} per run')
    log('\n\n ---- Table of results ----\n')
    n_figures = len(bases)
    extra_spaces = max(4 - n_figures, 0)
    log(((n_figures + 1 + extra_spaces - 4) * ' ') + 'code  hp_run  '
                                                     'result')
    numbers = ['']
    for base in bases:
        numbers = [number + str(i) for number in numbers for i in
                   range(base)]
    for hp_run, result in enumerate(results, 1):
        log((extra_spaces * ' ') + f' {numbers[hp_run - 1]} '
                                   f'{hp_run:>7} '
                                   f' {best_result_format(result)}')
    log(' ' + ('-' * 26) + '\n')
    best_result_string = best_result_format(best_so_far[0])
    log(f'++++ Best result was {best_result_string} on hp_run'
        f'={best_so_far[1]} with')
    set_and_log_h(best_so_far[2].keys(), best_so_far[2].values(), last_key,
                  set_h=False)


def over_hp(func):
    def wrapper_over_hp(*args, **kwargs):
        # http://stephantul.github.io/python/2019/07/20/product-dict/
        best_so_far_initial = (None, None, None, None)
        best_so_far = best_so_far_initial
        over_hp_start = perf_counter()
        results = list()
        full_results = list()
        for hp_run, bundle in enumerate(product(*values), 1):
            log('\n')
            time_elapsed = perf_counter() - over_hp_start
            log_intro_hp_run(n_h, hp_run, time_elapsed)
            set_and_log_h(keys, bundle, last_key)
            set_rngs()
            h['hp_run'] = hp_run  # Needed for TensorBoard and saved models
            best_of_this_hp_run, idx = func(*args, **kwargs)
            del h['hp_run']
            log(f'\nEnd of hp run {hp_run}.  Result of run:')
            log(best_of_this_hp_run)
            log(result)
            result = best_of_this_hp_run[idx]
            results.append(result)
            full_results.append(best_of_this_hp_run)
            if (best_so_far == best_so_far_initial) or (result < best_so_far[0]):
                best_so_far = (result, hp_run, h.copy(), best_of_this_hp_run)
        time_elapsed = perf_counter() - over_hp_start
        log_end_run(n_h, last_key, time_elapsed, results, best_so_far)
        for key in keys:
            h[key] = None
        return full_results

    return wrapper_over_hp


def test_hp(func):
    if n_h > 1:
        exit('The dictionary config.hyperparameters contains options - need '
             'singleton for test.')

    def wrapper_test_hp(*args, **kwargs):
        best_so_far_initial = (None, None, None, None)
        best_so_far = best_so_far_initial
        test_hp_start = perf_counter()
        for bundle in product(*values):
            log('\n')
            time_elapsed = perf_counter() - test_hp_start
            set_and_log_h(keys, bundle, last_key)
            results = func(*args, **kwargs)
        time_elapsed = perf_counter() - test_hp_start
        log(f'Time taken for test {time_elapsed}')
        log('\n')
        for key in keys:
            h[key] = None
        return results

    return wrapper_test_hp


def div(a, b):
    return a / b if b else np.inf


def savefig(title):
    title += '.pdf'
    fname = time_stamp(folder=c.LOGS_FOLDER, title=title)
    plt.savefig(fname)


def save_model(model, title=None, parameter_name=None, parameter=None):
    model_title = PROJECT_NAME + '_model'
    if title is not None:
        model_title += '_' + str(h.hp_run) + '_' + title
    if parameter is not None:
        model_title += '_'
        if parameter_name is not None:
            model_title += parameter_name
        model_title += str(parameter)
    model_title = time_stamp(model_title)
    torch.save(model, os.path.join(c.MODEL_FOLDER, model_title))
    return model_title


def load_model(model_title):
    return torch.load(os.path.join(c.MODEL_FOLDER, model_title))


def to_device_tensor(x):
    """
    Convert array to device tensor
    :param x: numpy array
    :return:  pytorch c.DEVICE tensor
    """
    return torch.FloatTensor(x).to(c.DEVICE)


def to_array(x):
    """
    Convert device tensor to array
    :param x: pytorch c.DEVICE tensor
    :return: numpy array
    """
    return x.cpu().detach().numpy()


consts = Constants()
consts.log()
consts.copy_config()  # to_json=False
