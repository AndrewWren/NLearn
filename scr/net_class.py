#!/usr/bin/env python3
import itertools
import math
import random
import warnings
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import config as c
import scripts.ml_utilities as mlu
from scripts.ml_utilities import c, h

warnings.filterwarnings("ignore", category=UserWarning)
margin = torch.ones(1).to(c.DEVICE)

def distance(x_i, x_j):
    x_i = x_i.cpu().detach().numpy().flatten()
    x_j = x_j.cpu().detach().numpy().flatten()
    return np.linalg.norm(x_i - x_j)


def distance_t(x_i, x_j):
    return torch.sqrt(torch.sum((x_i - x_j) * (x_i - x_j)))


def triplet_fn_norouzi(x, x_i, x_j, y, y_i, y_j, alpha=1, full_test=False):
    dist_i = distance(x, x_i)
    dist_j = distance(x, x_j)
    # print(f'{dist_i=}, {dist_j=}')
    if dist_i == dist_j:
        return torch.zeros(1, device=c.DEVICE)
    elif dist_i < dist_j:
        y_near = y_i
        y_far = y_j
    else:
        y_near = y_j
        y_far = y_i
    # print(f'{y_i[:3]=}, {y_j[:3]=}')
    dist_y_near = 0.5 * torch.dot(y, y_near)
    dist_y_far = 0.5 * torch.dot(y, y_far)
    # mlu.log(f'{dist_y_near=}, {dist_y_far=}')
    res = dist_y_far - dist_y_near + 1 * torch.ones(1,
            device=c.DEVICE)
    if full_test:
        return res
    else:
        return nn.ReLU()(res)


def harden(y):
    return torch.sgn(y)


def triplet_fn_wang(x, x_i, x_j, y, y_i, y_j, alpha=1):
    dist_i = distance(x, x_i)
    dist_j = distance(x, x_j)
    # print(f'{dist_i=}, {dist_j=}')
    if dist_i == dist_j:
        return torch.zeros(1, device=c.DEVICE)
    elif dist_i < dist_j:
        y_near = y_i
        y_far = y_j
    else:
        y_near = y_j
        y_far = y_i
    # print(f'{y_i[:3]=}, {y_j[:3]=}')
    dist_y_near = 0.5 * torch.dot(y, y_near)
    dist_y_far = 0.5 * torch.dot(y, y_far)
    # print(f'{dist_y_near=}, {dist_y_far=}')
    argument = dist_y_near - dist_y_far - alpha * margin
    # mlu.log(f'{dist_y_near.item()=}, {dist_y_far.item()=}, {argument.item(
    # )=}')
    """if (a:=argument.item()) < -5.:
        res = - argument
    elif a > 5.:#
        res = torch.zeros(1).to(c.DEVICE)
    else:
        res = - F.logsigmoid(argument)
    mlu.log(f'{dist_y_near.item()=}, {dist_y_far.item()=}, '
           f'{argument.item()=}, {res.item()=}')
    return res
"""
    return - argument + F.softplus(argument)


class Net(nn.Module):
    """The Net class to use in defining a model that can be run as in the
    code below.
    This class cannot be used directly, instead an individual type of net
    should be built as a sub-class of Net, and this sub-class must define
    the following methods:
        .__init__ in order to initalise with e.g. the hyperparameters for
        the shape of the net

        ._build which is run in the Net.data method when it is called.  This
        defines the layers of the Net instance,
        in particular shaping the first layer's input to match the input
        shape appropriately.  The initial underscore in ._build() indicates
        it's better not to run this publicly: instead create a new Net
        instance to re-initialise weights.  This is clearer code.

        .forward which actually runs the Net instance.
    """

    def __init__(self):
        super().__init__()
        self.data_train = None
        # self.y_train = None
        self.data_val = None
        self.input_shape = None
        self.y_val = None
        self.objective_fn = None
        self.triplet_fn = None
        self.n_epochs = None
        self.batch_size = 32
        self.optimizing_fn = None

    def data(self, data_train, data_val=None):
        self.data_train = data_train
        self.data_val = data_val
        self.input_shape = None # i.e. don't do anything with this
        self._build()

    def create_triplet_fn(self, triplet_fn):
        if triplet_fn == 'Norouzi+ 2012':
            self.triplet_fn = triplet_fn_norouzi
        elif triplet_fn == 'Wang+ 2016':
            self.triplet_fn = triplet_fn_wang

    def objective(self, objective_cls, *objective_args, **objective_kwargs):
        self.objective_fn = objective_cls(*objective_args, **objective_kwargs)

    def optimizer(self, optimizing_fn, *optimizing_args, **optimizing_kwargs):
        # TODO haven't tested yet with args
        self.optimizing_fn = optimizing_fn(self.parameters(), *optimizing_args,
                                           **optimizing_kwargs)

    """def loss(self, x, y, item=True):
        net = self.to(c.DEVICE)
        x_t = torch.FloatTensor(x).to(c.DEVICE)

        result = self.objective_fn(net(x_t))
        if item:
            return result.item()
        else:
            return result
    """

    def fit_triplets(self, print_rate=1,
                     callbacks=list()):
        self.n_epochs = h.N_EPOCHS
        self.batch_size = h.BATCH_SIZE
        optimizer = self.optimizing_fn
        torch.manual_seed(h.RANDOM_SEED)
        np.random.seed(h.RANDOM_SEED)

        # n_samples = self.data_train.shape[0]
        net = self.to(c.DEVICE)

        x_train = self.data_train
        x_val = self.data_val

        triplet_fn = self.triplet_fn

        train_batches_per_epoch = math.ceil(len(x_train) / batch_size)
        train_full_batches_per_epoch = len(x_train) // batch_size
        train_remainder = len(x_train) % batch_size
        train_triplets_per_epoch = train_full_batches_per_epoch * (
                batch_size // 3 ) + train_remainder // 3
        if x_val is not None:
            val_batches_per_epoch = int(math.ceil(len(x_val) / 32))
            val_remainder = len(x_val) % 32
            val_triplets_per_epoch = val_batches_per_epoch * (32 // 3) \
                                       + val_remainder // 3
            val_pairs_per_epoch = (val_batches_per_epoch *
                32 * (32 - 1) + val_remainder * (val_remainder - 1))/2
        best_so_far_initial = (None, (None, None, None, None, None), None) #
        # Always need a
        best_so_far = best_so_far_initial
        # value as
        # returned
        for epoch in range(1, self.n_epochs + 1):
            epoch_outputs = torch.empty(0, device=c.DEVICE)
            train_loss = 0.
            batches_train = torch.utils.data.DataLoader(x_train,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True)
            sample_size_total = 0
            self.training = True
            for b, batch_x in enumerate(batches_train, start=1): # as the
                # dataset
                if self.train_tile_net != None:
                    batch_x = batch_x[0]
                """print('\b' * 29, f"{f'Train batch {b}':>18} of "
                                 f"{train_batches_per_epoch:<6}", end='')
                """
                batch_x = batch_x.to(c.DEVICE)
                print('\b' * 29, f"Train batch {b:>5} of "
                                 f"{train_batches_per_epoch:<6}", end='')
            # has "y"
                # all
                # set the same
                optimizer.zero_grad()
                # print(f'batch = {batch_start}, batch_x.shape {batch_x.shape}')
                output = net(batch_x)
                epoch_outputs = torch.cat((epoch_outputs, output))
                # if epoch == 1:
                #     print(f'Lengths: {batch_x.size()}, {output.size()},'
                #           f' {batch_y.size()}')
                loss = torch.zeros(1).to(c.DEVICE)
                batch_length = len(batch_x)
                if batch_length < 3:
                    continue
                sample_size = int(round((batch_length - 1)
                                        * (batch_length - 2)
                                        * self.sample_proportion / 2))
                if sample_size == 0:
                    sample_size = 1
                sample_size_total += (sample_size * batch_length)
                for idx, x in enumerate(batch_x):
                    others = set(range(batch_length)) - set([idx])
                    ijs = random.sample(list(itertools.combinations(others,
                                                                    2)),
                                        sample_size)
                    for i, j in ijs:
                        x_i = batch_x[i]
                        x_j = batch_x[j]
                        y = output[idx]
                        y_i = output[i]
                        y_j = output[j]
                        loss += triplet_fn(x, x_i, x_j, y, y_i, y_j,
                                           alpha=self.alpha)
                        if torch.any(torch.isnan(loss) or (loss == np.inf)):
                            mlu.log(f'train error at batch {b}: {x=}, '
                                    f'{x_i=}, {x_j=}, {y=}, {y_i=}, {y_j=}')
                            return (None, (np.inf, epoch),
                                                   np.inf)
                # print(f'{loss.item()=}')
                bin_loss = self.beta * torch.sum((output - harden(
                    output)) ** 2) * sample_size / batch_length
                # mlu.log(f'{loss.item()=}, {bin_loss.item()=}')
                """if b == int(math.ceil(len(data_train) / batch_size)) - 1:
                    print(output)
                    print(bin_loss)
                """
                loss += bin_loss
                """hy = [harden(y) for y in output]
                # print(h)
                dots = ([torch.dot(hi, hy[j]) / self.n_code
                                  for i, hi in enumerate(hy) if i > 0
                                  for j in range(i)])
                """# if self.mu:
                #    proximity_loss = 2 * torch.sum(\
                #        torch.FloatTensor(
                #            dots)) / (len(batch_x) * (len(batch_x) - 1))
                # mlu.log(f'{proximity_loss.item()=}')
                #    loss += self.mu * proximity_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            # print(f'{bin_loss:.03f}')
            # print('\n')
            train_loss /= sample_size_total
            if x_val is not None:
                with torch.no_grad():
                    batches_val = torch.utils.data.DataLoader(
                        x_val, batch_size=32, shuffle=True,
                        pin_memory=True)
                    val_loss = 0.
                    val_hard_loss = 0.
                    val_hard_loss_n = 0.
                    proximity_loss = 0.
                    wrong_way_count = 0
                    wrong_sum = 0
                    right_sum = 0
                    sample_size_total = 0
                    # print(f'{loss.item()=}')
                    # print('Init', val_loss, end=' ')
                    self.training = False
                    for b, batch_x in enumerate(batches_val):  # as the
                    # dataset has
                        # "y" all
                        # set the same
                        if self.train_tile_net != None:
                            batch_x = batch_x[0]
                        batch_x = batch_x.to(c.DEVICE)
                        print('\b' * 29, f"Val batch {b:>5} of "
                                  f"{val_batches_per_epoch:<6}", end='')
                        # print(f'batch = {batch_start}, batch_x.shape {batch_x.shape}')
                        output = net(batch_x)
                        # if epoch == 1:
                        #     print(f'Lengths: {batch_x.size()}, {output.size()},'
                        #           f' {batch_y.size()}')
                        # TODO Consider what random generator to add to randperm
                        loss = torch.zeros(1).to(c.DEVICE)
                        hard_loss = torch.zeros(1).to(c.DEVICE)
                        hard_loss_n = torch.zeros(1).to(c.DEVICE)
                        batch_length = len(batch_x)
                        if batch_length < 3:
                            continue
                        sample_size = int(round((batch_length - 1)
                                                * (batch_length - 2)
                                                * self.sample_proportion / 2))
                        if sample_size == 0:
                            sample_size = 1
                        sample_size_total += (sample_size * batch_length)
                        for idx, x in enumerate(batch_x):
                            others = set(range(batch_length)) - set([idx])
                            ijs = random.sample(
                                list(itertools.combinations(others,
                                                            2)),
                                sample_size)
                            for i, j in ijs:
                                x_i = batch_x[i]
                                x_j = batch_x[j]
                                y = output[idx]
                                y_i = output[i]
                                y_j = output[j]
                                loss += triplet_fn(x, x_i, x_j, y, y_i, y_j,
                                                   alpha=self.alpha)
                                if torch.any(torch.isnan(loss) or (loss ==
                                                                    np.inf)):
                                    mlu.log(f'val error at batch {b}')
                                    return (None, (np.inf, epoch),
                                                           np.inf)
                                y = harden(y)
                                y_i = harden(y_i)
                                y_j = harden(y_j)
                                hard_loss += triplet_fn(x, x_i,
                                                        x_j, y,
                                                        y_i, y_j,
                                                        alpha=self.alpha)
                                loss_n = triplet_fn_norouzi(x, x_i,
                                                                 x_j, y,
                                                                y_i, y_j,
                                                            full_test=True)
                                if loss_n.item() > self.n_code + 1: # the +1
                                    # becuase of the margin of 1
                                    mlu.log(f'{b=}, {idx=}, {i=}, {j=}, '
                                            f'{loss_n.item()=}')
                                hard_loss_n += loss_n
                                loss_n = loss_n.item()
                                if loss_n >= 0:
                                    wrong_way_count += 1
                                    wrong_sum += loss_n
                                else:
                                    right_sum += loss_n
                                """if epoch == self.n_epochs:
                                    mlu.log(f'{torch.all(y_i == y).item()}, '
                                           f'{torch.all(y_j == y).item()}')
                                """
                                # mlu.log(f'{hard_loss_n.item()=}')
                        bin_loss = self.beta * torch.sum((output - harden(
                            output)) ** 2) * sample_size / batch_length
                        loss += bin_loss
                        """print(f'{b=}, {batch_length=}, {sample_size=},'
                              f' {loss=}')
                        """
                        val_loss += (loss + bin_loss).item()
                        val_hard_loss += hard_loss.item()
                        """if self.mu:
                            proximity_loss += torch.sum(
                                torch.FloatTensor(dots)).item()
                        """
                        # mlu.log(f'{proximity_loss=}')
                        val_hard_loss_n += hard_loss_n.item()

                        # print(f'{bin_loss=}')
                    val_loss /= sample_size_total
                    val_hard_loss /= sample_size_total
                    # print(f'{val_hard_loss_n=}, {sample_size_total=}')
                    val_hard_loss_n /= sample_size_total
                    wrong_way_mean = wrong_way_count / sample_size_total
                    wrong_mean = mlu.div(wrong_sum, wrong_way_count *
                                                   net.n_code)
                    right_mean = mlu.div(right_sum ,
                                (sample_size_total - wrong_way_count) *
                                net.n_code)
                    hard_loss_n = mlu.div(wrong_sum, sample_size_total *
                                                 net.n_code)
                    result = (wrong_way_mean, wrong_mean, right_mean,
                              hard_loss_n, epoch)
                    if (best_so_far == best_so_far_initial) or (
                            result < best_so_far[1]):
                        best_so_far = (self, result, proximity_loss)
                    self.training = True
            if epoch % print_rate == 0:
                if (x_val is not None) and self.mu:
                    mlu.log(
                        f'Epoch {epoch:>5}\t \t'
                        f'{train_loss=:.03f}\t \t '
                        f'val result={mlu.best_result_format(result[:-1])}',
                        backspaces=29)
                elif x_val is not None:
                    mlu.log(
                        f'Epoch {epoch:>5}\t \t'
                        f'{train_loss=:.03f}\t \t '
                        f'val result={mlu.best_result_format(result[:-1])}',
                        backspaces=29)
                else:
                        mlu.log(
                            f'Epoch {epoch:>5}\t \t'
                            f'{train_loss=:7.3f}',
                        backspaces=29)
            callback_results = list()
            for callback in callbacks:
                try:
                    callback_results.append(callback(self))
                except:
                    mlu.log(f'Callback function {callback} failed')
            if 'Stop' in callback_results:
                callback_stops = [callback for callback, callback_result
                                  in zip(callbacks, callback_results)
                                  if callback_result == 'Stop']
                mlu.log(f'Stopped by callback(s) {callback_stops} at epoch'
                      f' {epoch}')
                return best_so_far
        return best_so_far

    def fit(self, print_rate=1,
            callbacks=list()):
        self.n_epochs = h.N_EPOCHS
        self.batch_size = h.BATCH_SIZE
        batch_size = self.batch_size
        optimizer = self.optimizing_fn
        torch.manual_seed(h.RANDOM_SEED)
        np.random.seed(h.RANDOM_SEED)

        # n_samples = self.data_train.shape[0]
        net = self.to(c.DEVICE)

        data_train = self.data_train
        data_val = self.data_val

        objective_fn = self.objective_fn

        best_so_far_initial = (None, None) #
        # Always need a
        best_so_far = best_so_far_initial
        train_data_length = len(data_train)
        train_batches_per_epoch = math.ceil(train_data_length / batch_size)
        if data_val is not None:
            val_data_length = len(data_val)
        # value as
        # returned
        for epoch in range(1, self.n_epochs + 1):
            train_loss = 0.
            batches_train = torch.utils.data.DataLoader(data_train,
                                                        batch_size=batch_size,
                                                        shuffle=True)
            self.training = True
            for b, (batch_x, batch_y) in enumerate(batches_train, start=1):
                # as the
                # dataset
                if self.train_tile_net != None:
                    batch_x = batch_x[0]
                """print('\b' * 29, f"{f'Train batch {b}':>18} of "
                                 f"{train_batches_per_epoch:<6}", end='')
                """
                batch_x = batch_x.to(c.DEVICE)
                batch_y = batch_y.to(c.DEVICE)
                print('\b' * 29, f"Train batch {b:>5} of "
                                 f"{train_batches_per_epoch:<6}", end='')
            # has "y"
                # all
                # set the same
                optimizer.zero_grad()
                # print(f'batch = {batch_start}, batch_x.shape {batch_x.shape}')
                output = net(batch_x)
                # print(f'{len(batch_x)=}, {len(batch_y)=}, {len(output)=}')
                # if epoch == 1:
                #     print(f'Lengths: {batch_x.size()}, {output.size()},'
                #           f' {batch_y.size()}')
                # print(self.objective_fn)
                loss = self.objective_fn(output, batch_y) / 4
                bin_loss = self.beta * torch.sum((output - harden(
                    output)) ** 2) / 4
                loss += bin_loss
                if torch.any(torch.isnan(loss) or (loss == np.inf)):
                    mlu.log(f'train error at batch {b}')
                    return (None, (np.inf, epoch))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            # print(f'{bin_loss:.03f}')
            # print('\n')
            train_loss /= train_data_length
            if data_val is not None:
                with torch.no_grad():
                    batches_val = torch.utils.data.DataLoader(
                        data_val, batch_size=32, shuffle=True)
                    # print(f'{loss.item()=}')
                    # print('Init', val_loss, end=' ')
                    val_batches_per_epoch = int(math.ceil(len(data_val) / 32))
                    val_loss = 0.
                    self.training = False
                    for b, (batch_x, batch_y) in enumerate(batches_val):  #
                        # as the
                    # dataset has
                        # "y" all
                        # set the same
                        if self.train_tile_net != None:
                            batch_x = batch_x[0]
                        batch_x = batch_x.to(c.DEVICE)
                        print('\b' * 29, f"Val batch {b:>5} of "
                                  f"{val_batches_per_epoch:<6}", end='')
                        # print(f'batch = {batch_start}, batch_x.shape {batch_x.shape}')
                        output = harden(net(batch_x))
                        # print(output)
                        # if epoch == 1:
                        #     print(f'Lengths: {batch_x.size()}, {output.size()},'
                        #           f' {batch_y.size()}')
                        # TODO Consider what random generator to add to randperm
                        loss = self.objective_fn(output, harden(batch_y))
                        if torch.any(torch.isnan(loss) or (loss == np.inf)):
                            mlu.log(f'val error at batch {b}')
                            return (None, (np.inf, epoch))
                        val_loss += loss.item() / 4
                    val_loss /= val_data_length
                    result = (val_loss, epoch)
                    if (best_so_far == best_so_far_initial) or (
                            result < best_so_far[1]):
                        best_so_far = (self, result)
                    self.training = True
            if epoch % print_rate == 0:
                if (data_val is not None) and self.mu:
                    mlu.log(
                        f'Epoch {epoch:>5}\t \t'
                        f'{train_loss=:.3f}\t \t '
                        f'val result={mlu.best_result_format(result[:-1])}',
                        backspaces=29)
                elif data_val is not None:
                    mlu.log(
                        f'Epoch {epoch:>5}\t \t'
                        f'{train_loss=:.3f}\t \t '
                        f'val result={mlu.best_result_format(result[:-1])}',
                        backspaces=29)
                else:
                        mlu.log(
                            f'Epoch {epoch:>5}\t \t'
                            f'{train_loss=:7.3f}',
                        backspaces=29)
            callback_results = list()
            for callback in callbacks:
                try:
                    callback_results.append(callback(self))
                except:
                    mlu.log(f'Callback function {callback} failed')
            if 'Stop' in callback_results:
                callback_stops = [callback for callback, callback_result
                                  in zip(callbacks, callback_results)
                                  if callback_result == 'Stop']
                mlu.log(f'Stopped by callback(s) {callback_stops} at epoch'
                      f' {epoch}')
                return best_so_far
        return best_so_far

    def predict(self, x, pre_function=None):
        """

           :param x: numpy array of floats or ?? list of floats
           :type pre_function: python function.  Cannot be torch.nn.functional
        """
        with torch.no_grad():
            x_t = torch.FloatTensor(x).to(c.DEVICE)
            y_pred_t = self(x_t)
            y_pred = y_pred_t.cpu().numpy()
            if pre_function is not None:
                pre_function_v = np.vectorize(pre_function)
                y_pred = pre_function_v(y_pred)
            return y_pred

    def assess(self, x, metric, pre_function=None):
        """

        :type pre_function: python function.  Cannot be torch.nn.functional
        :type metric: python function or torch.nn.functional
        """
        y_pred = self.predict(x, pre_function=pre_function)
        try: # assuming the metric is for numpy arrays
            return metric(y)
        except: # when the metric is for torch tensors
            with torch.no_grad():
                y_pred_t = torch.FloatTensor(y_pred).to(c.DEVICE)
                return metric(y_pred_t).cpu().numpy()

    # TODO Create a wrapper for nn.Sequential along the lines of
    #  neural_net.train_two_part_nn._build
