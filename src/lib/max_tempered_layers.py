from time import perf_counter
import numpy as np
import torch
from src.lib.ml_utilities import c


class MaxTemperedOutFocused(torch.nn.Linear):
    def __init__(self, in_features, out_features, beta=0.2, bias=True,
                 bias_included=None, relu=False, cos=False,
                 dropout=0.):
        """

        :param in_features:
        :param out_features:
        :param beta:
        :param bias:
        :param bias_included: a dummy for compatibility with
        MaxTemperedInFocused
        """
        super().__init__(in_features, out_features, bias)
        self.beta = beta
        self.relu_layer = torch.nn.ReLU()
        self.relu_flag = relu
        self.dropout_layer = torch.nn.Dropout(p=dropout)
        self.dropout_flag = bool(dropout)
        self.id_in = torch.diag_embed(torch.ones(in_features)).to(c.DEVICE)

    def forward(self, input):
        weighted_inputs = torch.einsum('oj, ji, bi -> boi', self.weight,
                                       self.id_in, input)
        batchsize = input.size()[0]
        lineared_inputs = torch.einsum('oi, bi -> bo', self.weight, input)
        x = (1 - self.beta) * lineared_inputs \
               + self.beta * torch.max(weighted_inputs, dim=-1).values \
               + self.bias.repeat(batchsize, 1)
        if self.relu_flag:
            x = self.relu_layer(x)
        if self.dropout_layer:
            x = self.dropout_layer(x)
        return x


class MaxTemperedInFocused(torch.nn.Linear):
    def __init__(self, in_features, out_features, beta=0.2,
                 bias_included=False, bias=True, relu=False, cos=False,
                 dropout=0.):
        super().__init__(in_features, out_features, bias)
        self.beta = beta
        self.relu_layer = torch.nn.ReLU()
        self.relu_flag = relu
        self.dropout_layer = torch.nn.Dropout(p=dropout)
        self.dropout_flag = bool(dropout)
        self.id_in = torch.diag_embed(torch.ones(in_features)).to(c.DEVICE)
        self.bias_included = bias_included
        self.cos_flag = cos  # This should work, but surely isn't that
        # meaningful without a complex set up?

    def forward(self, input):
        batchsize = input.size()[0]
        if self.cos_flag:
            weight = torch.cos(self.weight)
        else:
            weight = self.weight
        return self.max_in(weight, batchsize, input)

    def max_in(self, weight, batchsize, input):
        summands = torch.max(
            torch.einsum(
                'oj, ji, bi -> bio',
                weight,
                self.id_in,
                input
            ) + self.bias_included * self.bias.repeat(batchsize,
                                                      self.in_features, 1),
            dim=-1
        )
        max_matrix = torch.zeros(
            batchsize,
            self.out_features,
            self.in_features
        ).to(c.DEVICE)
        b_index = torch.arange(batchsize).repeat_interleave(
            self.in_features).to(c.DEVICE)
        i_index = torch.arange(self.in_features).repeat(batchsize).to(
            c.DEVICE)
        o_index = summands.indices.flatten().to(c.DEVICE)
        max_matrix = max_matrix.index_put(
            (b_index, o_index, i_index),
            summands.values.flatten()
        )
        bias_repeated = self.bias.repeat(batchsize, 1)
        lineared_inputs = torch.einsum('oi, bi -> bo', weight, input) \
                          + bias_repeated
        x = (1 - self.beta) * lineared_inputs \
            + self.beta * (
                    torch.sum(max_matrix, dim=-1) \
                    + (not self.bias_included) * bias_repeated
            )
        if self.relu_flag:
            x = self.relu_layer(x)
        if self.dropout_layer:
            x = self.dropout_layer(x)
        return x


class MaxNet(torch.nn.Module):
    def __init__(self, input_width, output_width, focus, layers, width,
                 beta=0.2, bias_included=False, relu=False, cos=False,
                 dropout=0.):
        super().__init__()
        if focus == 'Out':
            MaxLayer = MaxTemperedOutFocused
        elif focus == 'In':
            MaxLayer = MaxTemperedInFocused
        self.beta = beta
        self.relu_flag = relu
        self.layer_first = MaxLayer(input_width, width, beta=beta,
                                    bias_included=bias_included, relu=relu,
                                    cos=cos, dropout=dropout)
        self.hidden_layers = torch.nn.ModuleList(
            [MaxLayer(width, width, beta=beta,
                      bias_included=bias_included, relu=relu, cos=cos,
                      dropout=dropout)
             for _ in range(layers - 2)]
        )
        self.layer_last = MaxLayer(width, output_width, beta=beta,
                                   bias_included=bias_included, cos=cos)

    def forward(self, x):
        x = self.layer_first(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.layer_last(x).squeeze(dim=-1)


class FFs(torch.nn.Module):

    def __init__(self, input_width=3, output_width=1, layers=10, width=50):
        super().__init__()
        self.layers = layers
        self.width = width
        self.input_width = input_width
        self.output_width = output_width
        self.relu = torch.nn.ReLU(inplace=True)
        self._build()

    def _build(self):
        ffs = [torch.nn.Linear(self.input_width, self.width)]
        ffs += [torch.nn.Linear(self.width, self.width) for _ in
                range(self.layers
                      - 2)]
        ffs += [torch.nn.Linear(self.width, self.output_width)]
        self.ffs = torch.nn.ModuleList(ffs)

    def forward(self, x):
        for ff in self.ffs[: -1]:
            x = ff(x)
            self.relu(x)
        return self.ffs[-1](x).squeeze(dim=-1)


def f(tensor):
    tensor = tensor.transpose(0, 1)
    return tensor[0] ** 3 - 2 * torch.cos(tensor[0] * tensor[1] - tensor[2]) \
           + 3 * tensor[2]


def train(net, input):
    start = perf_counter()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_function = torch.nn.MSELoss()
    for iteration, batch in enumerate(input):
        target = f(batch)
        prediction = net(batch)
        optimizer.zero_grad()

        loss = loss_function(prediction, target)
        loss.backward()
        optimizer.step()
        # print(iteration, '\t', loss.item())
    valid = torch.empty(32, 3)
    valid.uniform_(-1, to=1)
    target = f(valid)
    prediction = net(valid)
    loss = loss_function(prediction, target)
    end = perf_counter()
    time_taken = end - start
    # print(f'Validation \t{loss} \t time {time_taken}')
    return loss.item(), time_taken


def print_stats(x, name):
    print(f'{name}:\t{np.mean(x)}\t{np.std(x)}')


if __name__ == '__main__':
    TRIES = 10000
    input = torch.empty(30, 32, 3)
    input.uniform_(-1, to=1)

    max_losses = list()
    max_times = list()
    for t in range(TRIES):
        net = MaxNet('In', 2, 10, beta=0.2)
        loss, time_taken = train(net, input)
        max_losses.append(loss)
        max_times.append(time_taken)
        if t % 100 == 0:
            print(f'{t=}')
    print('\n')

    lin_losses = list()
    lin_times = list()
    for _ in range(TRIES):
        net = FFs(layers=2, width=10)
        loss, time_taken = train(net, input)
        lin_losses.append(loss)
        lin_times.append(time_taken)
    print('\n')

    print('\n')
    print_stats(max_losses, 'Max losses')
    print_stats(max_times, 'Max times')
    print_stats(lin_losses, 'Lin losses')
    print_stats(lin_times, 'Lin times')
