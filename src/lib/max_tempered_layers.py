import math
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


class ComplexLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
        torch.nn.init.uniform_(self.weight, a= - np.pi, b=np.pi)

    def forward(self, x):
        """Note because sums over inputs doesn't preserve modulus
        """
        x_r, x_i = x.permute(2, 0, 1)
        c_weight = torch.cos(self.weight)
        print(f'{c_weight=}')
        s_weight = torch.sin(self.weight)
        print(f'{s_weight=}')
        y_r = torch.nn.functional.linear(x_r, c_weight) \
              - torch.nn.functional.linear(x_i, s_weight)
        y_i = torch.nn.functional.linear(x_r, s_weight) \
              + torch.nn.functional.linear(x_i, c_weight)
        return torch.stack((y_r, y_i), dim=-1)





class ComplexReLU(torch.nn.Module):
    """
    See arXiv:1802.08026, eq. 16
    """
    def forward(self, x):
        x_r, x_i = x.permute(2, 0, 1)
        y_r = torch.heaviside(x_i, torch.FloatTensor([1])) \
              * torch.nn.functional.relu(x_r)
        y_i = torch.heaviside(x_r, torch.FloatTensor([0])) \
              * torch.nn.functional.relu(x_i)
        return torch.stack((y_r, y_i), dim=-1)


class MaxNet(torch.nn.Module):
    def __init__(self, input_width, output_width, focus, layers, width,
                 beta=0.2, bias_included=False, relu=False, cos=False,
                 dropout=0.):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.layers = layers
        self.width = width
        if focus == 'Out':
            self.MaxLayer = MaxTemperedOutFocused
        elif focus == 'In':
            self.MaxLayer = MaxTemperedInFocused
        self.beta = beta
        self.relu_flag = relu
        self.layer_first = self.MaxLayer(input_width, width, beta=beta,
                                    bias_included=bias_included, relu=relu,
                                    cos=cos, dropout=dropout)
        self.hidden_layers = torch.nn.ModuleList(
            [self.MaxLayer(width, width, beta=beta,
                      bias_included=bias_included, relu=relu, cos=cos,
                      dropout=dropout)
             for _ in range(layers - 2)]
        )
        self.layer_last = self.MaxLayer(width, output_width, beta=beta,
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
    unitary_max = UnitaryMax(4, 5)
    a = torch.randn(7, 4)
    circular_map = lambda x: torch.stack((torch.cos(x), torch.sin(x)), dim=-1)
    a = circular_map(a)
    print(abs_sq(a))
    print(a.size())
    b = unitary_max(a)
    print(b)
    print(b.size())
