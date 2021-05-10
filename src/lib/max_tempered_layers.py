import numpy as np
import torch


class MaxTemperedOutFocused(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.id_in = torch.diag_embed(torch.ones(in_features))

    def forward(self, input):
        # print(f'{self.weight.size()=}')
        # print(f'{self.id_in.size()=}')
        # print(f'{input.size()=}')
        weighted_inputs = torch.einsum('oj, ji, bi -> boi', self.weight,
                                    self.id_in, input)
        # print(f'{weighted_inputs=}')
        # print(f'{self.bias=}')
        return torch.max(weighted_inputs, dim=-1).values + self.bias.repeat(
            input.size()[0] , 1)


class MaxTemperedInFocused(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.id_in = torch.diag_embed(torch.ones(in_features))

    def forward(self, input):
        summands = torch.max(
            torch.einsum('oj, ji, bi -> bio', self.weight, self.id_in,
                              input),
            dim=-1
        )
        batchsize = input.size()[0]
        max_matrix = torch.zeros(
            batchsize,
            self.out_features,
            self.in_features
        )
        b_index = torch.arange(batchsize).repeat_interleave(self.in_features)
        i_index = torch.arange(self.in_features).repeat(batchsize)
        o_index = summands.indices.flatten()
        max_matrix = max_matrix.index_put(
            (b_index, o_index, i_index),
            summands.values.flatten()
        )
        return torch.sum(max_matrix, dim=-1) + self.bias.repeat(batchsize, 1)


class Net(torch.nn.Module):
    def __init__(self, focus, layers, width):
        super().__init__()
        if focus == 'Out':
            MaxLayer = MaxTemperedOutFocused
        elif focus == 'In':
            MaxLayer = MaxTemperedInFocused
        self.relu = torch.nn.ReLU()
        self.layer_first = MaxLayer(3, width)
        self.hidden_layers = torch.nn.ModuleList(
            [MaxLayer(width, width) for _ in range(layers - 2)]
        )
        self.layer_last = MaxLayer(width, 1)

    def forward(self, x):
        x = self.layer_first(x)
        #x = self.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            #x = self.relu(x)
        return self.layer_last(x).squeeze(dim=-1)


def f(tensor):
    tensor = tensor.transpose(0, 1)
    return tensor[0] ** 3 - 2 * torch.cos(tensor[0] * tensor[1] - tensor[2])\
           + 3 * tensor[2]


# def training():


if __name__ == '__main__':
    net = Net('In', 2, 10)
    input = torch.empty(30, 32, 3)
    input.uniform_(-1, to=1)

    # print(f'{target=}')
    # exit()
    # output = net(input)
    # print(f'{output=}')
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_function = torch.nn.MSELoss()
    for iteration, batch in enumerate(input):
        target = f(batch)
        prediction = net(batch)
        optimizer.zero_grad()

        loss = loss_function(prediction, target)
        loss.backward()
        optimizer.step()
        print(iteration, '\t', loss.item())
    valid = torch.empty(32, 3)
    valid.uniform_(-1, to=1)
    target = f(valid)
    prediction = net(valid)
    loss = loss_function(prediction, target)
    print('Validation \t', loss)
