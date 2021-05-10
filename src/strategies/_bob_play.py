import torch
from src.lib.torch_bin_dec import dec_2_bin, initialise_bin_dec
from src.lib.ml_utilities import c, h


class BobPlay:
    def __init__(self, bob):
        self.bob = bob

    def __call__(self):
        self.codes = self.bob.session.codes
        self.selections = self.bob.session.selections
        # self.targets = torch.flatten(self.bob.session.targets_t, start_dim=1)


class Circular(BobPlay):
    def __init__(self, bob):
        super().__init__(bob)
        self.input_width = (h.N_SELECT *
                                self.bob.session.tuple_specs.n_elements
                                * 2 + h.N_CODE)
        self.output_width = h.N_SELECT

    def __call__(self):
        super().__call__()
        torch.flatten(self.selections, start_dim=1)
        bob_input = torch.cat([self.selections, self.codes], 1)
        bob_q_estimates = self.bob(bob_input)
        return torch.argmax(bob_q_estimates, dim=1).long()


class CircularVocab(BobPlay):
    def __init__(self, bob):
        super().__init__(bob)
        self.input_width = self.bob.session.tuple_specs.n_elements * 2 \
                               + h.N_CODE
        self.output_width = 1

    def __call__(self):
        super().__call__()
        self.selections = torch.transpose(self.selections, 0, 1)
        bob_q_estimates = list()
        for selection in self.selections:  # TODO all in a single net call?
            bob_input = torch.cat(
                [torch.flatten(selection, start_dim=1), self.codes], 1
            )
            bob_q_estimates.append(self.bob.net(bob_input))
        return torch.argmax(
            torch.reshape(torch.stack(bob_q_estimates, dim=1),
                              (self.bob.session.size0, h.N_SELECT)),
            dim=1
        )
