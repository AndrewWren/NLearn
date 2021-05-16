import torch
from src.lib.torch_bin_dec import bin_2_dec
from src.lib.ml_utilities import c, h


class AliceTrain:
    def __init__(self, alice):
        self.alice = alice
        self.current_iteration = alice.session.current_iteration

    def __call__(self):
        self.targets = self.alice.session.session_spec.spec.circle(
            self.alice.session.targets_t
        )
        self.decisions = self.alice.session.decisions
        self.codes = self.alice.session.codes
        self.rewards = self.alice.session.rewards
        self.current_iteration = self.alice.session.current_iteration


class QPerCode(AliceTrain):
    def __init__(self, alice):
        super().__init__(alice)

    def __call__(self):
        super().__call__()
        alice_outputs = self.alice.training_net(self.targets)
        code_decs =  bin_2_dec(self.codes)
        alice_qs = alice_outputs[torch.arange(h.BATCHSIZE), code_decs]
        return self.alice.loss_function(alice_qs, self.rewards)
