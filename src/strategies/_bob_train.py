import torch
from src.lib.torch_bin_dec import bin_2_dec
from src.lib.ml_utilities import c, h


class BobTrain:
    def __init__(self, bob):
        self.bob = bob
        self.current_iteration = bob.session.current_iteration

    def __call__(self):
        self.decision_nos = self.bob.session.decision_nos
        self.codes = self.bob.session.codes
        self.rewards = self.bob.session.rewards
        self.selections = self.bob.session.session_spec.spec.circle(
            self.bob.session.selections
        )
        self.current_iteration = self.bob.session.current_iteration


class CircularVocab(BobTrain):
    def __init__(self, bob):
        super().__init__(bob)

    def __call__(self):
        super().__call__()
        decisions = torch.flatten(
            self.selections[torch.arange(self.bob.session.size0),
                            self.decision_nos],
            start_dim=1
        )
        bob_input = torch.cat([decisions, self.codes], dim=1)
        decisions_qs = self.bob.net(bob_input).reshape((
            self.bob.session.size0,))
        return self.bob.loss_function(decisions_qs, self.rewards)


class QPerNumber(BobTrain):
    def __init__(self, bob):
        super().__init__(bob)

    def __call__(self):
        super().__call__()
        selections = torch.flatten(self.selections, start_dim=1)
        q_estimates = self.bob.net(
            torch.cat([selections, self.codes], 1)
        )
        decision_q_estimates = q_estimates[
            torch.arange(self.bob.session.size0),
            self.decision_nos
        ]
        return self.bob.loss_function(decision_q_estimates, self.rewards)
