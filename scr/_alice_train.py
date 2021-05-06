import torch
from scr.ml_utilities import c, h, rng_c, to_array, \
    to_device_tensor, writer


class AliceTrain:
    def __init__(self, alice):
        self.alice = alice
        self.targets = self.alice.run.game_reports.targets
        self.codes = self.alice.run.game_reports.codes
        self.rewards = self.alice.run.game_reports.rewards
        self.decisions = self.alice.run.game_reports.decisions
        self.current_iteration = alice.run.current_iteration


class Basic(AliceTrain):
    def __init__(self, alice):
        super().__init__(alice)

    def __call__(self):
        targets = torch.flatten(self.targets, start_dim=1)
        alice_outputs = self.alice(targets)
        alice_qs = torch.einsum('bj, bj -> b', alice_outputs, self.codes)
        return self.alice.loss_function(alice_qs, self.rewards)


class FromDecisions(AliceTrain):
    def __init__(self, alice):
        super().__init__(alice)

    def __call__(self):
        targets = torch.flatten(self.targets, start_dim=1)
        alice_codes_from_targets = torch.sign(self.alice.net(targets))
        with torch.no_grad():
            decisions = torch.flatten(self.decisions, start_dim=1)
            alice_codes_from_decisions = torch.sign(self.alice.net(decisions))
        closeness = torch.einsum('ij, ij -> i', alice_codes_from_targets,
                                 alice_codes_from_decisions) / c.N_CODE
        if self.current_iteration < h.ALICE_PROXIMITY_BONUS:
            return self.alice.loss_function(closeness, self.rewards)
        if self.current_iteration >= h.ALICE_PROXIMITY_BONUS + \
                h.ALICE_PROXIMITY_SLOPE_LENGTH:
            bonus_prop = 1.
        else:
            bonus_prop = (self.current_iteration - h.ALICE_PROXIMITY_BONUS) / \
                         h.ALICE_PROXIMITY_SLOPE_LENGTH
        closeness_bonus = bonus_prop * (closeness == 1.).float()
        rewards_bonus = bonus_prop * (self.rewards == 1.).float()
        return self.alice.loss_function(closeness + closeness_bonus,
                                        self.rewards + rewards_bonus)



