import numpy as np


class MultiArmedBanditEnvironment:

    def __init__(self, arm_probs):
        self.arm_probs = arm_probs

    def take_action(self, arm_i):
        if arm_i >= 0 and arm_i < len(self.arm_probs):
            return np.random.binomial(n=1, p=self.arm_probs[arm_i])
        else:
            raise ValueError(f"The given value '{arm_i}' is invalid. Must be positive and below '{len(self.arm_probs)}'")
