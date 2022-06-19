import numpy as np


class MultiArmedBanditEnvironment:

    def __init__(self, arm_probs):
        self.arm_probs = arm_probs

    def take_action(self, arm_i):
        if arm_i >= 0 and arm_i < len(self.arm_probs):
            return np.random.binomial(n=1, p=self.arm_probs[arm_i])
        else:
            raise ValueError(f"The given value '{arm_i}' is invalid. Must be positive and below '{len(self.arm_probs)}'")


class AdversarialExp3Environment:

    def __init__(self, arm_probs):
        self.original_arm_probs = arm_probs
        self.arm_probs = arm_probs

        self.reward_tracking = [0 for _ in range(len(arm_probs))]

    def _update_reward_tracking(self, arm_i, reward):
        self.reward_tracking[arm_i] += reward

    def _temper_with_arm_probs(self):
        sorted_probs = np.flip(np.sort(self.original_arm_probs))
        sorted_arms_by_reward = np.argsort(self.reward_tracking)

        j = 0
        for i in sorted_arms_by_reward:
            self.arm_probs[i] = sorted_probs[j]
            j += 1

        print(f"Sorted probs {sorted_probs} | {sorted_arms_by_reward} -> {self.arm_probs}")

    def take_action(self, arm_i):
        if arm_i >= 0 and arm_i < len(self.arm_probs):
            reward = np.random.binomial(n=1, p=self.arm_probs[arm_i])
        else:
            raise ValueError(f"The given value '{arm_i}' is invalid. Must be positive and below '{len(self.arm_probs)}'")

        self._update_reward_tracking(arm_i, reward)
        self._temper_with_arm_probs()

        return reward
