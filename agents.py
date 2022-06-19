import numpy as np


class ThompsonAgent:

    def __init__(self, n_arms):
        self.beta_alpha = [1 for _ in range(n_arms)]  # beta dist alpha parameter
        self.beta_beta  = [1 for _ in range(n_arms)]  # beta dist beta parameter

    @property
    def reward(self):
        # the reward is the sum of the alphas minus the inital 1s
        return np.sum(self.beta_alpha) - len(self.beta_alpha)

    def update(self, arm_i, reward):
        if reward == 1:
            self.beta_alpha[arm_i] += 1
        elif reward == 0:
            self.beta_beta[arm_i] += 1
        else:
            raise ValueError(f"Unexpected reward value {reward}")

    def choose_action(self):
        beta_samples = np.random.beta(a=self.beta_alpha, b=self.beta_beta)
        return np.argmax(beta_samples)


class Exp3Agent:

    def __init__(self, n_arms, gamma) -> None:
        self.n_arms = n_arms
        self.gamma = gamma
        self.reward = 0
        self.weights = [1 for _ in range(n_arms)]

        self._update_action_probs()

    def _update_action_probs(self):
        self.action_probs = (1 - self.gamma) * np.array(self.weights) / np.array(self.weights).sum() + self.gamma / self.n_arms

    def update(self, arm_i, reward):
        self.reward += reward
        estimated_reward = reward / self.action_probs[arm_i]
        self.weights[arm_i] *= np.exp(self.gamma * estimated_reward / self.n_arms)
        self._update_action_probs()

    def choose_action(self):
        return np.where(np.random.multinomial(n=1, pvals=self.action_probs))[0][0]


# class UCB1Agent:

#     def __init__(self, n_arms) -> None:
#         self.n_arms = n_arms
#         self.action_counts = [0 for _ in range(n_arms)]
#         self.action_rewards = [0 for _ in range(n_arms)]

#     def update(self, arm_i, reward):
#         self.action_counts[arm_i] += 1
#         self.action_rewards[arm_i] += reward

#     def choose_action(self):
#         # return np.argmax(np.array(self.action_rewards) / self.n_arms + np.sqrt(2 * np.log() /))
#         pass
