import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import environments as envs
import agents as ags


def thompson_sampling(arm_probs, n_games, n_rounds):

    env = envs.MultiArmedBanditEnvironment(arm_probs=arm_probs)

    rewards_list = []
    weights_list = []

    for _ in tqdm(range(n_games)):

        agent = ags.ThompsonAgent(n_arms=len(arm_probs))

        temp_rewards_list = []
        temp_weights_list = []

        for _ in range(n_rounds):

            choosen_arm = agent.choose_action()
            reward = env.take_action(arm_i=choosen_arm)
            agent.update(arm_i=choosen_arm, reward=reward)

            temp_rewards_list.append(agent.reward)
            temp_weights_list.append([agent.beta_alpha, agent.beta_beta])

        rewards_list.append(temp_rewards_list)
        weights_list.append(temp_weights_list)

    return (rewards_list, weights_list)


def exp3_stochastic_arms(arm_probs, gamma=0.07, n_games=1, n_rounds=1000):

    env = envs.MultiArmedBanditEnvironment(arm_probs=arm_probs)

    rewards_list = []
    weights_list = []

    for _ in tqdm(range(n_games)):

        agent = ags.Exp3Agent(n_arms=len(arm_probs), gamma=gamma)

        temp_rewards_list = []
        temp_weights_list = []

        for _ in range(n_rounds):

            choosen_arm = agent.choose_action()
            reward = env.take_action(arm_i=choosen_arm)
            agent.update(arm_i=choosen_arm, reward=reward)

            temp_rewards_list.append(agent.reward)
            temp_weights_list.append(agent.action_probs)

        rewards_list.append(temp_rewards_list)
        weights_list.append(temp_weights_list)

    return (rewards_list, weights_list)


def exp3_adversarial_env(arm_probs, gamma=0.07, n_games=1, n_rounds=1000):

    rewards_list = []
    weights_list = []

    for _ in tqdm(range(n_games)):

        env = envs.AdversarialExp3Environment(arm_probs=arm_probs)
        agent = ags.Exp3Agent(n_arms=len(arm_probs), gamma=gamma)

        temp_rewards_list = []
        temp_weights_list = []

        for _ in range(n_rounds):

            # print(f"Adversarial report: \n\t{env.reward_tracking}")

            choosen_arm = agent.choose_action()
            reward = env.take_action(arm_i=choosen_arm)
            agent.update(arm_i=choosen_arm, reward=reward)

            temp_rewards_list.append(agent.reward)
            temp_weights_list.append(agent.action_probs)

        rewards_list.append(temp_rewards_list)
        weights_list.append(temp_weights_list)

    return (rewards_list, weights_list)


def thompson_adversarial_env(arm_probs, n_games, n_rounds):

    rewards_list = []
    weights_list = []

    for _ in tqdm(range(n_games)):

        env = envs.AdversarialExp3Environment(arm_probs=arm_probs)
        agent = ags.ThompsonAgent(n_arms=len(arm_probs))

        temp_rewards_list = []
        temp_weights_list = []

        for _ in range(n_rounds):

            print(f"Adverarial report: \n\t{env.reward_tracking}")

            choosen_arm = agent.choose_action()
            reward = env.take_action(arm_i=choosen_arm)
            agent.update(arm_i=choosen_arm, reward=reward)

            temp_rewards_list.append(agent.reward)
            temp_weights_list.append([agent.beta_alpha, agent.beta_beta])

        rewards_list.append(temp_rewards_list)
        weights_list.append(temp_weights_list)

    return (rewards_list, weights_list)


if __name__ == '__main__':

    # rewards_list, weights_list = thompson_sampling(arm_probs=[0.3, 0.5, 0.4, 0.45, 0.3, 0.35],
    #                                                n_games=30, n_rounds=1000)

    # rewards_list, weights_list = exp3_stochastic_arms(arm_probs=[0.3, 0.5, 0.4, 0.45, 0.3, 0.35],
    #                                                   gamma=0.07, n_games=30, n_rounds=5000)

    rewards_list, weights_list = exp3_adversarial_env(arm_probs=[0.3, 0.5, 0.4, 0.45, 0.3, 0.35],
                                                      gamma=0.07, n_games=10, n_rounds=5000)

    # rewards_list, weights_list = thompson_adversarial_env(arm_probs=[0.3, 0.5, 0.4, 0.45, 0.3, 0.35],
    #                                                       n_games=10, n_rounds=1000)
