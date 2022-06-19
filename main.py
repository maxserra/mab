import numpy as np
from tqdm import tqdm

import environments as envs
import agents as ags


def thompson_sampling(arm_probs, n_games, n_rounds):

    env = envs.MultiArmedBanditEnvironment(arm_probs=arm_probs)

    rewards_list = []

    for _ in tqdm(range(n_games)):

        agent = ags.ThompsonAgent(n_arms=len(arm_probs))

        for _ in range(n_rounds):

            choosen_arm = agent.choose_action()
            reward = env.take_action(arm_i=choosen_arm)
            agent.update(arm_i=choosen_arm, reward=reward)

        rewards_list.append(agent.reward)

    print(np.mean(rewards_list))


def exp3_stochastic_arms(arm_probs, n_games, n_rounds):

    env = envs.MultiArmedBanditEnvironment(arm_probs=arm_probs)

    rewards_list = []

    for _ in tqdm(range(n_games)):

        agent = ags.Exp3Agent(n_arms=len(arm_probs), gamma=0.07)

        for _ in range(n_rounds):

            choosen_arm = agent.choose_action()
            reward = env.take_action(arm_i=choosen_arm)
            agent.update(arm_i=choosen_arm, reward=reward)

        rewards_list.append(agent.reward)

    print(np.mean(rewards_list))


def exp3_adversarial_env(arm_probs, n_games, n_rounds):

    rewards_list = []

    for _ in tqdm(range(n_games)):

        env = envs.AdversarialExp3Environment(arm_probs=arm_probs)
        agent = ags.Exp3Agent(n_arms=len(arm_probs), gamma=0.07)

        for _ in range(n_rounds):

            print(f"Adverarial report: \n\t{env.reward_tracking} | {env.arm_probs}")

            choosen_arm = agent.choose_action()
            reward = env.take_action(arm_i=choosen_arm)
            agent.update(arm_i=choosen_arm, reward=reward)

        rewards_list.append(agent.reward)

    print(np.mean(rewards_list))


def thompson_adversarial_env(arm_probs, n_games, n_rounds):

    rewards_list = []

    for _ in tqdm(range(n_games)):

        env = envs.AdversarialExp3Environment(arm_probs=arm_probs)
        agent = ags.ThompsonAgent(n_arms=len(arm_probs))

        for _ in range(n_rounds):

            print(f"Adverarial report: \n\t{env.reward_tracking} | {env.original_arm_probs}")

            choosen_arm = agent.choose_action()
            reward = env.take_action(arm_i=choosen_arm)
            agent.update(arm_i=choosen_arm, reward=reward)

        rewards_list.append(agent.reward)

    print(np.mean(rewards_list))


if __name__ == '__main__':

    # thompson_sampling(arm_probs=[0.3, 0.5, 0.4, 0.45, 0.3, 0.35],
    #                  n_games=30, n_rounds=1000)

    # exp3_stochastic_arms(arm_probs=[0.3, 0.5, 0.4, 0.45, 0.3, 0.35],
    #                      n_games=30, n_rounds=1000)

    # exp3_adversarial_env(arm_probs=[0.3, 0.5, 0.4, 0.45, 0.3, 0.35],
    #                      n_games=10, n_rounds=1000)

    thompson_adversarial_env(arm_probs=[0.3, 0.5, 0.4, 0.45, 0.3, 0.35],
                             n_games=10, n_rounds=1000)
