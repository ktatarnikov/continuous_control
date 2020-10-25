from collections import deque

import numpy as np

import torch
from ddpg_agent import DDPGAgent
from unityagents import UnityEnvironment

torch.set_num_threads(1)


class TestRunner:
    def __init__(self, agent_count: int, env_path: str, checkpoint_path: str):
        self.agent_count = agent_count
        self.env = UnityEnvironment(file_name=env_path)
        self.brain_name = self.env.brain_names[0]
        self.agent = DDPGAgent(state_size=33, action_size=4, random_seed=42)
        self.checkpoint_path = checkpoint_path

    def run(self) -> None:
        self.agent.actor_local.load_state_dict(
            torch.load(f"{self.checkpoint_path}/checkpoint_actor.pth"))
        self.agent.critic_local.load_state_dict(
            torch.load(f"{self.checkpoint_path}/checkpoint_critic.pth"))

        scores_deque = deque(maxlen=100)
        score_average = 0

        for i_episode in range(10):
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            state = env_info.vector_observations  # get the current state
            self.agent.reset()

            score = np.zeros(self.agent_count)
            for j in range(1000):
                action = self.agent.act(state, add_noise=False)

                # send the action to the environment
                env_info = self.env.step(action)[self.brain_name]
                state = env_info.vector_observations  # get the next state
                reward = env_info.rewards  # get the reward
                done = env_info.local_done  # see if episode has finished
                score += reward

                if np.any(done):
                    break

            scores_deque.append(np.mean(score))
            score_average = np.mean(scores_deque)
            score_stddev = np.std(scores_deque)
            print(
                f'Episode {i_episode} - Average Score: {score_average:.2f}, StdDev: {score_stddev}'
            )

        self.env.close()
