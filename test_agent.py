import numpy as np

import torch
from ddpg_agent import DDPGAgent
from unityagents import UnityEnvironment

torch.set_num_threads(1)


class TestRunner:
    def __init__(self, env_path: str, checkpoint_path: str):
        self.env = UnityEnvironment(file_name=env_path)
        self.brain_name = self.env.brain_names[0]
        self.agent = DDPGAgent(state_size=33, action_size=4, random_seed=42)
        self.checkpoint_path = checkpoint_path

    def run(self) -> None:
        # load the weights from file
        self.agent.actor_local.load_state_dict(
            torch.load(f"{self.checkpoint_path}/checkpoint_actor.pth"))
        self.agent.critic_local.load_state_dict(
            torch.load(f"{self.checkpoint_path}/checkpoint_critic.pth"))

        rewards = np.zeros(20)

        for i in range(10):
            # reset the environment
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            state = env_info.vector_observations  # get the current state
            self.agent.reset()
            for j in range(200):
                action = self.agent.act(state, add_noise=False)

                # send the action to the environment
                env_info = self.env.step(action)[self.brain_name]
                state = env_info.vector_observations  # get the next state
                reward = env_info.rewards  # get the reward
                done = env_info.local_done  # see if episode has finished
                rewards += reward

                print("rewards: ", rewards)
                if np.any(done):
                    break

        self.env.close()


if __name__ == '__main__':
    # TestRunner("./reacher/Reacher_Linux/Reacher.x86_64", "./reacher").run()
    TestRunner("./reacher20/Reacher_Linux/Reacher.x86_64", "./reacher20").run()
