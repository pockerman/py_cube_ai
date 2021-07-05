import gym
from tensorboardX import SummaryWriter

from algorithms.q_iteration import QIteration
from algorithms.algorithm_base import TrainMode
from utils import INFO


class Agent(QIteration):

    def __init__(self, env_name: str, gamma: float) -> None:
        super(Agent, self).__init__(env=gym.make(env_name), gamma=gamma,
                                    update_values_on_start_itrs=False,
                                    n_max_itrs=100, tolerance=1.0e-10,
                                    train_mode=TrainMode.STOCHASTIC)
        self.state = self.train_env.reset()

    def play_episode(self, test_env):
        total_reward = 0.0
        state = test_env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = test_env.step(action)
            self._rewards[(state, action, new_state)] = reward
            self._transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == '__main__':

    ENV_NAME = "FrozenLake-v0"
    GAMMA = 0.9
    TEST_EPISODES = 20

    test_env = gym.make(ENV_NAME)

    agent = Agent(gamma=GAMMA, env_name=ENV_NAME)
    writer = SummaryWriter(comment="-q-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:

        iter_no += 1
        agent.train()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("%s Best reward updated %.3f -> %.3f" % (INFO, best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("%s Solved in %d iterations!" % (INFO, iter_no))
            break
    writer.close()
