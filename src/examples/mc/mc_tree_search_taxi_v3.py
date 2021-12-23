"""
Monte Carlo tree search on Taxi-v3
This example is edited from
https://github.com/ashishrana160796/prototyping-self-driving-agents/blob/master/milestone-four/monte_carlo_tree_search_taxi_v3.ipynb
"""
import gym
import copy
import random
import itertools
import sys
from time import time

from src.algorithms.planning.monte_carlo_tree_search import MCTreeSearch
from src.algorithms.planning.monte_carlo_tree_search import MCTreeSearchInput
from src.algorithms.planning.monte_carlo_tree_search import MCTreeNode


def moving_averages(v, n):
    n = min(len(v), n)
    ret = [.0]*(len(v)-n+1)
    ret[0] = float(sum(v[:n]))/n
    for i in range(len(v)-n):
        ret[i+1] = ret[i] + float(v[n+i] - v[i])/n
    return ret


class TaxiMCTreeSearch(MCTreeSearch):

    def __init__(self, algo_in: MCTreeSearchInput) -> None:
        super(TaxiMCTreeSearch, self).__init__(algo_in=algo_in)
        self.best_rewards = []
        self.best_actions = []
        self.start_time = None

    def reset(self) -> None:
        super(TaxiMCTreeSearch, self).reset()
        self.root = MCTreeNode(None, None)
        self.best_actions = []

    def upper_conf_bound(self, node: MCTreeNode) -> float:
        return node.ucb(c=self._c)

    def print_stats(self, num_exec, score, avg_time):
        sys.stdout.write(
            'execution number: \r%3d   total reward:%10.3f   average time:%4.1f s' % (num_exec, score, avg_time))
        sys.stdout.flush()
        if (num_exec % 10) == 0:
            print("execution number: \r%4d   total reward: %4.3f   average time: %4.2f s" % (num_exec, score, avg_time))

    def actions_before_training_begins(self, **options) -> None:
        super(TaxiMCTreeSearch, self).actions_before_training_begins(**options)
        self.start_time = time()

    def actions_after_episode_ends(self, **options):
        super(TaxiMCTreeSearch, self).actions_after_episode_ends()

        score = max(moving_averages(self.best_rewards, 100))
        avg_time = (time() - self.start_time) / (self.current_episode_index + 1)
        self.print_stats(self.current_episode_index + 1, score, avg_time)

    # This function determine complete exhaustive list of all the nodes.
    def node_expansion(self, space):
        if isinstance(space, gym.spaces.Discrete):
            return range(space.n)
        elif isinstance(space, gym.spaces.Tuple):
            return itertools.product(*[self.node_expansion(s) for s in space.spaces])
        else:
            raise NotImplementedError

    def backprop(self, node: MCTreeNode, **options):

        while node:
            node.visits += 1
            node.value += options['sum_reward']
            node = node.parent

    def step(self, **options) -> None:

        best_reward = float("-inf")

        # play the episode
        for itr in range(self.n_itrs_per_episode):

            sum_reward = 0
            terminal = False
            actions = []
            state = copy.copy(self.train_env)
            node = self.root

            while node.children:

                if node.explored_children < len(node.children):
                    child = node.children[node.explored_children]
                    node.explored_children += 1
                    node = child
                else:
                    node = max(node.children, key=self.upper_conf_bound)

                _, reward, terminal, _ = state.step(node.action)
                sum_reward += reward
                actions.append(node.action)

            # expansion of all the children nodes
            if not terminal:
                node.children = [MCTreeNode(parent=node, action=a) for a in self.node_expansion(state.action_space)]
                random.shuffle(node.children)

            # creating exhaustive list of actions
            while not terminal:
                action = state.action_space.sample()
                _, reward, terminal, _ = state.step(action)
                sum_reward += reward
                actions.append(action)

                if len(actions) > self.max_tree_depth:
                    sum_reward -= 100
                    break

            # retaining the best reward value and actions
            if best_reward < sum_reward:
                    best_reward = sum_reward
                    best_actions = actions

            # backpropagating in MCTS for assigning reward value to a node.
            self.backprop(node=node, **{'sum_reward': sum_reward})

        sum_reward = 0
        for action in self.best_actions:
            _, reward, terminal, _ = env.step(action)
            sum_reward += reward
            if terminal:
                break

        self.best_rewards.append(sum_reward)


if __name__ == '__main__':

    env = gym.make("Taxi-v3")

    algo_input = MCTreeSearchInput()
    algo_input.train_env = env
    algo_input.render_env = True
    algo_input.n_episodes = 5000
    algo_input.n_itrs_per_episode = 100
    algo_input.c = 1.5
    algo_input.max_tree_depth = 1000

    agent = MCTreeSearch(algo_in=algo_input)
    agent.train()



