"""
Simple implementation of approximate Monte Carlo method.

The initial implementation of the algorithm is from the
Reinforcement Learning In Motion series by Manning Publications
https://livevideo.manning.com/module/56_8_5/reinforcement-learning-in-motion/climbing-the-mountain-with-approximation-methods/approximate-monte-carlo-predictions?

"""
from typing import TypeVar

from src.algorithms.algorithm_base import AlgorithmBase
from src.algorithms.algo_config import AlgoConfig

Env = TypeVar("Env")
Policy = TypeVar("Policy")

class ApproxMonteCarloConfig(AlgoConfig):
    def __init__(self):
        super(ApproxMonteCarloConfig, self).__init__()
        self.alpha: float = 0.1
        self.policy: Policy = None


class ApproxMonteCarlo(AlgorithmBase):

    def __init__(self, algo_config: ApproxMonteCarloConfig) -> None:
        super(ApproxMonteCarlo, self).__init__(algo_in=algo_config)
        self.n_itrs_per_episode: int = algo_config.n_itrs_per_episode
        self.alpha: float = algo_config.alpha
        self.policy: Policy = algo_config.policy
        self.weights = {}


    def on_episode(self, **options) -> None:

        episode_reward = 0.0
        counter = 0
        memory = []

        for itr in range(self.n_itrs_per_episode):
            action = self.policy(self.state)

            next_state, reward, done, info = self.train_env.step(action)
            memory.append((state, action, reward))
