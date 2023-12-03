"""
Reinforce class. Class based implementation of the
REINFORCE algorithm.

The first  implementation was basically a wrapper
of the implementation from Udacity Deep RL repository. The second
version of the implementation is more in tandem
with the Reinforcement Learning in Action book

"""

import collections
from collections import deque
import torch
import numpy as np
from typing import Any, TypeVar, Callable
from pycubeai.algorithms.algorithm_base import AlgorithmBase
from pycubeai.algorithms.algo_config import AlgoConfig
from pycubeai.utils.replay_buffer import ReplayBuffer
from pycubeai.utils.exceptions import InvalidParameterValue

Optimizer = TypeVar("Optimizer")
PolicyNet = TypeVar("PolicyNet")
ActionSelector = TypeVar('ActionSelector')
Env = TypeVar('Env')


class ReinforceConfig(AlgoConfig):
    def __init__(self) -> None:
        super(ReinforceConfig, self).__init__()
        self.gamma = 1.0
        self.n_itrs_per_episode = 100
        self.queue_length = 100
        self.mean_reward_for_exit: float = -1.0
        self.optimizer: Optimizer = None
        self.policy_network: PolicyNet = None
        self.loss_func: Callable = None
        self.action_selector: ActionSelector = None


class Reinforce(AlgorithmBase):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient. The main
    ingredients of this class are the policy_network model and
    the optimizer model. The policy_network has to be a differentiable
    function a.k.a a torch.Module object
    """

    def __init__(self, algo_in: ReinforceConfig) -> None:
        super(Reinforce, self).__init__(algo_in=algo_in)
        self.n_itrs_per_episode = algo_in.n_itrs_per_episode
        self.mean_reward_for_exit: float = algo_in.mean_reward_for_exit
        self.gamma: float = algo_in.gamma
        #self.scores = []
        #self.scores_deque = deque(maxlen=algo_in.queue_length)
        #self.saved_log_probs = []
        self.total_rewards = []
        self.iterations_per_episode = []

        self.policy_network: PolicyNet = algo_in.policy_network
        self.optimizer = algo_in.optimizer
        self.loss_func: Callable = algo_in.loss_func
        self.memory: ReplayBuffer = ReplayBuffer(buffer_size=self.n_itrs_per_episode)
        self.action_selector: ActionSelector = algo_in.action_selector

    def play(self, env: Env, n_games: int, max_duration_per_game: int) -> None:

        score = []
        done = False
        state1 = env.reset()
        for i in range(n_games):
            t = 0
            while not done:  # F
                probs = self.policy_network.on_state(state1) #(torch.from_numpy(state1).float())
                action = self.action_selector(probs) #np.random.choice(np.array([0, 1]), p=pred.data.numpy())
                state2, reward, done, info = env.step(action)
                env.render(mode='rgb_array')
                state1 = state2
                t += 1
                if t > max_duration_per_game:
                    break
            state1 = env.reset()
            done = False
            score.append(t)
        score = np.array(score)

    def actions_before_training_begins(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """
        super(Reinforce, self).actions_before_training_begins(**options)

        if self.n_itrs_per_episode <= 0:
            raise InvalidParameterValue(param_name="n_itrs_per_episode", param_val=self.n_itrs_per_episode)

        if self.optimizer is None:
            raise InvalidParameterValue(param_name="optimizer", param_val="None")

        if self.loss_func is None:
            raise InvalidParameterValue(param_name="loss_func", param_val="None")

        if self.action_selector is None:
            raise InvalidParameterValue(param_name="action_selector", param_val="None")

        self.memory = ReplayBuffer(buffer_size=self.n_itrs_per_episode)
        self.total_rewards = []
        self.iterations_per_episode = []

    def actions_after_episode_ends(self, **options):
        """
        Algorithm specific actions after the episode ends
        :param options:
        :return:
        """
        # execute any base class actions
        super(Reinforce, self).actions_after_episode_ends(**options)

        # Reinforce specific actions

        # calculate discounts note that we do not
        # use the rewards simply calculate the discount
        times = [x["T"] for x in self.memory["info"]]

        # compute discount coeffs
        discounts = [self.gamma ** i for i in range(len(times) + 1)]

        rewards = self.memory["reward"]

        # compute discounted rewards
        disc_returns = sum([a * b for a, b in zip(discounts, rewards)]) #elf.rewards)])

        # Collect the states in the episode in a single tensor
        state_batch = self.memory.get_item_as_torch_tensor(name_attr="state")

        # Collect the actions in the episode in a single tensor
        action_batch = self.memory.get_item_as_torch_tensor(name_attr="action")

        # Re-compute the action probabilities for all the states in the episode
        pred_batch = self.policy_network(state_batch)

        # Subset the action-probabilities
        # associated with the actions that were actually taken
        prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()

        loss = self.loss_func(prob_batch, disc_returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # check about setting the break flag
        self.total_rewards.append(sum(rewards))

        # check if we have set a goal
        # for learning
        if self.mean_reward_for_exit > 0:
            if np.mean(self.total_rewards) >= self.mean_reward_for_exit:
                self.break_training_flag = True

    def actions_after_training_ends(self, **options) -> None:
        pass

    def on_episode(self, **options) -> None:
        """
        Run the algorithm on the episode. Most of the work
        is don in the actions_after_episode_ends. This
        function simply steps in the environment according
        to the action given bu the policy network and collects
        the statistics in the memory buffer
        :param options:
        :return:
        """

        total_episode_itr_counter = 0

        for itr in range(self.n_itrs_per_episode):

            # use the policy network to generate
            # the action probability distribution
            probs = self.policy_network.on_state(self.state) #act(state=self.state)

            # given the probability distribution
            # use it to generate an action. We use
            # a generic approach so that applications
            # can establish their own way for that
            action = self.action_selector(probs)

            ## TODO need a way to transform to log probabilities
            #self.saved_log_probs.append(log_prob)
            state, reward, done, _ = self.train_env.step(action)

            if self.render_env and \
                    (self.current_episode_index % self.render_env_freq) == 0:
                self.train_env.render(mode='rgb_array')

            total_episode_itr_counter += 1

            # add item in the memory
            self.memory.add(state=self.state, reward=reward, action=action,
                            next_state=state, done=done, info={"T": itr + 1})

            # update the state
            self.state = state
            if done:
                break

        self.iterations_per_episode.append(total_episode_itr_counter)

    def _discount_rewards(self, rewards:torch.Tensor) -> torch.Tensor:

        lenr = len(rewards)

        # A Compute exponentially decaying rewards
        disc_return = torch.pow(self.gamma, torch.arange(lenr).float()) * rewards

        # Normalize the rewards to be within the [0,1] interval to improve numerical stability
        disc_return /= disc_return.max()
        return disc_return



