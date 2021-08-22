"""
Value iteration algorithm.
Implementation refactored from
https://github.com/udacity/deep-reinforcement-learning

"""

import collections
from typing import Any

from algorithms.dp.dp_algorithm_base import DPAlgoBase
from algorithms.dp.policy_improvement import PolicyImprovement
from algorithms.dp.utils import state_actions_from_v as q_s_a
from policies.policy_base import PolicyBase
from policies.policy_adaptor_base import PolicyAdaptorBase


class ValueIteration(DPAlgoBase):
    """
    Value iteration algorithm encapsulated into a class
    The algorithm has two similar implementations regulated
    by the train_mode enum. When train_mode = TrainMode.DEFAULT
    the implementation from Sutton & Barto is used. When
    train_mode = TrainMode.STOCHASTIC the algorithm
    will query the environment to sample an action. Establishment
    of the state value table is done after the episodes are finished
    based on the counters accumulated in self._rewards and self._transits.
    Thus, in the later implementation, the environment is not queried
    for the dynamics i.e. self.env.P[state][action] as is done in the
    former implementation.
    """

    def __init__(self, n_max_iterations: int, tolerance: float,
                 env: Any, gamma: float, policy_init: PolicyBase,
                 policy_adaptor: PolicyAdaptorBase) -> None:
        """
        Constructor
        """
        super(ValueIteration, self).__init__(n_max_iterations=n_max_iterations, gamma=gamma,
                                             tolerance=tolerance, env=env, policy=policy_init)

        self._p_imprv = PolicyImprovement(env=env, v=self.v, gamma=gamma,
                                          policy_init=policy_init, policy_adaptor=policy_adaptor)

        self._rewards = collections.defaultdict(float)
        self._transits = collections.defaultdict(collections.Counter)

    def step(self, **options) -> None:
        """
        Do one step .
        """
        delta = 0
        for s in range(self.train_env.observation_space.n):
            v = self.v[s]
            self.v[s] = max(q_s_a(env=self.train_env, v=self.v, state=s, gamma=self.gamma))
            delta = max(delta, abs(self.v[s] - v))

        self.itr_control.residual = delta

        self._p_imprv.v = self.v
        self._p_imprv.step()
        self.policy = self._p_imprv.policy

    '''
    def select_action(self, state: int, update_tables: bool=False) -> int:
            best_action, best_value = None, None
            for action in range(self.train_env.action_space.n):
                action_value = self._calc_action_value(state, action, update_tables=update_tables)

                if best_value is None or best_value < action_value:
                    best_value = action_value
                    best_action = action
            return best_action

    def _step_stochastic(self):
        """
        Do a stochastic step in the environment by
        sampling an action from the  action space
        :return:
        """
        action = self.train_env.action_space.sample()
        new_state, reward, is_done, _ = self.train_env.step(action)
        self._rewards[(self.state, action, new_state)] = reward
        self._transits[(self.state, action)][new_state] += 1
        self.state = self.train_env.reset() if is_done else new_state

    def _step_default(self):
        """
        Default implementation of value iteration
        as described in Sutton & Barto
        """

        # stop condition
        delta = 0.0

        # update each state
        for s in range(self.train_env.nS):
            # Do a one-step lookahead to find the best action
            values = self._one_step_lookahead(state=s)
            best_action_value = np.max(values)

            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - self._v[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            self.values[s] = best_action_value

        # Check if we can stop
        self.residual = delta

    def _one_step_lookahead(self,  state: int, update_tables: bool=True):
        """
        Helper function used in _step_default
        to calculate the value for all actions in a given state.
        """
        values = np.zeros(self.train_env.nA)
        for action in range(self.train_env.nA):
            values[action] = self._calc_action_value(state=state, action=action,
                                                     update_tables=update_tables)

        return values

    def _calc_action_value(self, state: int, action: int, update_tables: bool=False) -> float:
        """
        Returns the action value for the given state and action.
        Note that when TrainMode.DEFAULT this also updates
        the self._rewards,  and self._transits tables when
        the update_tables boolean flag is True
        """
        action_value = 0.0
        if self._train_mode == TrainMode.DEFAULT:

            for prob, next_state, reward, done in self.train_env.P[state][action]:
                val = reward + self.gamma * self.values[next_state]
                action_value += prob * val

                # populate the rewards and the transitions
                # this may be meaningless here as in TrainMode.DEFAULT
                # we loop over all states

                if update_tables:
                    self._rewards[(state, action, next_state)] = reward
                    self._transits[(state, action)][next_state] += 1
            return action_value
        elif self._train_mode == TrainMode.STOCHASTIC:

            target_counts = self._transits[(state, action)]
            total = sum(target_counts.values())

            for tgt_state, count in target_counts.items():
                reward = self._rewards[(state, action, tgt_state)]
                val = reward + self.gamma * self.values[tgt_state]
                action_value += (count / total) * val
            return action_value
        else:
            raise ValueError("Invalid train mode. "
                             "Mode {0} not in [{1}, {2}]".format(self._train_mode,
                                                                 TrainMode.DEFAULT.name, TrainMode.STOCHASTIC.name))

    def _value_iteration(self):
        """
        Establish the value function table when using
        stochastic implementation
        :return:
        """
        for state in range(self.train_env.observation_space.n):
            state_values = [
                self._calc_action_value(state, action)
                for action in range(self.train_env.action_space.n)
            ]
            self.values[state] = max(state_values)
    '''



