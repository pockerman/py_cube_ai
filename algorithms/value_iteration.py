"""
Value iteration algorithm.
Code edited from Deep Reinforcement Learning Hands-On
by Maxim Lapan and from https://github.com/dennybritz/reinforcement-learning
"""
import numpy as np
import collections
from algorithms.algorithm_base import AlgorithmBase, TrainMode


class ValueIteration(AlgorithmBase):
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

    def __init__(self, env, gamma: float, train_mode: TrainMode = TrainMode.DEFAULT,
                 n_max_itrs: int = 1000, tolerance: float = 1.0e-5,
                 update_values_on_start_itrs: bool = True) -> None:
        """
        Constructor
        :param env:  environment: OpenAI env
        :param tolerance: We stop evaluation once our value
        function change is less than tolerance for all states.
        :param gamma: Gamma discount factor
        """
        super(ValueIteration, self).__init__(n_max_iterations=n_max_itrs,
                                             tolerance=tolerance, env=env)

        self._gamma = gamma
        self._v = collections.defaultdict(float)
        self._policy = None
        self._rewards = collections.defaultdict(float)
        self._transits = collections.defaultdict(collections.Counter)
        self._update_values_on_start_itrs = update_values_on_start_itrs
        self._train_mode: TrainMode = train_mode

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def values(self) -> collections.defaultdict:
        return self._v

    def actions_before_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """

        # call base class typically this should
        # reset the environment
        super(ValueIteration, self).actions_before_training_iterations(**options)

        if self._update_values_on_start_itrs:
            self._v = collections.defaultdict(float)

    def actions_before_stepping(self, **options) -> None:
        """
        Actions to be performed before episodes start
        """
        pass

    def actions_after_training_iterations(self, **options) -> None:
        """
        Actions to be performed after episodes finish
        """
        if self._train_mode == TrainMode.STOCHASTIC:
            # if using a stochastic train mode establish
            # the value function table
            self._value_iteration()

    def step(self, **options) -> None:
        """
        Value Iteration Algorithm.
        """

        if self._train_mode == TrainMode.DEFAULT:
            self._step_default()
        elif self._train_mode == TrainMode.STOCHASTIC:
            self._step_stochastic()
        else:
            raise ValueError("Invalid train mode. "
                             "Mode {0} not in [{1}, {2}]".format(self._train_mode,
                                                                 TrainMode.DEFAULT.name, TrainMode.STOCHASTIC.name))

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




