"""
Learner for controlling a differential drive robot
"""
from typing import TypeVar
from src.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoInput
from src.utils.mixins import WithMaxActionMixin, WithQTableMixin


Policy = TypeVar("Policy")


class DiffDriveRobotQLearner(TDAlgoBase, WithQTableMixin, WithMaxActionMixin):

    def __init__(self, algo_in: TDAlgoInput) -> None:
        super(DiffDriveRobotQLearner, self).__init__(algo_in=algo_in)
        self.q_table = {}
        self.policy: Policy = algo_in.policy


    def actions_before_training_begins(self, **options) -> None:
        super(DiffDriveRobotQLearner, self).actions_before_training_begins(**options)
        self.q_table = {}

    def on_episode(self, **options) -> None:

        # episode score
        episode_score = 0  # initialize score
        counter = 0

        for itr in range(self.n_itrs_per_episode):

            # epsilon-greedy action selection
            action = self.policy(self.q_table, state=self.state)


