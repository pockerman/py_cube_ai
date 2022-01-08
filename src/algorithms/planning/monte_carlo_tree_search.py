"""
Basic implementation of Monte Carlo tree search
"""
import math
from typing import TypeVar, List, Any
import abc

from src.utils.exceptions import Error, InvalidParameterValue
from src.algorithms.algorithm_base import AlgorithmBase
from src.algorithms.algo_input import AlgoInput

Action = TypeVar("Action")
State = TypeVar("State")
MCTreeNode = TypeVar("MCTreeNode")


class MCTreeSearchInput(AlgoInput):
    def __init__(self):
        super(MCTreeSearchInput, self).__init__()

        # the temperature coefficient
        # used when computing UCB
        self.c: float = 1.5

        # maximum tree depth allowed
        self.max_tree_depth: int = 0


class MCTreeNode(object):
    """
    The MCTreeNode class. Represents a node in the MCTree
    """
    def __init__(self, parent: MCTreeNode, action: Action, **kwargs):
        self.parent: MCTreeNode = parent
        self.action: Action = action
        self.state: State = None
        self.children: List[MCTreeNode] = []
        self.explored_children: int = 0
        self.total_score: float = 0.0
        self.total_visits: int = 0
        self.kwargs = kwargs

    def add_child(self, child: MCTreeNode) -> MCTreeNode:
        if child is None:
            raise Error("Attempt to add null child")

        self.children.append(child)

    def update_rollout_stats(self) -> None:
        self.total_visits += 1

    def ucb(self, c: float) -> float:
        """
        Compute the UCT on this node
        :param win_pct:
        :param c:
        :return:
        """
        total_rollouts = sum(child.total_visits for child in self.children)
        return self.win_pct + c*math.sqrt(math.log(self.parent.total_visits) / self.total_visits)

    @property
    def win_pct(self) -> float:
        return self.total_score / self.total_visits


class MCTreeSearch(AlgorithmBase):
    """
    The MCTreeSearch class. Basic implementation of Monte Carlo tree search
    """
    
    def __init__(self, algo_in: MCTreeSearchInput):
        super(MCTreeSearch, self).__init__(algo_in=algo_in)

        self.n_itrs_per_episode = algo_in.n_itrs_per_episode
        self._c = algo_in.c
        self.max_tree_depth = algo_in.max_tree_depth

        # the root of the tree
        self.root = MCTreeNode(parent=None, action=None)

    def reset(self) -> None:
        """
        Reset the underlying data
        """
        super(MCTreeSearch, self).reset()
        self.root = MCTreeNode(parent=None, action=None)

    def simulate_random_game(self):
        pass

    def select_child(self, node: MCTreeNode) -> MCTreeNode:
        """
        Select the child node to follow from the given node
        :param node:
        :return:
        """

        best_score = -1
        best_child = None
        for child in node.children:
            score = child.ucb(self._c)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self):
        """
        Add a new node to the tree
        :return:
        """
        pass

    @abc.abstractmethod
    def backprop(self, node: MCTreeNode, **options: Any) -> None:

        """
        Update your node values up to the root node from the traversed path.
        The number of reward-wins is incremented
        after this value update is increased across the nodes
        in the simulation count stored in the nodes, and if the new node is the desired goal.
        :return:
        """

    def actions_before_training_begins(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the training
        """
        super(MCTreeSearch, self).actions_before_training_begins(**options)

        if self.n_itrs_per_episode == 0:
            raise InvalidParameterValue(param_name="n_itrs_per_episode", param_val=self.n_itrs_per_episode)

    def actions_after_training_ends(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass
