import copy
import torch

from typing import TypeVar
from src.algorithms.algorithm_base import AlgorithmBase
from src.algorithms.algo_config import AlgoConfig
from src.utils.replay_buffer import ReplayBuffer
from src.optimization.pytorch_optimizer_builder import pytorch_optimizer_builder
from src.utils import INFO

Optimizer = TypeVar('Optimizer')
Loss = TypeVar('Loss')
Policy = TypeVar('Policy')


class DQNGridWorldConfig(AlgoConfig):
    def __init__(self):
        super(DQNGridWorldConfig, self).__init__()

        # the layers sizes
        self.l1: int = 64
        self.l2: int = 150
        self.l3: int = 100
        self.l4: int = 4
        self.synchronize_frequency: int = 50
        self.memory_size: int = 100
        self.batch_size: int = 200
        self.max_moves: int = 50
        self.learning_rate: float = 0.001
        self.gamma: float = 0.9
        self.optimizer: Optimizer = None
        self.loss: Loss = None
        self.policy: Policy = None


class DQNTargetNetworkGridWorld(AlgorithmBase):

    def __init__(self, config: DQNGridWorldConfig):
        super(DQNTargetNetworkGridWorld, self).__init__(algo_in=config)

        self.n_itrs_per_episode = config.n_itrs_per_episode
        self.max_moves = config.max_moves
        self.batch_size = config.batch_size
        self.synchronize_frequency = config.synchronize_frequency
        self.gamma = config.gamma
        self.net1 = torch.nn.Sequential(torch.nn.Linear(config.l1, config.l2), torch.nn.ReLU(),
                                   torch.nn.Linear(config.l2, config.l3), torch.nn.ReLU(), torch.nn.Linear(config.l3, config.l4))
        self.net2 = copy.deepcopy(self.net1)
        self.net2.load_state_dict(self.net1.state_dict())
        self.memory: ReplayBuffer = ReplayBuffer(buffer_size=config.memory_size)
        self.optimizer = pytorch_optimizer_builder(opt_type=config.optimizer, model_params=self.net1.parameters(),
                                                   **{"learning_rate": config.learning_rate})
        self.loss_fn = config.loss
        self.policy = config.policy

        self.iterations = []
        self.rewards = []
        self.losses = []
        self._update_net_counter = 0

    def actions_before_training_begins(self, **options) -> None:
        super(DQNTargetNetworkGridWorld, self).actions_before_training_begins(**options)

        self.iterations = []
        self.rewards = []
        self.losses = []
        self._update_net_counter = 0

    def actions_before_episode_begins(self, **options) -> None:
        super(DQNTargetNetworkGridWorld, self).actions_before_episode_begins()
        self.state = torch.from_numpy(self.state.observation).float()

    def actions_after_training_ends(self, **options) -> None:
        pass

    def actions_after_episode_ends(self, **options):
        pass

    def on_episode(self, **options) -> None:

        mov = 0
        episode_total_reward = 0
        episode_n_itrs = 0

        for itr in range(self.n_itrs_per_episode):

            self._update_net_counter += 1

            qval = self.net1(self.state).data.numpy()
            action_idx = self.policy.choose_action_index(qval)
            action = self.train_env.get_action(action_idx)

            next_time_step = self.train_env.step(action)
            state = torch.from_numpy(next_time_step.observation).float()
            reward = next_time_step.reward

            episode_total_reward += reward
            done = True if reward > 0 else False

            # update memory
            self.memory.add(state=self.state, done=done,
                            reward=reward, next_state=state)

            self.state = state
            episode_n_itrs += 1

            if len(self.memory) > self.batch_size:

                minibatch = self.memory.sample(self.batch_size)

                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

                Q1 = self.net1(state1_batch)

                with torch.no_grad():
                    Q2 = self.net2(state2_batch)  # B

                Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

                loss = self.loss_fn(X, Y.detach())

                print("{0} At episode {1} loss={2}".format(INFO,
                                                           self.current_episode_index,
                                                           loss.item()))

                self.optimizer.zero_grad()
                loss.backward()
                self.losses.append(loss.item())
                self.optimizer.step()

                if self._update_net_counter  % self.synchronize_frequency == 0:
                    self.net2.load_state_dict(self.net1.state_dict())

            # the episode is finished
            if reward != -1 or mov > self.max_moves:
                #status = 0
                #mov = 0
                break
        self.rewards.append(episode_total_reward)
        self.iterations.append(episode_n_itrs)
