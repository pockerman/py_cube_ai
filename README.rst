## rl-python

Implementation of reinforcement learning algorithms. Algorithms have been refactored/reimplemented
from various resources such as:

- <a href="https://github.com/udacity/deep-reinforcement-learning">Udacity DRL repository</a>
- <a href="https://livevideo.manning.com/module/56_8_7/reinforcement-learning-in-motion/">Reinforcement learning in motion</a>
- <a href="#">Deep Reinforcement Learning in Action</a>

## Dependencies

- <a href="#">OpenAI Gym</a>
- <a href="#">PyTorch</a>
- <a href="#">NumPy</a>
- <a href="https://cyberbotics.com/#cyberbotics">Webots</a>

## Installation
TODO 
### Installing webots and getting started
Checkout the instructions <a href="webots_howto.md">here</a> how to install and get started with Webots.

## Documentation
TODO

## Examples

### Reinforcement learning basic algorithms

- <a href="src/examples/dummy/dummy_gym_agent_example.py">Dummy agent on ```MountainCar-v0```</a>
- <a href="src/examples/armed_bandit_epsilon_greedy.py">Armed-bandit with epsilon greedy policy</a>
- <a href="#">Armed-bandit with softmax policy</a>
- <a href="src/examples/pytorch_examples/advertisement_placement.py">Contextual bandits</a>

#### Dynamic programming

- <a href="src/examples/dp/iterative_policy_evaluation_frozen_lake.py">Iterative policy evaluation on ```FrozenLake-v0```</a>
- <a href="src/examples/dp/policy_improvement_frozen_lake.py">Policy improvement on ```FrozenLake-v0```</a>
- <a href="src/examples/dp/policy_iteration_frozen_lake.py">Policy iteration on ```FrozenLake-v0```</a>
- <a href="src/examples/dp/value_iteration_frozen_lake.py">Value iteration on ```FrozenLake-v0```</a>

#### Monte Carlo

- <a href="src/examples/mc/mc_prediction_black_jack.py">Monte Carlo prediction on ```Blackjack-v0```</a>
- <a href="src/examples/mc/mountain_car_approximate_monte_carlo.py">Approximate Monte Carlo on ```MountainCar-v0```</a>
- <a href="src/examples/mc/mc_tree_search_taxi_v3.py.py">Monte Carlo tree search ```Taxi-v3```</a>

#### Temporal differencing

- <a href="src/examples/td/td_zero_cart_pole_v0.py">TD(0) on ```CartPole-v0```</a> 
- <a href="src/examples/td/cliff_walking_q_learning.py">SARSA on ```Cliffwalking-v0```</a> 
- <a href="src/examples/td/sarsa_cart_pole_v0.py">SARSA on ```CartPole-v0```</a> 
- <a href="src/examples/td/cliff_walking_q_learning.py">Q-learning on ```Cliffwalking-v0``` </a> 
- <a href="src/examples/td/q_learning_cart_pole_v0.py">Q-learning on ```CartPole-v0``` </a> 
- <a href="#">Expected SARSA  </a> (TODO)
- <a href="#">SARSA lambda  </a> (TODO)
- <a href="src/examples/td/td_zero_semi_gradient_mountain_car.py">TD(0) semi-gradient on ```MountainCar-v0```</a>
- <a href="src/examples/td/sarsa_semi_gradient_mountain_car_v0.py">SARSA semi-gradient on ```MountainCar-v0```</a>
- <a href="src/examples/td/q_learning_moutain_car_v0.py">Q-learning on ```MountainCar-v0```</a>
- <a href="src/examples/td/double_q_learning_cart_pole_v0.py">Double Q-learning on ```CartPole-v0```</a>

#### DQN

- <a href="src/examples/dqn/dqn_grid_world.py">Vanilla DQN on ```Gridworld```</a>
- <a href="src/examples/dqn/dqn_with_experience_replay_on_grid_world.py">DQN with experience replay on ```Gridworld```</a>
- <a href="#">DQN with target network on ```Gridworld```</a>
- <a href="src/examples/dqn/dqn_lunar_lander.py">Vanilla DQN on ```CartPole-v0```</a>
- <a href="src/examples/dqn/dqn_lunar_lander.py">Vanilla DQN on ```LunarLander-v2```</a>

#### Approximate methods

- <a href="#">Simple gradient descent solver</a>
- <a href="src/examples/pg/reinforce_cart_pole.py">REINFORCE on ```CartPole-v0```</a>
- <a href="src/examples/ac/a2c_cart_pole_v1.py">A2C on ```CartPole-v1```</a>

## Robotics simulations

- <a href="src/examples/webots/epuck_robot/controllers/epuck_q_learn_simple_controller.py">Q-learning with epuck robot.</a>

## References

- ```Deep Reinforcement Learning in Action```




 


