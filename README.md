## rl-python

Implementation of reinforcement learning algorithms. Algorithms have been refactored/reimplemented
from various resources such as:

- <a href="https://github.com/udacity/deep-reinforcement-learning">Udacity DRL repository</a>


### Acknowledgements

The inital version of the simulator is taken from https://github.com/nmccrea/sobot-rimulator (thanks a lot). 

## Dependencies

- OpenAI Gym
- PyTorch
- NumPy


## Examples

### Dynamic programming

- <a href="src/apps/dp/iterative_policy_evaluation_frozen_lake.py">Iterative policy evaluation on FrozenLake-v0</a>
- <a href="src/apps/dp/policy_improvement_frozen_lake.py">Policy improvement on FrozenLake-v0</a>
- <a href="src/apps/dp/policy_iteration_frozen_lake.py">Policy iteration on FrozenLake-v0</a>
- <a href="src/apps/dp/value_iteration_frozen_lake.py">Value iteration on FrozenLake-v0</a>

### Monte Carlo

- <a href="src/apps/mc_prediction_black_jack.py">Monte Carlo prediction on ```Blackjack-v0```</a>
- <a href="src/apps/mc_prediction_black_jack.py">Approximate Monte Carlo on ```MountainCar-v0```</a>

### Temporal differencing

- <a href="src/apps/td/cliff_walking_q_learning.py">SARSA on ```Cliffwalking-v0```</a> 
- <a href="src/apps/td/cliff_walking_q_learning.py">Q-learning on ```Cliffwalking-v0``` </a> 
- <a href="#">Expected SARSA  </a> 


### DQN

- <a href="src/apps/dqn/dqn_lunar_lander.py">Vanilla DQN on CartPole-v0</a>
- <a href="src/apps/dqn/dqn_lunar_lander.py">Vanilla DQN on LunarLander-v2</a>


### Policy gradient methods

- <a href="src/apps/pg/reinforce_cart_pole.py">REINFORCE on CartPole-v0</a>


## Images

![Mountain car](images/mountain_car.gif)
**Approximate Monte Carlo on Mountain Car**

![State value function](images/state_value_function_frozen_lake.png)
**Iterative policy evaluation on FrozenLake**

![State value function](images/q_learning_state_value.png)
**Q-learning on Cliffwalking**



 


