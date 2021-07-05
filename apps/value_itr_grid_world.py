import numpy as np
from algorithms.algorithm_base import TrainMode
from environments.gridworld import GridworldEnv
from algorithms.value_iteration import ValueIteration

if __name__ == '__main__':

    environment = GridworldEnv()
    print("Number of states: {0}".format(environment.nS))
    print("Number of actions: {0}".format(environment.nA))
    print("Environment shape: {0}".format(environment.shape))
    value_iteration = ValueIteration(env=environment, tolerance=1.0e-4, gamma=1.0,
                                     update_values_on_start_itrs=False, train_mode=TrainMode.DEFAULT)

    value_iteration.train()
    policy = value_iteration.get_policy()
    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), environment.shape))
    print("")

    print("Value Function:")
    v = value_iteration.values
    values = [v[x] for x in v]
    print(values)
    print("")

    values = [v[x] for x in v]
    v = np.array(values)
    print("Reshaped Grid Value Function:")
    print(v.reshape(environment.shape))
    print("")