import torch
import numpy as np
from matplotlib import pylab as plt
from src.worlds.grid_world import Gridworld, GridWorldActionType, GridworldInitMode
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecreaseOption
from src.utils import INFO

# constants to use
GAMMA = 0.9
EPS = 1.0
EPOCHS = 1000


def epsilon_decay(eps: float, episode_idx: int) -> float:
    """
    Decay epsilon according to a user defined rul
    :param eps:
    :param episode_index:
    :return:
    """
    if eps > 0.1:
        eps -= (1 / EPOCHS)
    return eps


def play_and_test(network, mode: GridworldInitMode = GridworldInitMode.STATIC, display: bool =True):

    # dummy counter for moves
    i = 0

    test_game = Gridworld(size=4, mode=mode)

    time_step = test_game.reset()
    state = torch.from_numpy(time_step.observation).float()

    if display:
        print("Initial State:")
        test_game.render()

    # flag to track the status of the game
    status = 1

    while status == 1:
        qval = network(state)
        qval_ = qval.data.numpy()

        action_ = np.argmax(qval_)
        action = test_game.get_action(action_)
        if display:
            print('Move #: %s; Taking action: %s' % (i, action.name))

        time_step = test_game.step(action)
        state = torch.from_numpy(time_step.observation).float()

        if display:
            test_game.render()

        reward = time_step.reward

        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1
        if i > 15:
            if display:
                print("Game lost; too many moves.")
            break
    win = True if status == 2 else False
    return win


if __name__ == '__main__':

    env = Gridworld(size=4, mode=GridworldInitMode.STATIC)
    env.render()

    l1 = 64
    l2 = 150
    l3 = 100
    l4 = 4

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4)
    )

    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    policy = EpsilonGreedyPolicy(n_actions=env.n_actions, eps=EPS,
                                 decay_op=EpsilonDecreaseOption.USER_DEFINED,
                                 user_defined_decrease_method=epsilon_decay,
                                 min_eps=0.00001)

    losses = []
    for i in range(EPOCHS):

        print("{0} At episode {1}".format(INFO, i))

        # at the beginning of every episode reset the
        # environment
        time_step = env.reset()
        state1 = torch.from_numpy(time_step.observation).float()
        done = time_step.last()

        itr_counter = 1
        while not done:

            qval = model(state1)
            qval_ = qval.data.numpy()

            # choose an action according to epsilon greedy policy
            aidx = policy.choose_action_index(values=qval_)
            time_step = env.step(action=env.get_action(aidx))

            state2 = torch.from_numpy(time_step.observation).float()
            reward = time_step.reward

            with torch.no_grad():
                newQ = model(state2.reshape(1, 64))
                
            maxQ = torch.max(newQ)
            if reward == -1:
                Y = reward + (GAMMA * maxQ)
            else:
                Y = reward

            Y = torch.Tensor([Y]).detach()
            X = qval.squeeze()[aidx]  # O
            loss = loss_fn(X, Y)  # P
            print("{0} At iteration {1} loss is {2}".format(INFO, itr_counter, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            state1 = state2
            itr_counter += 1
            done = time_step.last()

            if done:
                print("{0} Finished episode {1} ".format(INFO, i))

        # decay epsilon
        policy.actions_after_episode(episode_idx=i)

    plt.figure(figsize=(10, 7))
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.show()

    play_and_test(network=model)

