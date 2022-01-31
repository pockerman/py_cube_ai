import torch
import numpy as np
from matplotlib import pylab as plt
from src.worlds.grid_world import Gridworld, GridWorldActionType, InitMode
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecreaseOption


if __name__ == '__main__':

    env = Gridworld(size=4, mode=InitMode.STATIC)
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

    GAMMA = 0.9
    EPS = 1.0

    policy = EpsilonGreedyPolicy(n_actions=env.n_actions, eps=EPS,
                                 decay_op=EpsilonDecreaseOption.NONE)

    epochs = 1000
    losses = []
    for i in range(epochs):

        # at the beginning of every episode reset the
        # environment
        time_step = env.reset()  #= Gridworld(size=4, mode='static')
        state1 = torch.from_numpy(time_step.observation).float()
        status = 1

        while status == 1:

            qval = model(state1)
            qval_ = qval.data.numpy()

            # choose an action according to epsilon greedy policy
            aidx = policy.choose_action_index(values=qval_)
            """
            if (random.random() < epsilon):
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
            """

            #action = action_set[action_]
            #game.makeMove(action)
            time_step = env.step(action=env.get_action(aidx))

            #state2_ = env.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            state2 = torch.from_numpy(time_step.observation).float()

            reward = time_step.reward #game.reward()

            with torch.no_grad():
                newQ = model(state2.reshape(1, 64))
                
            maxQ = torch.max(newQ)
            if reward == -1:
                Y = reward + (GAMMA * maxQ)
            else:
                Y = reward
            Y = torch.Tensor([Y]).detach()
            X = qval.squeeze()[aidx] #[action_]

            loss = loss_fn(X, Y)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            state1 = state2
            if reward != -1:
                status = 0
        if epsilon > 0.1:
            epsilon -= (1 / epochs)

