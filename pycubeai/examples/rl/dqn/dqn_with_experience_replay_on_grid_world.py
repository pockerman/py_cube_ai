import torch
import numpy as np
from matplotlib import pylab as plt
from pycubeai.worlds.grid_world import Gridworld, GridworldInitMode
from pycubeai.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecayOption
from pycubeai.utils.replay_buffer import ExperienceTuple, ReplayBuffer
from pycubeai.utils import INFO

# constants to use
GAMMA = 0.9
EPS = 0.3
EPOCHS = 5000
BATCH_SIZE = 200
MEM_SIZE = 1000
MAX_MOVES = 50
LEARNING_RATE = 1e-3


def play_and_test(network, mode: GridworldInitMode = GridworldInitMode.STATIC, display: bool = True):

    # dummy counter for moves
    moves_played = 0

    test_game = Gridworld(size=4, mode=mode, noise_factor=100)

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
        moves_played += 1
        if moves_played > 15:
            if display:
                print("Game lost; too many moves.")
            break
    win = True if status == 2 else False
    return win


if __name__ == '__main__':

    losses = []
    h = 0

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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    replay = ReplayBuffer(buffer_size=MEM_SIZE)
    game = Gridworld(size=4, mode=GridworldInitMode.RANDOM, noise_factor=100)

    policy = EpsilonGreedyPolicy(n_actions=game.n_actions, eps=EPS,
                                 decay_op=EpsilonDecayOption.NONE,
                                 user_defined_decrease_method=None,
                                 min_eps=0.00001)

    for i in range(EPOCHS):

        time_step = game.reset()
        state1 = torch.from_numpy(time_step.observation).float()
        status = 1
        mov = 0

        while status == 1:

            mov += 1
            qval = model(state1)
            qval_ = qval.data.numpy()

            action_idx = policy.choose_action_index(values=qval_)

            # get the action index and execute
            # the action in the world
            action = game.get_action(action_idx)
            time_step = game.step(action)

            state2 = torch.from_numpy(time_step.observation).float()
            reward = time_step.reward

            done = True if reward > 0 else False

            # experience in the buffer
            replay.add(state=state1, reward=reward, done=done, action=action, next_state=state2)
            state1 = state2

            if len(replay) > BATCH_SIZE:

                # The major difference with experience replay training is
                # that we train with mini-batches of data when our replay list is full.
                # We randomly select a subset of experiences from the replay,
                # and we separate out the individual experience components
                # into state1_batch, reward_batch, action_batch, and state2_batch.
                # For example, state1 _batch is of dimensions batch_size × 64, or 100 × 64 in this case.
                # And reward_batch is just a 100-length vector of integers.
                # We follow the same training formula as we did earlier with fully
                # online training, but now we’re dealing with mini-batches.
                # We use the tensor gather method to subset the Q1 tensor (a 100 × 4 tensor)
                # by the action indices so that we only select the Q values
                # associated with actions that were actually chosen, resulting in a 100-length vector.

                minibatch = replay.sample(batch_size=BATCH_SIZE)

                # concatenate the state
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

                # Notice that the target Q value,
                # Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2,dim=1)[0]),
                # uses done_batch to set the right side to 0 if the game is done.
                # Remember, if the game is over after taking an action, which we call a terminal state,
                # there is no next state to take the maximum Q value on,
                # so the target just becomes the reward, rt+1.
                # The done variable is a Boolean, but we can
                # do arithmetic on it as if it were a 0 or 1 integer,
                # so we just take 1 - done so that if done = True, 1 - done = 0,
                # and it sets the right-side term to 0.

                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model(state2_batch)

                Y = reward_batch + GAMMA * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = \
                    Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()

                print("{0} At episode {1} loss={2}".format(INFO, i, loss.item()))
                losses.append(loss.item())
                optimizer.step()

            if reward != -1 or mov > MAX_MOVES:
                status = 0
                mov = 0
    losses = np.array(losses)

    plt.figure(figsize=(10, 7))
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.show()

    max_games = 1000
    wins = 0
    for i in range(max_games):
        win = play_and_test(network=model, mode=GridworldInitMode.RANDOM, display=False)
        if win:
            wins += 1

    win_perc = float(wins) / float(max_games)
    print("Games played: {0}, # of wins: {1}".format(max_games, wins))
    print("Win percentage: {}".format(win_perc))

