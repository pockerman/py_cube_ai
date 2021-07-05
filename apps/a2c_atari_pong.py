"""
Play Atari pong with A2C
"""
import gym
import numpy as np
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from tensorboardX import SummaryWriter
from networks.a2c_atari_nn import AtariA2C_NN
from agents.policy_agent import PolicyAgent
from utils.experience import ExperienceSourceFirstLast
from utils.reward_tracker import RewardTracker
from utils.tb_mean_tracker import TBMeanTracker
from utils.utilities import wrap_dqn


def unpack_batch(batch, net, gamma, reward_steps, device='cpu'):
    """
    Convert batch into training tensors
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        last_vals_np *= gamma ** reward_steps
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)

    return states_v, actions_t, ref_vals_v


if __name__ == "__main__":

    NUM_ENVS = 50
    GAMMA = 0.99
    REWARD_STEPS = 4
    LEARNING_RATE = 0.001
    ENV_NAME = "PongNoFrameskip-v4"
    ENTROPY_BETA = 0.01
    BATCH_SIZE = 128
    CLIP_GRAD = 0.1

    # Sets the number of threads used for intraop parallelism on CPU
    torch.set_num_threads(5)
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    #parser.add_argument("-n", "--name", required=True, help="Name of the run")
    #args = parser.parse_args()
    #device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device("cpu")

    make_env = lambda: wrap_dqn(gym.make(ENV_NAME))
    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-atari-pong-a2c_" + "v1")

    net = AtariA2C_NN(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)

    # the policy
    agent = PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []

    with RewardTracker(writer, stop_reward=18) as tracker:
        with TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                # if the batch is not full simply
                # continue
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, gamma=GAMMA,
                                                               reward_steps=REWARD_STEPS, device=device)
                # clear the batch so that
                # we accept new states
                batch.clear()

                optimizer.zero_grad()

                # pass the states from the model
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                # apply entropy and value gradients
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                # get full loss
                loss_v += loss_policy_v

                tb_tracker.track("advantage",       adv_v, step_idx)
                tb_tracker.track("values",          value_v, step_idx)
                tb_tracker.track("batch_rewards",   vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
                tb_tracker.track("loss_value",      loss_value_v, step_idx)
                tb_tracker.track("loss_total",      loss_v, step_idx)
                tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var",        np.var(grads), step_idx)