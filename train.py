from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
import torch.nn.functional as F
from environment import atari_env
from utils import ensure_shared_grads, np, v_wrap
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
from torchviz import make_dot



def train(rank, args, shared_model, optimizer, env_conf):

    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)

    tp_weight = args.tp

    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0], player.env.action_space, args.terminal_prediction, args.reward_prediction)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()

    # Below is where the cores are running episodes continously ...
    average_ep_length = 0

    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, 128).cuda())
                    player.hx = Variable(torch.zeros(1, 128).cuda())
            else:
                player.cx = Variable(torch.zeros(1, 128))
                player.hx = Variable(torch.zeros(1, 128))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)

        for step in range(args.num_steps):
            player.eps_len += 1
            player.action_train()
            if player.done:
                break

        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _, _, _ = player.model((Variable(player.state.unsqueeze(0)), (player.hx, player.cx)))
            R = value.data


        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        reward_pred_loss = 0
        terminal_loss = 0



        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R) # TODO why this is here?

        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * player.values[i + 1].data - player.values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - player.log_probs[i] * Variable(gae) - 0.01 * player.entropies[i]

            if args.reward_prediction:
                reward_pred_loss = reward_pred_loss + (player.reward_predictions[i] - player.rewards[i]).pow(2)

        if args.terminal_prediction: # new way of using emprical episode length as a proxy for current length.
            if player.average_episode_length is None:
                end_predict_labels = np.arange(player.eps_len-len(player.terminal_predictions), player.eps_len) / player.eps_len # heuristic
            else:
                end_predict_labels = np.arange(player.eps_len-len(player.terminal_predictions), player.eps_len) / player.average_episode_length

            for i in range(len(player.terminal_predictions)):
                terminal_loss = terminal_loss + (player.terminal_predictions[i] - end_predict_labels[i]).pow(2)

            terminal_loss = terminal_loss / len(player.terminal_predictions)


        player.model.zero_grad()
        #print(f"policy loss {policy_loss} and value loss {value_loss} and terminal loss {terminal_loss} and reward pred loss {reward_pred_loss}")

        total_loss = policy_loss + 0.5 * value_loss + tp_weight*terminal_loss + 0.5*reward_pred_loss


        total_loss.backward() # will free memory ...

        # Visualize Computation Graph
        #graph = make_dot(total_loss)
        #from graphviz import Source
        #Source.view(graph)

        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()

        if player.done:
            if player.average_episode_length is None: # initial one
                player.average_episode_length = player.eps_len
            else:
                player.average_episode_length = int(0.99 * player.average_episode_length + 0.01 * player.eps_len)
            #print(player.average_episode_length, 'current one is ', player.eps_len)
            player.eps_len = 0 # reset here
