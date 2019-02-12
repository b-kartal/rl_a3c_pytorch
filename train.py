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
    player.eps_len += 2 # TODO why by two? is this frame-skipping ?

    # Below is where the cores are running episodes continously ...
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

        # New part for the Reward Prediction Auxiliary Task
        #if args.reward_prediction:
        #    actual_rewards = v_wrap(np.asarray(player.rewards), np.float32).unsqueeze(1)
        #    predicted_rewards = Variable(torch.tensor(player.reward_predictions).unsqueeze(1), requires_grad=True)

        #    if gpu_id >= 0:
        #        with torch.cuda.device(gpu_id):
        #            predicted_rewards = predicted_rewards.cuda()
        #            actual_rewards = actual_rewards.cuda()

        #    reward_pred_loss = loss_fn(predicted_rewards, actual_rewards)

        # New Part for the Terminal Prediction Auxiliary Task
        if args.terminal_prediction and player.done: # this is the only time we can get labels
            # create the terminal self-supervised labels
            end_predict_labels = np.arange(1, (len(player.terminal_predictions) + 1)) / (len(player.terminal_predictions) + 1)

            for i in range(len(player.terminal_predictions)):
                 terminal_loss = terminal_loss + ( player.terminal_predictions[i] - end_predict_labels[i] ).pow(2)

            terminal_loss = terminal_loss / len(player.terminal_predictions)

            #print(terminal_loss)

            player.terminal_predictions = []  # Note that this is not done in clear_actions method as terminal labels are received at the end of episode

        player.model.zero_grad()
        #print(f"policy loss {policy_loss} and value loss {value_loss} and terminal loss {terminal_loss} and reward pred loss {reward_pred_loss}")

        total_loss = policy_loss + 0.5 * value_loss + 0.5*terminal_loss + 0.5*reward_pred_loss

        if args.terminal_prediction and player.done is False:
            total_loss.backward(retain_graph=True)
        else:
            total_loss.backward() # will free memory ...

        # Visualize Computation Graph
        #graph = make_dot(total_loss)
        #from graphviz import Source
        #Source.view(graph)

        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()