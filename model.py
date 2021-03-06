from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init


class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space, terminal_prediction, reward_prediction):
        super(A3Clstm, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTMCell(1024, 128) # it was 1024 x 512

        num_outputs = action_space.n

        self.critic_linear = nn.Linear(128, 1) # it was 512 x 1
        self.actor_linear = nn.Linear(128, num_outputs)

        self.terminal_aux_head = None
        if terminal_prediction: # this comes with the arg parser
            self.terminal_aux_head = nn.Linear(128, 1) # output a single prediction
        # TODO later reward prediction will be added here as well ...

        self.reward_aux_head = None
        if reward_prediction:
            self.reward_aux_head = nn.Linear(128,1) # output a single estimate of reward prediction


        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        # new added parts for auxiliary tasks within the network
        if terminal_prediction:
            self.terminal_aux_head.weight.data = norm_col_init(self.terminal_aux_head.weight.data, 1.0)
            self.terminal_aux_head.bias.data.fill_(0)

        if reward_prediction:
            self.reward_aux_head.weight.data = norm_col_init(self.reward_aux_head.weight.data, 1.0)
            self.reward_aux_head.bias.data.fill_(0)


        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        self.train()
        inputs, (hx, cx) = inputs
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        if self.terminal_aux_head is None:
            terminal_prediction = None
        else:
            terminal_prediction = torch.sigmoid(self.terminal_aux_head(x))

        if self.reward_aux_head is None:
            reward_prediction = None
        else:
            reward_prediction = self.reward_aux_head(x)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx), terminal_prediction, reward_prediction # last two outputs are auxiliary tasks