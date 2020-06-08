"""
Revised from CommNetMLP
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import itertools
from models import MLP
from action_utils import select_action, translate_action
from gnn_layers import GraphAttention

class GCommNetMLP(nn.Module):
    def __init__(self, args, num_inputs):
        super(GCommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent
        
        dropout = 0
        negative_slope = 0.2
        nheads = 4
        
        if args.gnn_type == 'gat':
            self.gconv1 = [GraphAttention(args.hid_size, int(args.hid_size/4), dropout=dropout, negative_slope=negative_slope, concat=True) for _ in range(nheads)]
            for i, attention in enumerate(self.gconv1):
                self.add_module('attention_{}'.format(i), attention)
            self.gconv2 = GraphAttention(args.hid_size, args.hid_size, dropout=dropout, negative_slope=negative_slope, concat=False)
            
            
#         if args.gnn_type == 'gcn':
#             self.gconv1 = GCNConv(args.hid_size, args.hid_size, cached=False, normalize=False)
#             self.gconv2 = GCNConv(args.hid_size, args.hid_size, cached=False, normalize=False)
#             self.gconv3 = GCNConv(args.hid_size, args.hid_size, cached=False, normalize=False)
        
        self.hard_attn1 = nn.Sequential(
            nn.Linear(self.hid_size*2, int(self.hid_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.hid_size/2), int(self.hid_size/8)),
            nn.ReLU(),
            nn.Linear(int(self.hid_size/8), 2))

        self.hard_attn2 = nn.Sequential(
            nn.Linear(self.hid_size*2, int(self.hid_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.hid_size/2), int(self.hid_size/8)),
            nn.ReLU(),
            nn.Linear(int(self.hid_size/8), 2))
        
        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            # support multi action
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o)
                                        for o in args.naction_heads])
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)


        # Since linear layers in PyTorch now accept * as any number of dimensions
        # between last and first dim, num_agents dimension will be covered.
        # The network below is function r in the paper for encoding
        # initial environment stage
        self.encoder = nn.Linear(num_inputs, args.hid_size)

        # if self.args.env_name == 'starcraft':
        #     self.state_encoder = nn.Linear(num_inputs, num_inputs)
        #     self.encoder = nn.Linear(num_inputs * 2, args.hid_size)
        if args.recurrent:
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        if args.recurrent:
            self.init_hidden(args.batch_size)
            self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)

        else:
            if args.share_weights:
                self.f_module = nn.Linear(args.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                                for _ in range(self.comm_passes)])
        # else:
            # raise RuntimeError("Unsupported RNN type.")

        # main function for converting current hidden state to next state
        # self.f = nn.Linear(args.hid_size, args.hid_size)
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                            for _ in range(self.comm_passes)])
        # self.C = nn.Linear(args.hid_size, args.hid_size)

        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()
        
        self.value_head = nn.Linear(self.hid_size, 1)


    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()

        return num_agents_alive, agent_mask

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, extras = x
            x = self.encoder(x)

            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
        else:
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state


    def forward(self, x, info={}):
        # TODO: Update dimensions
        """Forward function for CommNet class, expects state, previous hidden
        and communication tensor.
        B: Batch Size: Normally 1 in case of episode
        N: number of agents

        Arguments:
            x {tensor} -- State of the agents (N x num_inputs)
            prev_hidden_state {tensor} -- Previous hidden state for the networks in
            case of multiple passes (1 x N x hid_size)
            comm_in {tensor} -- Communication tensor for the network. (1 x N x N x hid_size)

        Returns:
            tuple -- Contains
                next_hidden {tensor}: Next hidden state for network
                comm_out {tensor}: Next communication tensor
                action_data: Data needed for taking next action (Discrete values in
                case of discrete, mean and std in case of continuous)
                v: value head
        """

        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        for i in range(self.comm_passes):
            # Choose current or prev depending on recurrent
            comm = hidden_state

            # Get the next communication vector based on next hidden state

            # Mask communcation from dead agents
            comm = comm * agent_mask
            
            # should make sure that adj will be in backprop
            adj = self.get_adj_matrix(self.hard_attn1, hidden_state, agent_mask, self.args.directed, self.args.self_loop)
            
#             adj = torch.ones(n, n)

            if self.args.gnn_type == 'gat':
                comm = torch.cat([att(comm, adj) for att in self.gconv1], dim=1)
                comm = self.gconv2(comm, adj)
        
            # Mask communication to dead agents
            comm = comm * agent_mask
            c = self.C_modules[i](comm)

            if self.args.recurrent:
                # skip connection - combine comm. matrix and encoded input for all agents
                x = x.squeeze()
                inp = x + c
                
                output = self.f_module(inp, (hidden_state, cell_state))

                hidden_state = output[0]
                cell_state = output[1]

            else: # MLP|RNN
                # Get next hidden state from f node
                # and Add skip connection from start and sum them
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)

        # v = torch.stack([self.value_head(hidden_state[:, i, :]) for i in range(n)])
        # v = v.view(hidden_state.size(0), n, -1)
        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size, n, self.hid_size)

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            action = [F.log_softmax(head(h), dim=-1) for head in self.heads]

        if self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone())
        else:
            return action, value_head

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))
    
    
    def get_adj_matrix(self, hard_attn_model, hidden_state, agent_mask, directed=True, self_loop=True):
        # hidden_state size: n * hid_size
        n = self.args.nagents
        # hard_attn_input size: n * n * (2*hid_size)
        hard_attn_input = torch.cat([hidden_state.repeat(1, n).view(n * n, -1), hidden_state.repeat(n, 1)], dim=1).view(n, -1, 2 * self.hid_size)
        # hard_attn_output size: n * n * 2
        if directed:
            hard_attn_output = F.gumbel_softmax(hard_attn_model(hard_attn_input), hard=True)
        else:
            hard_attn_output = F.gumbel_softmax(0.5*hard_attn_model(hard_attn_input)+0.5*hard_attn_model(hard_attn_input.permute(1,0,2)), hard=True)
        # hard_attn_output size: n * n * 1
        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
        # agent_mask and its transpose size: n * n
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        # adj size: n * n
        adj = hard_attn_output.squeeze() * agent_mask * agent_mask_transpose
        
        if not self_loop:
            self_loop_mask = torch.ones(n, n) - torch.eye(n, n)
            adj = adj * self_loop_mask 
            
        return adj
        
            
        

