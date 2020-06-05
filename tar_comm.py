import torch
import torch.nn.functional as F
from torch import nn

from models import MLP
from action_utils import select_action, translate_action
import numpy as np
# import itertools
# from utils import *

class TarCommNetMLP(nn.Module):
    """
    MLP based CommNet. Uses communication vector to communicate info
    between agents
    """
    def __init__(self, args, num_inputs):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MLP {object} -- Self
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
        """

        super(TarCommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent
        self.qk_hid_size = args.qk_hid_size
        self.value_hid_size = args.value_hid_size

        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o)
                                        for o in args.naction_heads])
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)


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

        # Our main function for converting current hidden state to next state
        # self.f = nn.Linear(args.hid_size, args.hid_size)
        if args.share_weights:
            self.C_module = nn.Linear(args.value_hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.value_hid_size, args.hid_size)
                                            for _ in range(self.comm_passes)])

        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        # print(self.C)
        # self.C.weight.data.zero_()
        # Init weights for linear layers
        # self.apply(self.init_weights)

        self.value_input_dim = args.hid_size + args.nagents * sum(args.num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(self.value_input_dim, self.hid_size),
            nn.ReLU(),
            nn.Linear(self.hid_size, self.hid_size),
            nn.ReLU(),
            nn.Linear(self.hid_size, 1))

        # soft attention layers 
        self.wq = nn.Linear(args.hid_size, args.qk_hid_size)
        self.wk = nn.Linear(args.hid_size, args.qk_hid_size)
        self.wv = nn.Linear(args.hid_size, args.value_hid_size)


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

        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            agent_mask *= comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)

        for i in range(self.comm_passes):

            # calculate for soft attention
            q = self.wq(hidden_state)
            k = self.wk(hidden_state)
            v = self.wv(hidden_state)
            soft_attn = F.softmax(torch.matmul(q, k.transpose(0, 1)) / np.sqrt(self.qk_hid_size), dim=1)
            soft_attn = soft_attn.unsqueeze(-1).expand(n, n, self.value_hid_size)
            soft_attn = soft_attn.unsqueeze(0).expand(batch_size, n, n, self.value_hid_size)
            
            # Choose current or prev depending on recurrent
            comm = v.view(batch_size, n, self.value_hid_size) if self.args.recurrent else v
            # Get the next communication vector based on next hidden state
            comm = comm.unsqueeze(-2).expand(-1, n, n, self.value_hid_size)
            
            # Mask comm_in
            # Mask communcation from dead agents
            comm = comm * agent_mask
            # Mask communication to dead agents
            comm = comm * agent_mask_transpose

            # Combine all of C_j for an ith agent which essentially are h_j
            comm_sum = (comm * soft_attn.transpose(1, 2)).sum(dim=1)
            c = self.C_modules[i](comm_sum)

            if self.args.recurrent:
                # skip connection - combine comm. matrix and encoded input for all agents
                inp = x + c

                inp = inp.view(batch_size * n, self.hid_size)

                output = self.f_module(inp, (hidden_state, cell_state))

                hidden_state = output[0]
                cell_state = output[1]

            else: # MLP|RNN
                # Get next hidden state from f node
                # and Add skip connection from start and sum them
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)
        
        h = hidden_state.view(batch_size, n, self.hid_size)

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action_out = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            action_out = [F.log_softmax(head(h), dim=-1) for head in self.heads]

        action = select_action(self.args, action_out)  

        actions = [x.squeeze().data.numpy() for x in action]
        actions = torch.Tensor(actions)
        actions = actions.transpose(0, 1).view(self.args.nagents, self.args.dim_actions)
        
        # here only consider discrete actions
        critic_in = self.build_q_input(hidden_state, actions)
        value_head = self.value_head(critic_in)
        
#         baseline = self.compute_counterfactual_baseline(hidden_state, actions, action_out)    
            
        if self.args.recurrent:
            # what is the difference between using clone and not using it?
            return action_out, action, value_head, (hidden_state.clone(), cell_state.clone())
        else:
            return action_out, action, value_head, hidden_state.clone()

        
    def build_q_input(self, hidden_state, actions):
        n = self.args.nagents
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions   

        actions = actions.unsqueeze(0).expand(n, n, dim_actions)
        actions_onehot = [torch.Tensor(n, n, num_actions[i]) for i in range(dim_actions)]    
        for i in range(dim_actions):
            actions_onehot[i].zero_()
            actions_onehot[i].scatter_(2, actions[:,:,i].unsqueeze(dim=-1).long(), 1)
            # size: n * (n*num_actions[i])
            actions_onehot[i] = actions_onehot[i].view(n, n*num_actions[i])
        # size: n * (hid_size + n*sum(num_actions))
        critic_in = torch.cat([hidden_state, torch.cat(actions_onehot, dim=1)], dim=1)
        
        return critic_in
    
#     def compute_counterfactual_baseline(self, hidden_state, actions, action_out):
#         n = self.args.nagents
#         num_actions = self.args.num_actions
#         dim_actions = self.args.dim_actions 
        
#         action_index_list = [[j for j in range(num_actions[i])]  for i in range(dim_actions)]
#         # a list of all the possible combinations of differnet action heads 
#         action_index_combo = list(itertools.product(*action_index_list))
        
#         baseline = []
#         for agent_idx in range(n):
#             # element size: 1 * num_actions[i]
#             log_p_agent = [action_out[i][:, agent_idx, :].view(-1, num_actions[i]) for i in range(dim_actions)]
#             agent_baseline = []
#             for action_combo in action_index_combo:
#                 # size: 1 * dim_actions
#                 action_combo = torch.Tensor(action_combo).view(-1, dim_actions)
#                 action_marginalized = actions.clone()
#                 action_marginalized[agent_idx, :] = action_combo
#                 # size: 1 * (hid_size + n*sum(num_actions))
#                 q_input = self.build_q_input(hidden_state, actions)[agent_idx].unsqueeze(0)
#                 # size: 1 * 1
#                 agent_q_marginalized = self.value_head(q_input).detach()
#                 # size: 1 * dim_actions
#                 agent_actions = action_marginalized[agent_idx, :].view(-1, dim_actions)
#                 # size: 1 * 1
#                 # changing log_prob_agent will not change log_p_agent or action_out
#                 log_prob_agent = multinomials_log_density(agent_actions, log_p_agent)
#                 agent_baseline.append(agent_q_marginalized * torch.exp(log_prob_agent.clone().detach()))              
#             # size: 1 * 1
#             agent_baseline = torch.cat(agent_baseline, dim=1).sum(dim=1).unsqueeze(dim=-1)
#             baseline.append(agent_baseline)
#         # size: n * 1
#         baseline = torch.cat(baseline, dim=0)
        
#         return baseline

        
    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1).clone()

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
        
        
    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

            
    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

