import torch
import torch.nn as nn
import torch.nn.functional as F

## centralized critic
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.input_dim = args.nagents * (args.hid_size + sum(args.num_actions))
        self.hidden_dim = args.hid_size
    
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, 1)

    def forward(self, hidden_state, actions):
        critic_in = self._build_input(hidden_state, actions)
        h1 = F.relu(self.linear1(critic_in))
        h2 = F.relu(self.linear2(h1))
        value = self.linear3(h2)
        
        return value
    
    def _build_input(self, hidden_state, actions):
        batch_size = hidden_state.size()[0]
        n = self.args.nagents
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions
        # use one-hot actions as the input of the critic
        # element of actions_onehot size: batch_size * n * num_actions[i]
        actions_onehot = [torch.Tensor(batch_size, n, num_actions[i]) for i in range(dim_actions)]
        for i in range(dim_actions):
            actions_onehot[i].zero_()
            actions_onehot[i].scatter_(2, actions[:,:,i].unsqueeze(dim=-1).long(), 1)
            # size: batch_size * (n*num_actions[i])
            actions_onehot[i] = actions_onehot[i].view(batch_size, n*num_actions[i])
        # size: batch_size * (n*hid_size + n*sum(num_actions))
        critic_in = torch.cat([hidden_state, torch.cat(actions_onehot, dim=1)], dim=1)
        
        return critic_in


