from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
import itertools

Transition = namedtuple('Transition', ('state', 'hidden_state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state', 'reward', 'misc'))


class Trainer(object):
    def __init__(self, args, actor, critic, env):
        self.args = args
        self.actor = actor
        self.critic = critic
        self.env = env
        self.display = False
        self.last_step = False
        
        self.actor_optimizer = optim.RMSprop(actor.parameters(),
            lr = args.actor_lrate, alpha=0.97, eps=1e-6)
        self.critic_optimizer = optim.RMSprop(critic.parameters(),
            lr = args.critic_lrate, alpha=0.97, eps=1e-6)  
        
        self.actor_params = list(actor.parameters())
        self.critic_params = list(critic.parameters())
        self.params = self.actor_params + self.critic_params


    def get_episode(self, epoch):
        episode = []
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        for t in range(self.args.max_steps):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)
            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.actor.init_hidden(batch_size=state.shape[0])

                x = [state, prev_hid]
                action_out, value, prev_hid = self.actor(x, info)
                hidden_state = prev_hid[0].clone().detach()

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value = self.actor(x, info)

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            next_state, reward, done, info = self.env.step(actual)

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()

            trans = Transition(state, hidden_state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat)
    
    def comp_critic_grad(self, batch):
        stat = dict()
        # action space
        num_actions = self.args.num_actions
        # number of action head
        dim_actions = self.args.dim_actions        

        n = self.args.nagents
        batch_size = len(batch.state)
        # size: batch_size * n * hid_size
        hidden_state = torch.stack(batch.hidden_state, dim=0)
        # size: batch_size * (n*hid_size)
        hidden_state = hidden_state.view(batch_size, n*self.args.hid_size)
        # size: batch_size * n
        rewards = torch.Tensor(batch.reward)
        actions = torch.Tensor(batch.action)
        # size: batch_size * n * dim_actions.  have been detached
        actions = actions.transpose(1, 2).view(-1, n, dim_actions) 
        # size: batch_size * n
        episode_masks = torch.Tensor(batch.episode_mask)
        # size: batch_size * n
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        # size: (batch_size*n)
        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)
        
        # initialize?
        # use reversed?
            
        # size: batch_size * 1
        # Q(macro_hidden_state, (macro_action_head1, macro_action_head2, ...))
        # the input size is fixed even if there can be dead agents
        critic_out = self.critic(hidden_state, actions)
        # expand size of critic_out:
        critic_out = critic_out.expand(batch_size, n)

        
        # mask the dead agents (in fact the agent is dead after taking action? so should mask next step?)
        # size: batch_size * n
        alive_masks = alive_masks.view(batch_size, n)
        # pad at the start
        if self.args.env_name == "traffic_junction":
            alive_masks = torch.cat([torch.zeros(1, n), alive_masks[:-1, :]], dim=0)
            
            
        coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
        prev_coop_return = 0
        prev_ncoop_return = 0
        
        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])
        
        critic_loss = (returns - critic_out).pow(2) # divided by?
        critic_loss *= alive_masks
        critic_loss = critic_loss.sum()
        stat['critic_loss'] = critic_loss.item()
        
        # backward
        critic_loss.backward() 
        
        return stat
        
    def comp_actor_grad(self, batch):
        stat = dict()
        # action space
        num_actions = self.args.num_actions
        # number of action head
        dim_actions = self.args.dim_actions        

        n = self.args.nagents
        batch_size = len(batch.state)
        # size: batch_size * n * hid_size
        hidden_state = torch.stack(batch.hidden_state, dim=0)
        # size: batch_size * (n*hid_size)
        hidden_state = hidden_state.view(batch_size, n*self.args.hid_size)
        # size: batch_size * n
        rewards = torch.Tensor(batch.reward)
        actions = torch.Tensor(batch.action)
        # size: batch_size * n * dim_actions.  have been detached
        actions = actions.transpose(1, 2).view(-1, n, dim_actions) 
        # size: batch_size * n
        episode_masks = torch.Tensor(batch.episode_mask)
        # size: batch_size * n
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        # size: (batch_size*n)
        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)
        # mask the dead agents (in fact the agent is dead after taking action? so should mask next step?)
        # size: batch_size * n
        alive_masks = alive_masks.view(batch_size, n)
        # pad at the start
        if self.args.env_name == "traffic_junction":
            alive_masks = torch.cat([torch.zeros(1, n), alive_masks[1:, :]], dim=0)
            
        advantages = torch.Tensor(batch_size, n)
        alive_masks = alive_masks.view(-1)
        action_out = list(zip(*batch.action_out))
        # length: dim_actions
        # element size: batch_size * n * num_actions[i]
        action_out = [torch.cat(a, dim=0) for a in action_out]
        
        returns = self.critic(hidden_state, actions).detach()
        returns = returns.expand(batch_size, n)
        
#         ## calculate counterfactual baseline
#         action_index_list = [[j for j in range(num_actions[i])]  for i in range(dim_actions)]
#         # a list of all the possible combinations of differnet action heads 
#         action_index_combo = list(itertools.product(*action_index_list))

#         baseline = []
#         for agent_idx in range(n):
#             # element size: batch_size * num_actions[i]
#             log_p_agent = [action_out[i][:, agent_idx, :].view(-1, num_actions[i]) for i in range(dim_actions)]
#             # print(log_p_agent[1].size())
#             agent_baseline = []
#             # prob = []
#             for action_combo in action_index_combo:
#                 # size: 1 * dim_actions
#                 action_combo = torch.Tensor(action_combo).view(-1, dim_actions)
#                 action_marginalized = actions.clone()
#                 action_marginalized[:, agent_idx, :] = action_combo
#                 # size: batch_size * 1
#                 critic_marginalized = self.critic(hidden_state, action_marginalized).detach()
#                 # size: batch_size * dim_actions
#                 agent_actions = action_marginalized[:, agent_idx, :]
#                 # size: batch_size * 1
#                 log_prob_agent = multinomials_log_density(agent_actions, log_p_agent)
                
#                 agent_baseline.append(critic_marginalized * torch.exp(log_prob_agent))
#                 # prob.append(torch.exp(log_prob_agent))

#             # size: batch_size * 1
#             agent_baseline = torch.cat(agent_baseline, dim=1).sum(dim=1).unsqueeze(dim=-1)
#             # prob = torch.cat(prob, dim=1).sum(dim=1)
#             # print(prob)  # the element here should be one 
#             baseline.append(agent_baseline)

#         # size: batch_size * n
#         baseline = torch.cat(
#             baseline, dim=1)   

        values = torch.cat(batch.value, dim=0)
        values = values.view(batch_size, n)

        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]
        
        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        # element size: (batch_size*n) * num_actions[i]
        log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
        # size: (batch_size*n) * dim_actions
        actions = actions.contiguous().view(-1, dim_actions)

        if self.args.advantages_per_action:
            # size: (batch_size*n) * dim_actions
            log_prob = multinomials_log_densities(actions, log_p_a)
            # the log prob of each action head is multiplied by the advantage
            action_loss = -advantages.contiguous().view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else: # any difference between this and the advantages_per_action after action_loss.sum()?
            # size: (batch_size*n) * 1
            log_prob = multinomials_log_density(actions, log_p_a)
            action_loss = -advantages.contiguous().view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()
        
        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        actor_loss = action_loss + self.args.value_coeff * value_loss
        
        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                actor_loss -= self.args.entr * entropy
                
        stat['actor_loss'] = actor_loss.item()
        
        actor_loss.backward()
        
        
#################### debug #######################
        coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        rewards = torch.Tensor(batch.reward)
        returns = torch.Tensor(batch_size, n)
        prev_coop_return = 0
        prev_ncoop_return = 0
        
        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])
            
        advantages = torch.Tensor(batch_size, n)
        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i]
        
        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        # element size: (batch_size*n) * num_actions[i]
        log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
        # size: (batch_size*n) * dim_actions
        actions = actions.contiguous().view(-1, dim_actions)

        if self.args.advantages_per_action:
            # size: (batch_size*n) * dim_actions
            log_prob = multinomials_log_densities(actions, log_p_a)
            # the log prob of each action head is multiplied by the advantage
            actor_loss = -advantages.contiguous().view(-1).unsqueeze(-1) * log_prob
            actor_loss *= alive_masks.unsqueeze(-1)
        else: # any difference between this and the advantages_per_action after actor_loss.sum()?
            # size: (batch_size*n) * 1
            log_prob = multinomials_log_density(actions, log_p_a)
            actor_loss = -advantages.contiguous().view(-1) * log_prob.squeeze()
            actor_loss *= alive_masks

        actor_loss = actor_loss.sum()
        stat['ref_actor_loss'] = actor_loss.item() 
        
#################### debug #######################
        
        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.critic_optimizer.zero_grad()
        s = self.comp_critic_grad(batch)
        merge_stat(s, stat)
        for p in self.critic_params:
            if p._grad is not None:
#                 print(p._grad.data)
                p._grad.data /= stat['num_steps']
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        s = self.comp_actor_grad(batch)
        merge_stat(s, stat)
        for p in self.actor_params:
            if p._grad is not None:
#                 print(p._grad.data)
                p._grad.data /= stat['num_steps']
        self.actor_optimizer.step()

        return stat

    def state_dict(self):
        return {'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_state_dict(self, state):
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
