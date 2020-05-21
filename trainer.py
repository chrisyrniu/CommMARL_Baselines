 from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *

Transition = namedtuple('Transition', ('state', 'hidden_state', 'action', 'action_out', 'episode_mask', 'episode_mini_mask', 'next_state', 'reward', 'misc'))


class Trainer(object):
    def __init__(self, args, actor, critic, target_critic, env):
        self.args = args
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
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
                action_out = self.actor(x, info)

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

            trans = Transition(state, hidden_state, action, action_out, episode_mask, episode_mini_mask, next_state, reward, misc)
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
    
    def update(self, batch):
        # action space
        num_actions = self.args.num_actions
        # number of action head
        dim_actions = self.args.dim_actions        

        n = self.nagents
        batch_size = len(batch.state)
        # size: batch_size * n * hid_size
        hidden_state = torch.stack(batch.hidden_state, dim=0)
        # size: batch_size * (n*hid_size)
        hidden_state = hidden_state.view(batch_size, n*self.args.hid_size)
        # size: batch_size * n
        rewards = torch.Tensor(batch.state)
        actions = torch.Tensor(batch.action)
        # size: batch_size * n * dim_actions.  have been detached
        actions = actions.transpose(1, 2).view(-1, n, dim_actions) 
        # size: batch_size * n
        episode_masks = torch.Tensor(batch.episode_mask)
        # size: batch_size * n
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        # size: (batch_size*n)
        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)
        
        ## update critic
        self.critic_optimizer.zero_grad()

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
            
        # size: batch_size * 1
        # Q(macro_hidden_state, (macro_action_head1, macro_action_head2, ...))
        # the input size is fixed even if there can be dead agents
        critic_out = self.critic(critic_in)
        # target critic should use hidden states and actions from next time step
        next_critic_in = critic_in[:-1, :].clone()
        next_critic_out = self.target_critic(next_critic_in)
        # pad 0 in the end
        next_critic_out = torch.cat([next_critic_out, torch.zeros((1, 1))], dim=0)
        # expand size of critic_out:
        critic_out = critic_out.expand(batch_size, n)
        next_critic_out = next_critic_out.expand(batch_size, n)
        # use episode_mask to mark the episode as end, so they do not have next critic
        # use episode_mini_mask to mark an agent just complete (dead), so it does not has next critic
        # in tj env, if an agent will die in the next step, there will be no reward, and the value in \
        # episode_mini_masks and alive masks will be 0
        td_errors = rewards + self.args.gamma * next_critic_out * episode_masks * episode_mini_masks - critic_out
        # mask the dead agents (in fact the agent is dead after taking action? so should mask next step?)
        # size: batch_size * n
        alive_masks = alive_masks.view(batch_size, n)
        # pad at the start
        if self.args.env_name == "traffic_junction":
            alive_masks = torch.cat([torch.zeros(1, n), alive_masks[1:, :]], dim=0)
        td_errors *= alive_masks
        crtic_loss = td_errors.pow(2).sum() # divided by?
        
        # backward and update
        critic_loss.backward()
        for p in self.critic_params:
            if p._grad is not None:
                print(p.grad.data)
                p._grad.data /= batch_size
                print(p.grad.data)        
        
        self.critic_optimizer.step()
        
        
        ## update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.args.tau) + param.data * self.args.tau)  
            
            
        ## update actor
        advantages = torch.Tensor(batch_size, n)
        alive_masks = alive_masks.view(-1)
        action_out = list(zip(*batch.action_out))
        # length: dim_actions
        # element size: batch_size * n * num_actions[i]
        action_out = [torch.cat(a, dim=0) for a in action_out]
        
        returns = self.critic(critic_in)
        returns = returns.expand(batch_size, n)
        
        # calculate counterfactual baseline
        
        
        
        

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
        actions = actions.contiguous().view(-1, dim_actions)

        if self.args.advantages_per_action:
            log_prob = multinomials_log_densities(actions, log_p_a)
        else:
            log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

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
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        for p in self.actor_params:
            if p._grad is not None:
                print(p.grad.data)
                p._grad.data /= stat['num_steps']
                print(p.grad.data)
        self.optimizer.step()

        return stat

    def state_dict(self):
        return {'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_state_dict(self, state):
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
