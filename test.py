import sys
import time
import signal
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import visdom
import data
from models import *
from comm import CommNetMLP
from gc_comm import GCCommNetMLP
from ga_comm import GACommNetMLP
from gcomm import GCommNetMLP
from tar_comm import TarCommNetMLP
from utils import *
from action_utils import parse_action_args
from trainer import Trainer
from multi_processing import MultiProcessTrainer
import gym

gym.logger.set_level(40)

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=1,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=16,
                    help='How many processes to run')
# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--qk_hid_size', default=16, type=int,
                    help='key and query size for soft attention')
parser.add_argument('--value_hid_size', default=32, type=int,
                    help='value size for soft attention')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')
parser.add_argument('--directed', action='store_true', default=False,
                    help='if the graph formed by the agents is directed')
parser.add_argument('--self_loop', action='store_true', default=False,
                    help='if self loop in gnn')
parser.add_argument('--gnn_type', default='gat', type=str,
                    help='type of gnn to use (gcn|gat)')
# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed') # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
# other
parser.add_argument('--run_num', default=1, type=int,
                    help='load models in which run')
parser.add_argument('--ep_num', default=100, type=int,
                    help='load models saved from which epoch')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')


parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")

# CommNet specific args
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False,
                    help="enable ic3net model")
parser.add_argument('--gccomm', action='store_true', default=False,
                    help="enable gccomm model (with commnet or ic3net)")
parser.add_argument('--tarcomm', action='store_true', default=False,
                    help="enable tarmac model (with commnet or ic3net)")
parser.add_argument('--gacomm', action='store_true', default=False,
                    help="enable gacomm model")
parser.add_argument('--gcomm', action='store_true', default=False,
                    help="enable gcomm model")
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')


init_args_for_env(parser)
args = parser.parse_args()

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games
    if args.env_name == "traffic_junction":
        args.comm_action_one = True

if args.gacomm or args.gcomm:
    args.commnet = 1
    args.mean_ratio = 0

# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

render = args.render

args.render = False
env = data.init(args.env_name, args, False)

num_inputs = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'


parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)

print(args)

if args.gcomm:
    policy_net = GCommNetMLP(args, num_inputs)
elif args.gacomm:
    policy_net = GACommNetMLP(args, num_inputs)
elif args.commnet:
    if args.gccomm:
        policy_net = GCCommNetMLP(args, num_inputs)
    elif args.tarcomm:
        policy_net = TarCommNetMLP(args, num_inputs)
    else:
        policy_net = CommNetMLP(args, num_inputs)
elif args.random:
    policy_net = Random(args, num_inputs)
elif args.recurrent:
    policy_net = RNN(args, num_inputs)
else:
    policy_net = MLP(args, num_inputs)

if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
disp_trainer.display = True
def disp():
    x = disp_trainer.get_episode()

args.render = render
if args.nprocesses > 1:
    trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
else:
    trainer = Trainer(args, policy_net, data.init(args.env_name, args))


if args.gcomm:
    model_dir = Path('./saved') / args.env_name / 'gcomm' / args.gnn_type
elif args.gacomm:
    model_dir = Path('./saved') / args.env_name / 'gacomm'
elif args.gccomm:
    if args.ic3net:
        model_dir = Path('./saved') / args.env_name / 'gc_ic3net'
    elif args.commnet:
        model_dir = Path('./saved') / args.env_name / 'gc_commnet'
    else:
        model_dir = Path('./saved') / args.env_name / 'other'
elif args.tarcomm:
    if args.ic3net:
        model_dir = Path('./saved') / args.env_name / 'tar_ic3net'
    elif args.commnet:
        model_dir = Path('./saved') / args.env_name / 'tar_commnet'
    else:
        model_dir = Path('./saved') / args.env_name / 'other'
elif args.ic3net:
    model_dir = Path('./saved') / args.env_name / 'ic3net'
elif args.commnet:
    model_dir = Path('./saved') / args.env_name / 'commnet'
else:
    model_dir = Path('./saved') / args.env_name / 'other'
if args.env_name == 'grf':
    model_dir = model_dir / args.scenario
curr_run = 'run' + str(args.run_num)

run_dir = model_dir / curr_run 
    
def run(num_epochs): 
    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        for n in range(args.epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
            batch, s = trainer.run_batch(ep)
            print('batch: ', n)
            merge_stat(s, stat)
            trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        epoch = ep + 1

        np.set_printoptions(precision=2)

        print('Epoch {}\tReward {}\tTime {:.2f}s'.format(
                epoch, stat['reward']/stat['num_episodes'], epoch_time
        ))

def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    trainer.load_state_dict(d['trainer'])

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if args.ep_num == 0:
    path = run_dir / 'model.pt'
else:
    path = run_dir / ('model_ep%i.pt' %(args.ep_num))
    
load(path)

run(args.num_epochs)
if args.display:
    env.end_display()

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    import os
    os._exit(0)
