import time
from utils import *
import torch
import torch.multiprocessing as mp

class MultiProcessWorker(mp.Process):
    # TODO: Make environment init threadsafe
    def __init__(self, id, trainer_maker, comm, seed, *args, **kwargs):
        self.id = id
        self.seed = seed
        super(MultiProcessWorker, self).__init__()
        self.trainer = trainer_maker()
        self.comm = comm

    def run(self):
        torch.manual_seed(self.seed + self.id + 1)
        np.random.seed(self.seed + self.id + 1)

        while True:
            task = self.comm.recv()
            if type(task) == list:
                task, epoch = task

            if task == 'quit':
                return
            elif task == 'run_batch':
                batch, stat = self.trainer.run_batch(epoch)
                self.trainer.critic_optimizer.zero_grad()
                s = self.trainer.comp_critic_grad(batch)
                merge_stat(s, stat)
                self.comm.send(stat)
            elif task == 'send_critic_grads':
                critic_grads = []
                for p in self.trainer.critic_params:
                    if p._grad is not None:
                        critic_grads.append(p._grad.data)
                self.comm.send(critic_grads)
            elif task == 'comp_actor':
                self.trainer.actor_optimizer.zero_grad()
                s = self.trainer.comp_actor_grad(batch) 
                self.comm.send(s)
            elif task == 'send_actor_grads':
                actor_grads = []
                for p in self.trainer.actor_params:
                    if p._grad is not None:
                        actor_grads.append(p._grad.data)
                self.comm.send(actor_grads)


class MultiProcessTrainer(object):
    def __init__(self, args, trainer_maker):
        self.comms = []
        self.trainer = trainer_maker()
        # itself will do the same job as workers
        self.nworkers = args.nprocesses - 1
        for i in range(self.nworkers):
            comm, comm_remote = mp.Pipe()
            self.comms.append(comm)
            worker = MultiProcessWorker(i, trainer_maker, comm_remote, seed=args.seed)
            worker.start()
        self.critic_grads = None
        self.actor_grads = None
        self.worker_critic_grads = None
        self.worker_actor_grads = None
        self.is_random = args.random

    def quit(self):
        for comm in self.comms:
            comm.send('quit')

    def obtain_grad_pointers(self):
        # only need perform this once
        if self.critic_grads is None:
            self.critic_grads = []
            for p in self.trainer.critic_params:
                if p._grad is not None:
                    self.critic_grads.append(p._grad.data)
        if self.actor_grads is None:
            self.actor_grads = []
            for p in self.trainer.actor_params:
                if p._grad is not None:
                    self.actor_grads.append(p._grad.data)


    def train_batch(self, epoch):
        # run workers in parallel
        for comm in self.comms:
            comm.send(['run_batch', epoch])

        # run its own trainer
        batch, stat = self.trainer.run_batch(epoch)
        self.trainer.critic_optimizer.zero_grad()
        s = self.trainer.comp_critic_grad(batch)
        merge_stat(s, stat)

        # check if workers are finished
        for comm in self.comms:
            s = comm.recv()
            merge_stat(s, stat)

        # add gradients
        self.obtain_grad_pointers()
        # add critic gradients of workers
        if self.worker_critic_grads is None:
            self.worker_critic_grads = []
            for comm in self.comms:
                comm.send('send_critic_grads')
                self.worker_critic_grads.append(comm.recv())
                
        for i in range(len(self.critic_grads)):
            for g in self.worker_critic_grads:
                self.critic_grads[i] += g[i]
            self.critic_grads[i] /= stat['num_steps']
        self.trainer.critic_optimizer.step()
            
        self.trainer.update_target_critic()
        
        for comm in self.comms:
            comm.send(['comp_actor', epoch])
        
        self.trainer.actor_optimizer.zero_grad()
        s = self.trainer.comp_actor_grad(batch)
        merge_stat(s, stat)
        
        for comm in self.comms:
            s = comm.recv()
            merge_stat(s, stat)

        # add actor gradients of workers
        if self.worker_actor_grads is None:
            self.worker_actor_grads = []
            for comm in self.comms:
                comm.send('send_actor_grads')
                self.worker_actor_grads.append(comm.recv())
                
        for i in range(len(self.actor_grads)):
            for g in self.worker_actor_grads:
                self.actor_grads[i] += g[i]
            self.actor_grads[i] /= stat['num_steps']
        self.trainer.actor_optimizer.step()
        
        return stat

    def state_dict(self):
        return self.trainer.state_dict()

    def load_state_dict(self, state):
        self.trainer.load_state_dict(state)
