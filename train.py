#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 08:06:03 2022

@author: mohammad
"""

"""
first reference:
super mario project in 
https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
"""

import torch
from torch import nn
from torchvision import transforms as T
from src.model import MarioNet

from src.env import create_env
from src.logger import MetricLogger

import numpy as np
from collections import deque
import random, datetime, os, copy

from pathlib import Path


class Mario_train:
    def __init__(self, state_dim, action_dim, log_dir, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_dir = log_dir
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net
        
        self.memory = deque(maxlen=15000)
        self.batch_size = 32
        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4 
        self.learn_every = 3  
        self.sync_every = 1e4

    def act(self, state):
        
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
        #self.net.target_conv.load_state_dict(self.net.conv.state_dict())
        #self.net.target_fc.load_state_dict(self.net.fc.state_dict())
    def save(self):
        '''save_path = (
            self.save_dir / 
            f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), 
                 exploration_rate=self.exploration_rate),
            save_path,
        )'''
        #save_path = self.save_dir #+ "/mario_net.pt"
        torch.save(self.net.state_dict(), self.save_dir)
        print(f"MarioNet saved to {self.save_dir} at step {self.curr_step}")
    
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            #self.save()
            pass

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
    

            
if __name__ == "__main__":   
    env = create_env()

    env.reset()
    next_state, reward, done, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()


    log_dir = Path("checkpoints") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S")
    log_dir.mkdir(parents=True)
    
    save_dir = Path("trained_model", "mario_net.pt") 
    #save_dir.mkdir(parents=True)

    mario = Mario_train(state_dim=(4, 84, 84), action_dim=env.action_space.n, 
                  log_dir=log_dir, save_dir=save_dir)

    logger = MetricLogger(log_dir)

    episodes = 1000
    for e in range(episodes):

        state = env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if e % 200 == 0:
            logger.record(episode=e, epsilon=mario.exploration_rate, 
                          step=mario.curr_step)
    
    mario.save()
