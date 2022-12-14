#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:31:24 2022

@author: mohammad
"""

from src.model import MarioNet

from src.env import create_env

from pathlib import Path

import torch
from matplotlib import animation
import matplotlib.pyplot as plt
import subprocess as sp
from IPython import display


class Mario_test:
    def __init__(self, state_dim, action_dim, trained_model_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trained_model_dir = trained_model_dir
        
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")
            
        self.net.load_state_dict(torch.load(trained_model_dir))
        self.net.eval()
        
        self.curr_step = 0
        
    def act(self, state):
        state = state.__array__()
        if self.use_cuda:
            state = torch.tensor(state).cuda()
        else:
            state = torch.tensor(state)
        state = state.unsqueeze(0)
        action_values = self.net(state, model="online")
        action_idx = torch.argmax(action_values, axis=1).item()

        self.curr_step += 1
        return action_idx
    
'''class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", 
                        "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", 
                        "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.__array__())'''

    
'''def save_frames_as_gif(frames, path='./', filename='mario_test.mp4'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), 
               dpi=72)

    patch = plt.imshow(frames[4])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), 
                                   interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)'''


if __name__ == "__main__":
    env = create_env()

    use_cuda = torch.cuda.is_available()
    trained_model_dir = Path("trained_model", "mario_net.pt")
    mario = Mario_test(state_dim=(4, 84, 84), action_dim=env.action_space.n, 
                  trained_model_dir=trained_model_dir)
    
    state = env.reset()
    #monitor = Monitor(256, 240, Path("gif", "mario_test.mp4"))
    frames = []
    while True:
        #frames.append(env.render(mode="rgb_array"))
        plt.imshow(env.render(mode='rgb_array'))
        display.display(plt.gcf())
        display.clear_output(wait=True)
        action = mario.act(state)
        #monitor.record(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        #frames.append(env.render(mode="rgb_array"))
        if done or info["flag_get"]:
            break
        
    #save_frames_as_gif(frames, path='./gif/')        
        