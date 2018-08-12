import datetime
import time
import os
import sqlite3
import pandas as pd
import numpy as np
import h5py
import random
import gym




if __name__ == '__main__':

    all_envs = gym.envs.registry.all()

    env = gym.make('NChain-v0')
    #state_size = env.observation_space.shape[0]
    #action_size = env.action_space.n

    for episode in range(10000):

        state = env.reset()

        done = False
        while(done == False):

            action_forward = 1

            next_state, reward, done, _ = env.step(action_forward)

            if done == True:
                if reward > 2:
                    print("REWARD: " + str(reward))


    print("Done")
