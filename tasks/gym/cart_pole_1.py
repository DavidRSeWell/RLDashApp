import datetime
import time
import os
import sqlite3
import pandas as pd
import numpy as np
import h5py
import random
import gym

from scipy.stats import norm
from celery import Celery
from scipy.stats import norm



def policy_iteration():
    '''

    :return:
    '''




if __name__ == '__main__':

    env = gym.make('Blackjack-v0')
    #state_size = env.observation_space.shape[0]
    #action_size = env.action_space.n

    for episode in range(100):

        state = env.reset()
        done = False
        print("Dealt hand: {hand_1} : {hand_2}".format(hand_1=env.player[0],hand_2=env.player[1]))
        while(done==False):

            random_action = np.random.randint(0,2)

            next_state, reward, done, _ = env.step(random_action)

            print("Player was dealt: {}".format(env.player[-1]))
            print("player total is now: {}".format(next_state))

            print('Episode {episode}'.format(episode=str(episode)))

    print('done running the silly cart')