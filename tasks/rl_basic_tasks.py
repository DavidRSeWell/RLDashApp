import datetime
import time
import os
import sqlite3
import pandas as pd
import numpy as np
import h5py

from scipy.stats import norm
from celery import Celery
from scipy.stats import norm


def format_query(q):

    q = q.replace('\n',' ')

    q = ' '.join(q.split())

    return q

def execute_sql(db,q):

    conn = sqlite3.connect(db)

    c = conn.cursor()

    q = format_query(q)

    c.execute(q)

    conn.commit()

    conn.close()



celery_app = Celery('rl_basic', backend='redis://localhost', broker='redis://localhost')


@celery_app.task(name='tasks.rl_basic_tasks.k_arm_bandit')
def k_arm_bandit(hdf_file,lever_data,epochs,epsilon):

    '''

    lever_data contains information for mean,std of lever distributions
    and then epsilon value
    :param lever_data:
    :param  epochs: number of iterations to run algo
    :return:

    '''
    lever_data = lever_data['data']

    f = h5py.File(hdf_file, 'w', libver='latest')

    arr = np.array([0.0]) # init avg_reward

    dset = f.create_dataset("avg_reward", chunks=(2,), maxshape=(None,), data=arr)

    f.swmr_mode = True

    count_values = np.zeros(len(lever_data))

    q_values = np.zeros(len(lever_data))

    avg_reward = 0
    for i in range(1,epochs + 1):

        print("Epoch: " + str(i))

        print("Avg Reward: " + str(avg_reward))

        print("Reward Vector: " + str(q_values))

        rand_uniform = np.random.uniform(0,1)

        action = None # will be the index of the lever selected

        if rand_uniform <= epsilon:
            # take random action
            action = np.random.randint(0,len(lever_data))
        else:
            action = np.argmax(q_values)

        mean_current_lever = lever_data[action]['mean']

        std_current_lever = lever_data[action]['std']

        reward = norm.rvs(loc=mean_current_lever,scale=std_current_lever)

        # update lever average and count
        count_values[action] += 1

        q_values[action] = q_values[action] + (1 / count_values[action]) * (reward - q_values[action])

        # Update overall average reward
        avg_reward = avg_reward + (1.0 / i) * (reward - avg_reward)

        new_shape = (len(dset) + 1,)

        dset.resize(new_shape)

        dset[-1] = np.array([avg_reward])

        dset.flush()

    data = list(dset[:])

    q_dset = f.create_dataset("q_values",data=q_values)

    lever_count = f.create_dataset("lever_count",data=count_values)

    q_dset.flush()

    lever_count.flush()


    f.close()

    return data,avg_reward,list(q_values),list(count_values)




if __name__ == '__main__':


    lever_data = {
        'data':
        [
            {
                'mean':1,
                'std':1
            },
            {
                'mean': 2,
                'std': 1
            },
            {
                'mean': 3,
                'std': 1
            }
        ]
    }

    data, avg_reward, q_values, count_values = k_arm_bandit('/Users/befeltingu/RLResearch/Data/test_loss.h5',lever_data,1000,.1)

    print("done testing brosehpha")




