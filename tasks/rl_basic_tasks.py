import datetime
import time
import os
import sqlite3
import pandas as pd

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
def k_arm_bandit(lever_data,epochs):
    '''

    lever_data contains information for mean,std of lever distributions
    and then epsilon value
    :param lever_data:
    :param  epochs: number of iterations to run algo
    :return:

    '''

    avg_reward = 0
    for i in range(epochs):

        pass




