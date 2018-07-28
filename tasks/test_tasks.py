import datetime
import time
import os
import sqlite3
import pandas as pd

from scipy.stats import norm
from celery import Celery


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

celery_app = Celery('test_graph', backend='redis://localhost', broker='redis://localhost')


@celery_app.task(name='tasks.test_tasks.test_graph')
def test_graph():

    delete_q = "delete from LossTable where 1 = 1"

    execute_sql('/Users/befeltingu/RLResearch/Data/test_db',delete_q)

    for i in range(100):

        y_norm = norm.rvs(loc=0, size=1)

        insert_q = "Insert into LossTable values(0,'test_model', {inc}, {loss_val})".format(inc=i,loss_val=y_norm[0])

        execute_sql('/Users/befeltingu/RLResearch/Data/test_db',insert_q)