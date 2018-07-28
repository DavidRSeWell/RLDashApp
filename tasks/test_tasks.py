import datetime
import time
import os

from celery import Celery


celery_app = Celery('test', backend='redis://localhost', broker='redis://localhost')


@celery_app.task
def test_graph(hdf_path):

    pass