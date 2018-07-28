import sqlite3
import pandas as pd

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



run_create_tables = 0
if run_create_tables:

    q = "CREATE TABLE LossTable(test_id int, model_name text, inc int, loss_value real)"

    execute_sql('test_db',q)


run_insert_into_loss = 0
if run_insert_into_loss:


    for i in range(100):

        y_norm = norm.rvs(loc=0, size=1)

        insert_q = "Insert into LossTable values(0,'test_model', {inc}, {loss_val})".format(inc=i,loss_val=y_norm[0])

        execute_sql('test_db',insert_q)


run_view_db = 0
if run_view_db:

    q = 'select * from LossTable'

    conn = sqlite3.connect('test_db')

    db_df = pd.read_sql(q,conn)

    print('done viewing db')








