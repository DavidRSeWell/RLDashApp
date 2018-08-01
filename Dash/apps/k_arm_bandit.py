# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
import pandas as pd
import sqlite3
import plotly.figure_factory as ff
import json
import h5py
import time
import dash_table_experiments as dt
import os

from app import app
from dash.dependencies import Input, Output, State,Event
from scipy.stats import norm,bernoulli,beta,binom
from tasks.rl_basic_tasks import k_arm_bandit


layout_posterior = go.Layout(

    xaxis={'title': 'X'},
    yaxis={'title': 'Reward dist', 'range': [-1, 1]},

)

fig = dict(data=[],layout=layout_posterior)

layout = html.Div([

    html.Div([

        html.Div([

            html.Form([

                html.Div([

                    html.Label('Lever number'),
                    dcc.Slider(
                        id='lever-number',
                        min=1,
                        max=10,
                        marks={i: str(i) for i in range(1, 11)},
                        value=5,
                    ),

                ],className="form-group"),

                html.Div([

                    html.Label('Epsilon'),
                    dcc.Slider(
                        id='epsilon-value',
                        min=0,
                        max=1,
                        step=0.1,
                        value=0,
                    ),

                ],className="form-group"),


            ],id='model-parameter-form'),

            html.Button(id='submit-button', n_clicks=0, type="button",children='Sample', className="btn btn-primary"),
            html.Button(id='run-model', n_clicks=0, type="button", children='Run', className="btn btn-primary"),
            html.Button(id='toggle-model', n_clicks=0, type="button", children='Toggle', className="btn btn-primary")

        ],className="col-sm-12"),



    ],className="row"),

    html.Div([

        html.Div([

            html.H3(children="Reward distribution",className='text-center'),

            dcc.Graph(id='reward-distribution', figure=fig)

        ],className="col-sm-12")

    ],className="row",style={"margin-top":"10px"}),

    html.Div([

        html.Div([

            html.H3(children="Loss Graph",className='text-center'),

            dcc.Graph(id='reward-graph')

        ],className="col-sm-12")

    ],className="row",style={"margin-top":"10px"}),

    html.Div([
        html.Div([
            html.H4('Completed tests DataTable'),
            dt.DataTable(
                    rows=[],
                    # optional - sets the order of columns
                    columns=['files'],
                    row_selectable=True,
                    filterable=True,
                    sortable=True,
                    selected_row_indices=[],
                    id='datatable-file'
                )

        ],id='file-datatable-container',className="col-sm-12"),

        html.Button(id='fetch-files', n_clicks=0, type="button", children='GetFiles', className="btn btn-primary"),
        html.Button(id='delete-files', n_clicks=0, type="button", children='DeleteFiles', className="btn btn-danger")

    ],className="row"),

    html.Div(id='interval-container'),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='model-state',children='stop'),

    html.Div(id='run-model-hidden', style={'display': 'none'}),
    html.Div(id='place-holder-div', style={'display': 'none'}),


],style = {"margin-top":"50px","margin-left":"0px","margin-right":"0px"},className="container",id='main-container')



#################
# Lever Graph
#################
@app.callback(
    dash.dependencies.Output('reward-distribution','figure'),
    [dash.dependencies.Input('submit-button','n_clicks')],
    [dash.dependencies.State('lever-number','value')]
)
def update_current_sample(n_clicks,input1):

    traces = []
    for i in range(input1):

        # get random mean
        rand_mean = np.random.uniform(0,5)

        y_norm = norm.rvs(loc=rand_mean,size=1000)

        name = 'lever' + str(i) + ' mean: ' + str(rand_mean)

        tracei = {
            "type":"violin",
            "y":y_norm,
            "box": {
                "visible": True
            },
            "name":name,
            "mean":rand_mean,
            "std":1
        }

        traces.append(tracei)


    layout_posterior = go.Layout(

        xaxis={'title': 'lever number'},
        yaxis={'title': 'Reward dist', 'range': [-1, 1]},

    )

    fig = dict(data=traces)

    return fig

'''
@app.callback(
    dash.dependencies.Output('reward-graph','figure'),
    [dash.dependencies.Input('interval-component','n_intervals')],
    [dash.dependencies.State('model-state','children'),
     dash.dependencies.State('epsilon-value', 'value'),
     dash.dependencies.State('lever-number', 'value')]
)
def display_loss(n_clicks,model_state,epsilon_value,lever_number):

    if model_state == 'stop':

        return {'data':[]}

    print("Inverval")

    time.sleep(2)

    path_name = 'kbandit_{num_levers}_{epsilon}.h5'.format(num_levers=lever_number, epsilon=epsilon_value)

    hdf_file_path = '/Users/befeltingu/RLResearch/Data/' + path_name

    f = h5py.File(hdf_file_path, 'r', libver='latest',
                  swmr=True)

    dset = f["avg_reward"][:] # fetch all the datas

    figure = {
        'data': [
            go.Scatter(
                x=[x for x in range(len(dset))],
                y=dset,

            )
        ],
        'layout': go.Layout(
            xaxis={'title': 'iteration'},
            yaxis={'title': 'loss'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
        )
    }

    f.close()

    return figure
'''


#################
# Reward Graph
#################

@app.callback(
    Output('reward-graph', 'figure'),
    [Input('datatable-file', 'selected_row_indices')],
    [State('datatable-file', 'rows')])
def update_figure(selected_row_indices,rows):

    dff = pd.DataFrame(rows)

    traces = []

    for i in (selected_row_indices or []):

        file_name = str(rows[i]['files'])

        file_path = '/Users/befeltingu/RLResearch/Data/k_arm_bandit/' + file_name

        f = h5py.File(file_path, 'r', libver='latest',
                      swmr=True)

        dset = f["avg_reward"][:]  # fetch all the datas

        num_bandits = file_name.split('_')[1]

        #epsilon = float(file_name.split('_')[-1][:3])

        trace = go.Scatter(
                    x=[x for x in range(len(dset))],
                    y=dset,
                    name=file_name

                )

        traces.append(trace)

        f.close()

    figure = {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'iteration'},
            yaxis={'title': 'loss'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            #legend={'x': 0, 'y': 1},
        )
    }

    return figure


#################
# Run Model
#################
@app.callback(
    dash.dependencies.Output('run-model-hidden','children'),
    [],
    [dash.dependencies.State('reward-distribution','figure'),
     dash.dependencies.State('epsilon-value', 'value')],
    [Event('run-model', 'click')]
)
def run_model(reward_figure,epsilon_value):

    print("Running run_model task")

    path_name = 'kbandit_{num_levers}_{epsilon}.h5'.format(num_levers=len(reward_figure['data']),epsilon=epsilon_value)

    hdf_file_path = '/Users/befeltingu/RLResearch/Data/k_arm_bandit/' + path_name

    epochs = 1000

    result = k_arm_bandit.delay(hdf_file_path, reward_figure, epochs, epsilon_value)

    dset, avg_reward, q_values, count_values = result.get()

    print("Done running model")
    print("avg reward: " + str(avg_reward))

#################
# File DataTable
#################
@app.callback(
    dash.dependencies.Output('datatable-file','rows'),
    [],
    [],
    [Event('fetch-files','click')]
)
def populate_file_container():


    file_list = []
    for file in os.listdir('/Users/befeltingu/RLResearch/Data/k_arm_bandit/'):
        file_list.append(file)

    file_df = pd.DataFrame({'files':file_list})


    return file_df.to_dict('records')

@app.callback(
    dash.dependencies.Output('place-holder-div','children'),
    [],
    [],
    [Event('delete-files','click')]
)
def populate_file_container():

    for file in os.listdir('/Users/befeltingu/RLResearch/Data/k_arm_bandit/'):

        os.remove('/Users/befeltingu/RLResearch/Data/k_arm_bandit/' + file)

    return ''



'''
@app.callback(
    dash.dependencies.Output('interval-container','children'),
    [],
    [],
    [Event('run-model', 'click')]
)
def run_model():

    return dcc.Interval(
        id='interval-component',
        interval=2 * 1000,  # in milliseconds
        n_intervals=0
    )'''


@app.callback(
    dash.dependencies.Output('model-state','children'),
    [],
    [dash.dependencies.State('model-state','children')],
    [Event('toggle-model', 'click')]
)
def toggle_model(model_state):

    if model_state == 'stop':
        return "start"

    else:
        return "stop"



