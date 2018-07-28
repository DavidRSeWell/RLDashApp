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

from app import app
from dash.dependencies import Input, Output, State,Event
from scipy.stats import norm,bernoulli,beta,binom
from tasks.test_tasks import test_graph


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
                        id='level-number',
                        min=1,
                        max=10,
                        marks={i: str(i) for i in range(1, 11)},
                        value=5,
                    ),

                ],className="form-group"),

                html.Div([

                    html.Label('Epsilon'),
                    dcc.Slider(
                        min=0,
                        max=1,
                        step=0.1,
                        value=0,
                    ),

                ],className="form-group"),

            ]),

            html.Button(id='submit-button', n_clicks=0, type="button",children='Sample', className="btn btn-primary"),
            html.Button(id='run-model', n_clicks=0, type="button", children='Run', className="btn btn-primary")

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

            dcc.Graph(id='loss-graph')

        ],className="col-sm-12")

    ],className="row",style={"margin-top":"10px"}),

    dcc.Interval(
        id='interval-component',
        interval=2*1000, # in milliseconds
        n_intervals=0
    ),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='run-model-hidden', style={'display': 'none'})


],style = {"margin-top":"50px"},className="container")



@app.callback(
    dash.dependencies.Output('reward-distribution','figure'),
    [dash.dependencies.Input('submit-button','n_clicks')],
    [dash.dependencies.State('level-number','value')]
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

@app.callback(
    dash.dependencies.Output('loss-graph','figure'),
    [dash.dependencies.Input('interval-component','n_intervals')]
)
def display_loss(n_intervals):

    q = 'select * from LossTable'

    conn = sqlite3.connect('/Users/befeltingu/RLResearch/Data/test_db')

    db_df = pd.read_sql(q, conn)

    figure = {
        'data': [
            go.Scatter(
                x=db_df['inc'],
                y=db_df['loss_value'],

            )
        ],
        'layout': go.Layout(
            xaxis={'title': 'iteration'},
            yaxis={'title': 'loss'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
        )
    }
    conn.close()

    return figure



@app.callback(
    dash.dependencies.Output('run-model-hidden','children'),
    [],
    [dash.dependencies.State('reward-distribution','figure')],
    [Event('run-model', 'click')]
)
def run_model(figure):
    print("Running run_model task")
    test_graph.delay()



