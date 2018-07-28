# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff
import json

from app import app
from dash.dependencies import Input, Output, State
from scipy.stats import norm,bernoulli,beta,binom


y0 = np.random.randn(50)
y1 = np.random.randn(50)
y2 = np.random.randn(50)
y3 = np.random.randn(50)
y4 = np.random.randn(50)
y5 = np.random.randn(50)
y6 = np.random.randn(50)
y7 = np.random.randn(50)
y8 = np.random.randn(50)
y9 = np.random.randn(50)

trace0 = go.Box(
    y=y0,
    name="lever1"
)
trace1 = go.Box(
    y=y1
)
trace2 = go.Box(
    y=y2
)
trace3 = go.Box(
    y=y3
)
trace4 = go.Box(
    y=y4
)
trace5 = go.Box(
    y=y5
)
trace6 = go.Box(
    y=y6
)
trace7 = go.Box(
    y=y7
)
trace8 = go.Box(
    y=y8
)
trace9 = go.Box(
    y=y9
)

layout_posterior = go.Layout(

    xaxis={'title': 'X'},
    yaxis={'title': 'Reward dist', 'range': [-1, 1]},

)

fig = dict(data=[trace0,trace1,trace3,trace4,trace5,trace6,trace7,trace8,trace9],layout=layout_posterior)

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

            html.Button(id='submit-button', n_clicks=0, children='Sample', className="btn btn-primary"),
            html.Button(id='run-model', n_clicks=0, children='Sample', className="btn btn-primary")

        ],className="col-sm-12")


    ],className="row"),

    html.Div([

        html.Div([

            html.H3(children="Reward distribution",className='text-center'),

            dcc.Graph(id='reward-graph', figure=fig)

        ],className="col-sm-12")

    ],className="row",style={"margin-top":"10px"}),

    html.Div([

        html.Div([

            html.H3(children="Loss Graph",className='text-center'),

            dcc.Graph(id='loss-graph', figure=fig)

        ],className="col-sm-12")

    ],className="row",style={"margin-top":"10px"}),

    # Hidden div inside the app that stores the intermediate value
    #html.Div(id='current_sample', style={'display': 'none'})


],style = {"margin-top":"50px"},className="container")



@app.callback(
    dash.dependencies.Output('reward-graph','figure'),
    [dash.dependencies.Input('submit-button','n_clicks')],
    [dash.dependencies.State('level-number','value')]
)
def update_current_sample(n_clicks,input1):

    traces = []
    for i in range(input1):

        # get random mean
        rand_mean = np.random.uniform(0,5)

        y_norm = norm.rvs(loc=rand_mean,size=1000)
        #y_norm = np.random.randn(100000)

        name = 'lever' + str(i)
        #tracei = go.Violin(
        #    y=y_norm,
        #    name=name
        #)
        #traces.append(tracei)

        tracei = {
            "type":"violin",
            "y":y_norm,
            "box": {
                "visible": True
            },
            "name":name
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
    [dash.dependencies.Input('run-model','n_clicks')]
)
def run_model(n_clicks):

    pass
