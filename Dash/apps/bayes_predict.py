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

# global theta value
global theta
theta = 0

n = 10 # for lambert example

# posterior dist
a = 3
b = 9

x_beta = np.linspace(beta.ppf(0.01,a,b),beta.ppf(0.99,a,b),100)

y_beta = beta.pdf(x_beta,a,b)

x1 = np.random.randn(200)

hist_data = [x1]

trace = go.Scatter(
    x = x_beta,
    y = y_beta
)

layout_posterior = go.Layout(

            xaxis={'title': 'X'},
            yaxis={'title': 'Prob Theta', 'range': [0, 5]},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1}
        )

fig2 = dict(data=[trace],layout=layout_posterior)


##########################
# prediction distribution
##########################
global posterior_histogram
posterior_histogram = []


layout = html.Div([

    html.Div([

        html.Div([
            html.H3(children="Posterior Distribution"),
            dcc.Graph(id='dist-1',figure=fig2),
            html.Button(id='submit-button', n_clicks=0, children='Sample', className="btn btn-primary")

        ], className="col-sm"),


        html.Div([
            html.H3(children="Likelihood"),
            dcc.Graph(id='dist-2'),

        ], className="col-sm"),

        html.Div([
            html.H3(children="Predicted Posterior"),
            dcc.Graph(id='dist-3'),

        ], className="col-sm"),


    ],className="row"),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='current_sample', style={'display': 'none'})


],style = {"margin-top":"50px"},className="container")


@app.callback(
    dash.dependencies.Output('current_sample','children'),
    [dash.dependencies.Input('submit-button','n_clicks')]

)
def update_current_sample(n_clicks):

    # first sample from current posterior beta distribution
    p = beta.rvs(a, b, size=1)[0]
    print("Sample: " + str(p))

    return json.dumps({'sample':p})



@app.callback(
    dash.dependencies.Output('dist-2', 'figure'),
    [dash.dependencies.Input('current_sample','children')])
def update_figure(sample_data):

    # first sample from current posterior beta distribution
    sample_data = json.loads(sample_data)

    p = sample_data['sample']

    x_binom_pdf = np.arange(0,
                  10)

    y_binom_pdf = binom.pmf(x_binom_pdf, n, p)


    trace = go.Scatter(
        x=x_binom_pdf,
        y=y_binom_pdf
    )

    layout_posterior = go.Layout(

        xaxis={'title': 'X'},
        yaxis={'title': 'Prob X', 'range': [0, 1]},
        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
        legend={'x': 0, 'y': 1}
    )

    fig2 = dict(data=[trace], layout=layout_posterior)

    return fig2


@app.callback(
    dash.dependencies.Output('dist-3', 'figure'),
    [dash.dependencies.Input('current_sample','children')])
def update_figure(sample_data):

    # first sample from current posterior beta distribution
    sample_data = json.loads(sample_data)

    p = sample_data['sample']

    sample_x = binom.rvs(n,p,1)

    posterior_histogram.append(sample_x)

    trace = go.Histogram(
        x=posterior_histogram,

        xbins=dict(
            start=0,
            end=10,
            size=1
        )
    )

    layout_posterior = go.Layout(

        xaxis={'title': 'X', 'range': [0,10]},
        yaxis={'title': 'Prob X~'},
        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
        legend={'x': 0, 'y': 1}
    )

    fig3 = dict(data=[trace],layout=layout_posterior)

    return fig3