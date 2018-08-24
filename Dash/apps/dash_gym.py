# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff
import json
import base64

#from app import app
from dash.dependencies import Input, Output, State
from scipy.stats import norm,bernoulli,beta,binom


init_video = open("/Users/befeltingu/RLResearch/Data/gym/videos/CartPole-v1.mp4", 'rt').read()

encoded_data = base64.b64encode(init_video)

app = dash.Dash()

server = app.server

app.config.suppress_callback_exceptions = True

##########################
# prediction distribution
##########################
global posterior_histogram
posterior_histogram = []


layout = html.Div([

    html.Div([

        html.Div([

            html.Video(
                id='test-video',preload="auto",autoPlay="autoplay",src='data:video/mp4;base64,{}'.format(encoded_data)
            )
        ], id='video-holder-guy',className="col-sm"),


    ],className="row"),

    html.Button(id='load-video', n_clicks=0, type="button", children='GetVideo', className="btn btn-primary"),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='current_sample', style={'display': 'none'})


],style = {"margin-top":"50px"},className="container")


@app.callback(
    dash.dependencies.Output('test-video','autoPlay'),
    [dash.dependencies.Input('load-video','n_clicks')]
)
def update_current_sample(n_clicks):

    return "autoplay"



if __name__ == '__main__':

    app.layout = layout

    app.run_server(debug=True)
