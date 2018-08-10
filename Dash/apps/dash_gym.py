# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff
import json
import base64

from app import app
from dash.dependencies import Input, Output, State
from scipy.stats import norm,bernoulli,beta,binom


init_video = open("/Users/befeltingu/RLResearch/Data/gym/videos/CartPole-v1.mp4", 'rt').read()

encoded_data = base64.b64encode(init_video)


##########################
# prediction distribution
##########################
global posterior_histogram
posterior_histogram = []


layout = html.Div([

    html.Div([

        html.Div([

            html.Video(
                id='test-video',src='data:video/mp4;base64,{}'.format(encoded_data)
            )
        ], className="col-sm"),


    ],className="row"),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='current_sample', style={'display': 'none'})


],style = {"margin-top":"50px"},className="container")

