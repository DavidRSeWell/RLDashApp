# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import json
import base64

from app import app
from dash.dependencies import Input, Output, State
from scipy.stats import norm,bernoulli,beta,binom



possible_holdings = [[x,"No",y,0,0] for x in range(2,22) for y in range(1,11)]
possible_holdings_Ace = [[x,"Yes",y,0,0] for x in range(2,22) for y in range(1,11)]

possible_holdings = possible_holdings + possible_holdings_Ace

possible_holdings_df = pd.DataFrame(possible_holdings,columns=["holding","Ace","Dealer","hit","stand"])

print("states={}".format(len(possible_holdings_df)))

layout = html.Div([

    html.Div([

        html.Div([
            html.H4(children="Black holdings"),
            dt.DataTable(
                rows=possible_holdings_df.to_dict('records'),
                # optional - sets the order of columns
                columns=["holding","Ace","Dealer", "hit", "stand"],
                row_selectable=True,
                filterable=True,
                sortable=True,
                selected_row_indices=[],
                id='datatable-file'
            )


        ], className="col-sm"),


    ],className="row"),

    html.Div([
        html.Div([
            html.Button("Run Model",className="btn btn-primary")
        ])
    ]),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='current_sample', style={'display': 'none'})


],style = {"margin-top":"50px"},className="container")


if __name__ == '__main__':

    app = dash.Dash()

    server = app.server

    app.config.suppress_callback_exceptions = True

    app.layout = layout

    app.run_server(debug=True)
