import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from Dash.apps import bayes_predict, ES_1


app.layout = html.Div([

    html.Div([

        html.Div([

            html.Div([
                html.Button(id="bayes-page", children="Bayes", className="btn btn-secondary"),
                html.Button(id="evolution-page", children="Evolution", className="btn btn-secondary"),

            ], className="btn-group-vertical"),

        ],className="col-sm-2"),

        html.Div([

            dcc.Location(id='url', refresh=False),
            html.Div(id='page-content')

        ], className="col-sm-10"),


    ],className="row"),

],style = {"margin-top":"50px"}, className="container")

@app.callback(Output('page-content', 'children'),
              [Input('bayes-page', 'pathname'),
               Input('evolution-page', 'value')])
def display_page(pathname,value):

    if pathname == '/apps/app1':
         return ES_1.layout
    elif pathname == '/apps/app2':
         return bayes_predict.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)