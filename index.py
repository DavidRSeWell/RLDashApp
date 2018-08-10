import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt

from dash.dependencies import Input, Output,State
from app import app
from Dash.apps import bayes_predict, ES_1,datatable_example,graph_test,k_arm_bandit,dash_gym
from Dash.apps.gym import blackjack
'''
<nav class="navbar navbar-dark fixed-top bg-dark flex-md-nowrap p-0 shadow">
            <a class="navbar-brand col-sm-3 col-md-2 mr-0 text-primary" href="#">Machine Learning</a>
            <ul class="navbar-nav px-3">
                <li class="nav-item text-nowrap">
                    <a class="nav-link" href="#">Sign out</a>
                </li>
            </ul>
</nav>
'''

app.layout = html.Div([

        html.Div([

            html.Div([

                html.Div([

                    html.A(
                        html.Button('K-Arm Bandit', type="button", className="btn btn-primary")
                        , href="/k_arm"),

                    html.A(
                        html.Button('Bayes Predict', type="button", className="btn btn-primary")
                        , href="/bayes_predict"),

                    html.A(
                        html.Button('ES Strategies',type="button", className="btn btn-primary")
                    ,href="/ES_1"),

                    html.A(
                        html.Button('DataTable',type="button", className="btn btn-primary")
                    ,href="/dash_table"),

                    html.A(
                        html.Button('TestGraph',type="button", className="btn btn-primary")
                    ,href="/test_graph"),

                    html.A(
                        html.Button('GYM',type="button", className="btn btn-primary")
                    ,href="/dash_gym"),
                    html.A(
                        html.Button('BlackJack',type="button", className="btn btn-primary")
                    ,href="/blackjack"),


                ], className="btn-group-vertical")

            ],className="col-sm-2"),

            html.Div([

                dcc.Location(id='url', refresh=False),
                html.Div(id='page-content')

            ], className="col-sm-10")


        ],className="row"),

        html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'})

    ],style = {"margin-top":"50px","margin-left":"0px","margin-right":"0px"}, className="container")






@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):

    if pathname == '/ES_1':

         return ES_1.layout

    elif pathname == '/bayes_predict':

         return bayes_predict.layout

    elif pathname == '/dash_table':

        return datatable_example.layout

    elif pathname == '/test_graph':

        return graph_test.layout

    elif pathname == '/k_arm':

        return k_arm_bandit.layout

    elif pathname == '/dash_gym':

        return dash_gym.layout

    elif pathname == '/blackjack':

        return blackjack.layout

    elif pathname == '/':

        return bayes_predict.layout

    else:
        return "404"



if __name__ == '__main__':

    app.run_server(debug=True)