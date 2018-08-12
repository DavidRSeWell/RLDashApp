# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import pandas as pd
import plotly.graph_objs as go

from igraph import *

app = dash.Dash()
server = app.server
app.config.suppress_callback_exceptions = True

# Append Bootstrap

bootstrap_css = "https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css"

dashboard_css = "https://getbootstrap.com/docs/4.1/examples/dashboard/dashboard.css"

app.css.append_css({

    "external_url": bootstrap_css,
    'external_url_2': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

# setup Graph
nr_vertices = 4
v_label = map(str, range(nr_vertices))
nr_vertices = 4
G = Graph.Tree(4,2)
G.es["label"] = ["bet","check","call"]
G.vs["label"] = ["SB","BB","BB","SB"]
lay = G.layout('rt',root=(0,0))

# get positions of nodes / edges
position = {k: lay[k] for k in range(nr_vertices)}
Y = [lay[k][1] for k in range(nr_vertices)]
M = max(Y)

es = EdgeSeq(G) # sequence of edges
E = [e.tuple for e in G.es] # list of edges

L = len(position)
Xn = [position[k][0] for k in range(L)]
Yn = [2*M-position[k][1] for k in range(L)]
Xe = []
Ye = []
edge_labels = []

for edge in E:
    Xe+=[position[edge[0]][0],position[edge[1]][0], None]
    Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

labels = v_label


lines = go.Scatter(x=Xe,
                   y=Ye,
                   mode='lines',
                   line=dict(color='rgb(210,210,210)', width=1),
                   hoverinfo='none'
                   )
dots = go.Scatter(x=Xn,
                  y=Yn,
                  mode='markers',
                  name='',
                  marker=dict(symbol='dot',
                                size=18,
                                color='#6175c1',    #'#DB4551',
                                line=dict(color='rgb(50,50,50)', width=1)
                                ),
                  text=labels,
                  hoverinfo='text',
                  opacity=0.8
                  )


def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
    L = len(pos)
    if len(text) != L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = go.Annotations()
    for k in range(L):
        annotations.append(
            go.Annotation(
                text=G.vs['label'][k],  # or replace labels with a different list for the text within the circle
                x=pos[k][0], y=2 * M - position[k][1],
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False)
        )

    # now make line annotations
    for edge in G.es:
        edge_text = edge['label']
        node0, node1 = edge.tuple

        x0 = Xn[node0]
        x1 = Xn[node1]
        y0 = Yn[node0]
        y1 = Yn[node1]
        x_pos = (x0 - (x0 - x1) / 2.0)
        y_pos = (y0 - (y0 - y1) / 2.0)

        annotations.append(
            go.Annotation(
                text=edge_text,
                x=x_pos,
                y=y_pos,
                showarrow=False
            )
        )

    return annotations

axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            )

layout_game = dict(title= 'AKQ Game',
              annotations=make_annotations(position, G.vs['label']),
              font=dict(size=12),
              showlegend=False,
              xaxis=go.XAxis(axis),
              yaxis=go.YAxis(axis),
              margin=dict(l=40, r=40, b=85, t=100),
              hovermode='closest',
              plot_bgcolor='rgb(248,248,248)'
              )


data=go.Data([lines, dots])
fig=dict(data=data, layout=layout_game)
fig['layout'].update(annotations=make_annotations(position, G.vs['label']))


hands = [['A',0.33,0],
        ['K',0.33,0],
        ['Q',0.33,0]]



hands_df = pd.DataFrame(hands,columns=['hand','policy','value'])


layout = html.Div([

    html.Div(children=[

        html.Div([

            html.H4('Hand policy and value'),

            html.Div(children=[
                    dt.DataTable(
                            rows=hands_df.to_dict('records'),
                            # optional - sets the order of columns
                            columns=['hand','policy','value'],
                            row_selectable=True,
                            filterable=True,
                            #sortable=True,
                            selected_row_indices=[],
                            id='datatable-file'
                    )
                ]
            ,id='policy-datatable'),

            html.Pre(id='click-data', style=styles['pre']),

        ], className='col-sm-4'),

        html.Div(
            dcc.Graph(figure=fig,id='tree-1')
        ,className='col-sm-8')

    ],className='row'),

    html.Div(className='row', children=[

        html.Div([

            dcc.Graph(id='reward-graph')

        ],className='col-sm')

    ])


],style = {"margin-top":"50px"},className="container")


'''
@app.callback(
    dash.dependencies.Output('policy-datatable', 'children'),
    [dash.dependencies.Input('tree-1','clickData')])
def update_figure(clickData):

    hands = [
        {
            '0':['A',0.33,0],
            '1':['K',0.33,0],
            '2':['Q',0.33,0]
        }
    ]

    hands_df = pd.DataFrame(hands)

    table = dt.DataTable(
        rows=hands_df.to_dict('records'),
        # optional - sets the order of columns
        columns=['Hand','policy','value'],
        row_selectable=True,
        filterable=True,
        sortable=True,
        selected_row_indices=[],
        id='datatable-file'
    )

    return table'''





if __name__ == '__main__':



    app.layout = layout

    app.run_server(debug=True)

