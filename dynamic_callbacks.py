import dash  # version 1.13.1
from dash import dcc, html
from dash.dependencies import Input, Output, ALL, State, MATCH, ALLSMALLER
import plotly.express as px
import pandas as pd
import numpy as np

df = pd.read_csv(
    "https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Callbacks/Pattern%20Matching%20Callbacks/Caste.csv")
df.rename(columns={'under_trial': 'under trial', 'state_name': 'state'}, inplace=True)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div(children=[
        dcc.Input(id='add-chart'),
    ]),
    html.Div(id='container', children=[])
])


@app.callback(
    Output('container', 'children'),
    [Input('add-chart', 'value')],
    [State('container', 'children')]
)
def display_graphs(n_clicks, div_children):

    print(n_clicks)
    print(type(int(n_clicks)))
    div_children = []
    for x in range(int(n_clicks)):
        new_child = html.Div(
            style={'width': '45%', 'display': 'inline-block', 'outline': 'thin lightgrey solid', 'padding': 10},
            children=[
                dcc.Graph(
                    id={
                        'type': 'dynamic-graph',
                        'index': x
                    },
                    figure={}
                ),
                dcc.RadioItems(
                    id={
                        'type': 'dynamic-choice',
                        'index': x
                    },
                    options=[{'label': 'Bar Chart', 'value': 'bar'},
                             {'label': 'Line Chart', 'value': 'line'},
                             {'label': 'Pie Chart', 'value': 'pie'}],
                    value='bar',
                ),
                dcc.Dropdown(
                    id={
                        'type': 'dynamic-dpn-s',
                        'index': x
                    },
                    options=[{'label': s, 'value': s} for s in np.sort(df['state'].unique())],
                    multi=True,
                    value=["Andhra Pradesh", "Maharashtra"],
                ),
                dcc.Dropdown(
                    id={
                        'type': 'dynamic-dpn-ctg',
                        'index': x
                    },
                    options=[{'label': c, 'value': c} for c in ['caste', 'gender', 'state']],
                    value='state',
                    clearable=False
                ),
                dcc.Dropdown(
                    id={
                        'type': 'dynamic-dpn-num',
                        'index': x
                    },
                    options=[{'label': n, 'value': n} for n in ['detenues', 'under trial', 'convicts', 'others']],
                    value='convicts',
                    clearable=False
                )
            ]
        )
        div_children.append(new_child)
    return div_children


@app.callback(
    Output({'type': 'dynamic-graph', 'index': MATCH}, 'figure'),
    [Input(component_id={'type': 'dynamic-dpn-s', 'index': MATCH}, component_property='value'),
     Input(component_id={'type': 'dynamic-dpn-ctg', 'index': MATCH}, component_property='value'),
     Input(component_id={'type': 'dynamic-dpn-num', 'index': MATCH}, component_property='value'),
     Input({'type': 'dynamic-choice', 'index': MATCH}, 'value')]
)
def update_graph(s_value, ctg_value, num_value, chart_choice):
    #print(s_value)
    dff = df[df['state'].isin(s_value)]

    if chart_choice == 'bar':
        dff = dff.groupby([ctg_value], as_index=False)[['detenues', 'under trial', 'convicts', 'others']].sum()
        fig = px.bar(dff, x=ctg_value, y=num_value)
        return fig
    elif chart_choice == 'line':
        if len(s_value) == 0:
            return {}
        else:
            dff = dff.groupby([ctg_value, 'year'], as_index=False)[
                ['detenues', 'under trial', 'convicts', 'others']].sum()
            fig = px.line(dff, x='year', y=num_value, color=ctg_value)
            return fig
    elif chart_choice == 'pie':
        fig = px.pie(dff, names=ctg_value, values=num_value)
        return fig


if __name__ == '__main__':
    app.run_server(debug=True)

# https://youtu.be/4gDwKYaA6ww