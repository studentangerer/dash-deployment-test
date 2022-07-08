from collections import defaultdict

import dash  # version 1.13.1
from dash import dcc, html
from dash.dependencies import Input, Output, ALL, State, MATCH, ALLSMALLER
import plotly.express as px
import pandas as pd
import numpy as np

import sim_v1
class global_data:
    number_of_runs = 1
    sim_duration = 10800  # Zeit in Sekunden(3h), in welcher Kunden kommen
    onequeue = False
    number_of_checkout_points = 30
    prob_cust_spon = 0.2
    prob_cust_reg = 0.4
    prob_cust_stock = 0.4
    distribution_type = "exponential"
    distribution_value = 6
    data_visualization = "dashboard"
    queue_desicion = "shortest"
    line_length_data = sim_v1.dashboard_scenario(sim_duration, onequeue, number_of_checkout_points,prob_cust_spon, prob_cust_reg,
                       prob_cust_stock, distribution_type, distribution_value, data_visualization, queue_desicion)
    print(f"We are in dashboard:")
    print(line_length_data)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Markdown('''
            ### Discrete Event Simulation
            Input some values'''),
    ]),
    html.Div(
        [
        dcc.Markdown("How long customers will arrive [s]"),
        dcc.Input(id = "sim_duration_field", value=5000),
        dcc.Markdown("Number of queues"),
        dcc.Input(id='number_of_checkout_points', value=4),
        dcc.Markdown("Probabilitys of customer types:"),
        dcc.Input(id='prob_cust_spon_filed', placeholder="spontaneous", value=0.2),
        dcc.Input(id='prob_cust_reg_filed', placeholder="regular", value=0.4),
        dcc.Input(id='prob_cust_stock_filed', placeholder="stock", value=0.4),
        dcc.Markdown("Distribution Type Arrivals:"),
        dcc.Dropdown(['exponential', 'random'], id='distribution_type_field', value='exponential'),
        dcc.Markdown("Distribution value"),
        dcc.Input(id="distribution_value_field", value=6),
        dcc.Markdown("How to customers should select to queue:"),
        dcc.Dropdown(['random', 'shortest'], id='queue_desicion', value="random", style={'width': '30%'}),
    
    ]),
    html.Div(children=[
        html.Button("Run Simulation", id='run_simulation_button'),
        html.Div(id='output-state')
    ]),
    html.Div(id='container', children=[])
])
@app.callback(
    Output('container', 'children'),
    Input('run_simulation_button', 'n_clicks'),
    State('sim_duration_field', 'value'),
    State('number_of_checkout_points', 'value'),
    State('prob_cust_spon_filed', 'value'),
    State('prob_cust_reg_filed', 'value'),
    State('prob_cust_stock_filed', 'value'),
    State('distribution_type_field', 'value'),
    State('distribution_value_field', 'value'),
    State('queue_desicion','value'),
    
)
def update_values(n_clicks, sim_duration, number_of_checkout_points, prob_cust_spon, prob_cust_reg, prob_cust_stock,
                  distribution_type, distribution_value, queue_desicion):
    print("update_values was triggered")
    div_children = []
    global_data.line_length_data = sim_v1.dashboard_scenario(int(sim_duration), False, int(number_of_checkout_points),
                                float(prob_cust_spon), float(prob_cust_reg), float(prob_cust_stock), distribution_type,
                                                        int(distribution_value),
                                                             "dashboard", queue_desicion)
    print("We ran the simulation")
    for x in range(int(number_of_checkout_points)):
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
                    options=[{'label': 'Line Chart', 'value': 'line'},
                             {'label': 'Pie Chart', 'value': 'pie'}],
                    value='line',
                ),
            ]
        )
        div_children.append(new_child)
    return div_children


@app.callback(
    Output({'type': 'dynamic-graph', 'index': MATCH}, 'figure'),
    [Input({'type': 'dynamic-choice', 'index': MATCH}, 'value'),
     State({'type': 'dynamic-choice', 'index': MATCH}, 'id')]
)

def update_graph(chart_choice, id):
    line_number = id["index"]
    if line_number == 0:
        print(global_data.line_length_data[0][0])
        print(global_data.line_length_data[0][1])

    if chart_choice == 'bar':

        return {}
    elif chart_choice == 'line':
        print(f"LineNumber: {line_number}")
        x_vals = global_data.line_length_data[line_number][0]
        y_vals = global_data.line_length_data[line_number][1]
        df = px.line(x=x_vals, y=y_vals)
        fig = px.line(df, x=x_vals, y=y_vals, labels={"x": "Simulation Time (min)",
                                                      "y": "Number of Customer in Line"},
                      title=f"Line Number: {line_number+1}")
        fig.update_traces(line_shape="hv")
        return fig
    else:
        return {}
    
    




if __name__ == '__main__':
    app.run_server(debug=True)

