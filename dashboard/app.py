from flask import Flask
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px


server = Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)


df = pd.read_csv('C:/Users/teeyob/EagleEye_Fraud_Detection/data/merged_fraud_data.csv')  # Update with your actual data path


df['signup_time'] = pd.to_datetime(df['signup_time'])
df['purchase_time'] = pd.to_datetime(df['purchase_time'])
df['month'] = df['purchase_time'].dt.to_period('M')
df['is_fraud'] = df['class']  


app.layout = html.Div([
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Fraud Trends Over Time', children=[
            html.H3('Monthly Fraud Trends'),
            dcc.Graph(id='line-chart')
        ]),
        dcc.Tab(label='Geographic Fraud Analysis', children=[
            html.H3('Fraud Cases by Location'),
            dcc.Graph(id='geo-bar-chart')
        ]),
        dcc.Tab(label='Fraud by Device', children=[
            html.H3('Fraud Cases by Device'),
            dcc.Graph(id='device-bar-chart')
        ]),
        dcc.Tab(label='Fraud by Browser', children=[
            html.H3('Fraud Cases by Browser'),
            dcc.Graph(id='browser-bar-chart')
        ]),
    ])
])


@app.callback(
    Output('line-chart', 'figure'),
    Input('tabs', 'value')
)
def update_line_chart(_):
    fraud_trends = df[df['is_fraud'] == 1].groupby(df['month']).size().reset_index(name='fraud_cases')
    fraud_trends['month'] = fraud_trends['month'].dt.to_timestamp()
    fig = px.line(fraud_trends, x='month', y='fraud_cases', title="Monthly Fraud Cases Over Time")
    return fig

@app.callback(
    Output('geo-bar-chart', 'figure'),
    Input('tabs', 'value')
)
def update_geo_chart(_):
    geo_data = df[df['is_fraud'] == 1]['country'].value_counts().reset_index()
    geo_data.columns = ['Country', 'Fraud Cases']
    fig = px.bar(geo_data, x='Country', y='Fraud Cases', title="Fraud Cases by Location")
    return fig

@app.callback(
    Output('device-bar-chart', 'figure'),
    Input('tabs', 'value')
)
def update_device_chart(_):
    device_data = df[df['is_fraud'] == 1]['device_id'].value_counts().reset_index()
    device_data.columns = ['Device', 'Fraud Cases']
    fig = px.bar(device_data, x='Device', y='Fraud Cases', title="Fraud Cases by Device")
    return fig

@app.callback(
    Output('browser-bar-chart', 'figure'),
    Input('tabs', 'value')
)
def update_browser_chart(_):
    browser_data = df[df['is_fraud'] == 1]['browser'].value_counts().reset_index()
    browser_data.columns = ['Browser', 'Fraud Cases']
    fig = px.bar(browser_data, x='Browser', y='Fraud Cases', title="Fraud Cases by Browser")
    return fig

# Run server
if __name__ == '__main__':
    app.run_server(debug=True)
