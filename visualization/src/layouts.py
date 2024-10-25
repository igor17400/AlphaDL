from dash import html, dcc

def create_layout():
    return html.Div([
        html.H1('Stock Market Prediction Dashboard'),
        dcc.Dropdown(
            id='model-selector',
            options=[
                {'label': 'LSTM', 'value': 'lstm'},
                {'label': 'Transformer', 'value': 'transformer'}
            ],
            value='lstm'
        ),
        dcc.Graph(id='prediction-graph'),
        dcc.Graph(id='evaluation-metrics')
    ])
