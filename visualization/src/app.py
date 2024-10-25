import dash
from dash import html, dcc
from layouts import create_layout
from callbacks import register_callbacks

app = dash.Dash(__name__)

app.layout = create_layout()
register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)
