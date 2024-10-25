from dash.dependencies import Input, Output
import joblib
import plotly.graph_objs as go

def register_callbacks(app):
    @app.callback(
        [Output('prediction-graph', 'figure'),
         Output('evaluation-metrics', 'figure')],
        [Input('model-selector', 'value')]
    )
    def update_graphs(selected_model):
        # Load predictions and results
        predictions = joblib.load('predictions.joblib')
        results = joblib.load('results.joblib')

        # Create prediction graph
        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(y=predictions[selected_model], mode='lines', name='Predicted'))
        pred_fig.add_trace(go.Scatter(y=predictions['actual'], mode='lines', name='Actual'))
        pred_fig.update_layout(title='Stock Price Prediction')

        # Create evaluation metrics graph
        metrics = results[selected_model]
        eval_fig = go.Figure(data=[go.Bar(x=list(metrics.keys()), y=list(metrics.values()))])
        eval_fig.update_layout(title='Model Evaluation Metrics')

        return pred_fig, eval_fig
