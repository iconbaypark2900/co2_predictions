import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import requests
import plotly.graph_objs as go

# Initialize the Dash application
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("CO2 Prediction Dashboard"),
    
    dcc.Input(id='input-sequence', type='text', placeholder='Enter CO2 sequence'),
    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='prediction-output'),

    dcc.Graph(id='prediction-graph')
])

@app.callback(
    Output('prediction-output', 'children'),
    Output('prediction-graph', 'figure'),
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('input-sequence', 'value')]
)
def update_output(n_clicks, sequence):
    if n_clicks > 0 and sequence:
        # Prepare the sequence for prediction
        sequence = [float(num) for num in sequence.split(',')]  # Example conversion, adjust as needed
        payload = {'sequence': sequence}
        
        # Make a request to the Flask API for prediction
        response = requests.post('http://localhost:5000/predict', json=payload)
        prediction = response.json().get('prediction')

        # Prepare data for the graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=sequence + prediction, mode='lines', name='CO2 Levels'))

        return f'Predicted CO2 Level: {prediction}', fig

    return '', go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)
