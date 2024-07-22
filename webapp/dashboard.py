import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('models/sarima_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load historical data
historical_data = pd.read_csv('data/raw/co2_data.csv', parse_dates=['datetime'], index_col='datetime')

# Number of years to predict
years = 5  # You can change this value as needed
steps = years * 12

# Predict using the SARIMA model
predictions_scaled = model.get_forecast(steps=steps).predicted_mean
predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

# Generate future dates for the prediction
future_dates = pd.date_range(start=historical_data.index[-1], periods=steps + 1, freq='M')[1:]

# Initialize the Dash application
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("CO2 Prediction Dashboard"),
    dcc.Graph(
        id='prediction-graph',
        figure={
            'data': [
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data['value'],
                    mode='lines',
                    name='Historical CO2 Levels',
                    line=dict(color='blue')
                ),
                go.Scatter(
                    x=future_dates,
                    y=predictions.flatten(),
                    mode='lines',
                    name='Predicted CO2 Levels',
                    line=dict(color='orange', dash='dash')
                )
            ],
            'layout': go.Layout(
                title='CO2 Levels: Historical and Predicted',
                xaxis_title='Date',
                yaxis_title='CO2 Levels (ppm)',
                legend_title='Legend',
                hovermode='x unified',  # Show all data points for a given x value
                template='plotly_white',  # Use a clean white template
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                annotations=[
                    dict(
                        x=future_dates[0],
                        y=predictions.flatten()[0],
                        text=f"Start: {predictions.flatten()[0]:.2f} ppm",
                        showarrow=True,
                        arrowhead=1
                    ),
                    dict(
                        x=future_dates[-1],
                        y=predictions.flatten()[-1],
                        text=f"End: {predictions.flatten()[-1]:.2f} ppm",
                        showarrow=True,
                        arrowhead=1
                    )
                ]
            )
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)