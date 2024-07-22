# CO2 Prediction Dashboard

This project is a web application that visualizes historical CO2 levels and predicts future CO2 levels using a SARIMA model. The application is built using Dash and Plotly for the web interface and visualization, and uses a pre-trained SARIMA model for predictions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data](#data)
- [Model Training](#model-training)
- [Dashboard](#dashboard)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/co2-prediction-dashboard.git
    cd co2-prediction-dashboard
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Train the SARIMA model:**
    Ensure that your historical CO2 data is available in `data/raw/co2_data.csv`. Then run the model training script:
    ```sh
    python models/sarima_model.py
    ```

2. **Run the dashboard:**
    Start the Dash application to visualize the historical and predicted CO2 levels:
    ```sh
    python webapp/dashboard.py
    ```

3. **Open the dashboard:**
    Open your web browser and navigate to `http://127.0.0.1:8050/` to view the dashboard.

## Project Structure
co2-prediction-dashboard/
│
├── data/
│ └── raw/
│ └── co2_data.csv
│
├── models/
│ ├── sarima_model.py
│ ├── sarima_model.pkl
│ └── scaler.pkl
│
├── utils/
│ └── data_preprocessing.py
│
├── webapp/
│ └── dashboard.py
│
├── requirements.txt
└── README.md
Feel free to customize this README.md as needed for your project.


## Data

The historical CO2 data should be placed in the `data/raw/co2_data.csv` file. The data should have a `datetime` column for the date and a `value` column for the CO2 levels.

## Model Training

The SARIMA model is trained using the script in `models/sarima_model.py`. This script preprocesses the data, splits it into training and validation sets, trains the SARIMA model, and saves the trained model and scaler.

## Dashboard

The dashboard is implemented in `webapp/dashboard.py`. It loads the historical data and the trained SARIMA model, makes predictions, and visualizes both the historical and predicted CO2 levels using Plotly.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
