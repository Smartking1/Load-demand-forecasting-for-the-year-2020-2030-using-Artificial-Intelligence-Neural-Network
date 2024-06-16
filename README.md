# Electrical Load-demand-using-Artificial-Neural-Network
# BreadcrumbsLoad-demand-forecasting-for-the-year-2020-2030-using-Artificial-Intelligence-Neural-Network

## Project Overview

This project aims to forecast energy consumption demand for the years 2020 to 2030 using Artificial Intelligence (AI) and Neural Networks, specifically Long Short-Term Memory (LSTM) models. The focus is on three sectors: Residential, Commercial, and Industrial energy consumption.

## Project Structure

### Data Preparation

- **Data Source**: Energy consumption data from the years 2008 to 2018.
- **Preprocessing**: Normalization of data using MinMaxScaler.

### Model Building

- **Model**: LSTM neural networks.
- **Training**: The models are trained using historical energy consumption data with a look-back period of 3 years.
- **Forecasting**: Energy consumption for each sector is forecasted from 2019 to 2030.

### Visualization

- **Graphs**: Visualization of actual and forecasted energy consumption.
- **Outputs**: Generated images showing energy consumption trends and forecasts.

## Installation and Requirements

To run this project, you need to have the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow
```

## Code Execution

The project code is structured in several steps, as outlined below:

### Step 1: Data Preparation

Load and prepare the data for modeling.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('/content/drive/MyDrive/Load Demand/Energy_data2.csv')
data.isnull().sum()  # Ensure no missing values
```

### Step 2: Normalize Data

Normalize the energy consumption data.

```python
data = {
    'Year': [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
    'Residential': [7910.08, 8075.00, 8205.20, 8285.60, 8350.00, 8773.13, 8933.23, 9093.33, 9253.43, 9413.53, 9573.63],
    'Commercial': [3852.00, 3865.50, 3925.80, 4004.70, 4025.40, 4424.78, 4542.21, 4659.64, 4777.07, 4894.50, 5011.93],
    'Industrial': [1502.50, 1585.00, 1589.40, 1615.50, 1648.00, 1615.08, 1617.73, 1620.38, 1623.03, 1625.68, 1628.33]
}

df = pd.DataFrame(data)
scalers = {}
for sector in ['Residential', 'Commercial', 'Industrial']:
    scaler = MinMaxScaler()
    df[sector + '_scaled'] = scaler.fit_transform(df[[sector]])
    scalers[sector] = scaler
```

### Step 3: Create Sequences for LSTM

Generate input and output sequences for the LSTM model.

```python
look_back = 3
sectors = ['Residential', 'Commercial', 'Industrial']

X = {sector: [] for sector in sectors}
y = {sector: [] for sector in sectors}

for i in range(len(df) - look_back):
    for sector in sectors:
        X[sector].append(df[sector + '_scaled'].values[i:i + look_back])
        y[sector].append(df[sector + '_scaled'].values[i + look_back])

for sector in sectors:
    X[sector] = np.array(X[sector])
    y[sector] = np.array(y[sector])
    X[sector] = np.reshape(X[sector], (X[sector].shape[0], X[sector].shape[1], 1))
```

### Step 4: Build LSTM Models

Build and compile LSTM models for each sector.

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense

models = {}
for sector in sectors:
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    models[sector] = model
```

### Step 5: Train the Models

Train the LSTM models for each sector.

```python
epochs = 100
batch_size = 1

for sector in sectors:
    models[sector].fit(X[sector], y[sector], epochs=epochs, batch_size=batch_size, verbose=2)
```

### Step 6: Forecast Energy Consumption

Forecast energy consumption for each sector from 2019 to 2030.

```python
forecast_horizon = 12
forecast = {sector: [] for sector in sectors}

for sector in sectors:
    last_sequence = X[sector][-1]
    for _ in range(forecast_horizon):
        next_year = models[sector].predict(last_sequence.reshape(1, look_back, 1))
        forecast[sector].append(next_year[0, 0])
        last_sequence = np.append(last_sequence[1:], next_year)

forecast_residential = scalers['Residential'].inverse_transform(np.array(forecast['Residential']).reshape(-1, 1))
forecast_commercial = scalers['Commercial'].inverse_transform(np.array(forecast['Commercial']).reshape(-1, 1))
forecast_industrial = scalers['Industrial'].inverse_transform(np.array(forecast['Industrial']).reshape(-1, 1))

forecast_years = range(2019, 2031)
forecast_df = pd.DataFrame({
    'Year': forecast_years,
    'Forecasted_Residential_Energy_Consumption': forecast_residential.flatten(),
    'Forecasted_Commercial_Energy_Consumption': forecast_commercial.flatten(),
    'Forecasted_Industrial_Energy_Consumption': forecast_industrial.flatten()
})

forecast_csv_filename = 'energy_consumption_forecast.csv'
forecast_df.to_csv(forecast_csv_filename, index=False)
print("Forecasted energy consumption data saved to:", forecast_csv_filename)
```

### Step 7: Visualization

Plot actual and forecasted energy consumption data.

```python
import matplotlib.pyplot as plt

# Load the data
data = {
    'Year': [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
    'Residential': [7910.08, 8075.00, 8205.20, 8285.60, 8350.00, 8773.13, 8933.23, 9093.33, 9253.43, 9413.53, 9573.63]
}
df = pd.DataFrame(data)

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Residential'], marker='o')
plt.title('Actual Residential Energy Consumption')
plt.xlabel('Year')
plt.ylabel('Energy Consumption (MW)')
plt.grid(True)
plt.tight_layout()
output_image_filename = 'residential_energy_consumption.png'
plt.savefig(output_image_filename)
plt.show()
print("Graph saved as:", output_image_filename)

# Repeat for other sectors and total energy consumption
```

## Results

- **Forecasted Data**: `energy_consumption_forecast.csv`
- **Graphs**: 
  - `residential_energy_consumption.png`
  - `energy_consumption_actual_values.png`
  - Forecasted energy consumption graphs for each sector and total energy consumption.

## Conclusion

The project successfully forecasts the energy consumption demand for the Residential, Commercial, and Industrial sectors from 2020 to 2030 using LSTM models. The results are visualized and saved as CSV and PNG files for further analysis and reporting.

## License

This project is licensed under the MIT License.

## Contact

For any queries or further information, please contact [Your Name] at [Your Email].

---

This README provides a comprehensive overview of the project, instructions for running the code, and details on the results and outputs generated.
