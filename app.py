from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__, template_folder='templates')

# Load the dataset
df = pd.read_csv('dataset.csv')

# Verify the dataset is loaded correctly
print("Dataset Preview:")
print(df.head())

# Train the regression model for predicting rainfall
X_rainfall = df[['temperature', 'wind_speed']]
y_rainfall = df['rainfall']
rainfall_model = LinearRegression()
rainfall_model.fit(X_rainfall, y_rainfall)

# Train the regression model for predicting temperature
X_temperature = df[['rainfall', 'wind_speed']]
y_temperature = df['temperature']
temperature_model = LinearRegression()
temperature_model.fit(X_temperature, y_temperature)

# Threshold values
FLOOD_LEVEL_THRESHOLD = 80  # Rainfall level triggering flood warnings
DAM_MAX_LEVEL = 100  # Dam overflow threshold
DROUGHT_TEMPERATURE_THRESHOLD = 40  # High temperature threshold for drought

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_rainfall', methods=['GET', 'POST'])
def predict_rainfall():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            temperature = float(request.form['temperature'])
            wind_speed = float(request.form['wind_speed'])
            prediction = rainfall_model.predict([[temperature, wind_speed]])[0]
        except ValueError:
            error = "Please enter valid numerical values for temperature and wind speed."
        except Exception as e:
            error = str(e)
    return render_template('predict_rainfall.html', prediction=prediction, error=error)

@app.route('/predict_temperature', methods=['GET', 'POST'])
def predict_temperature():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            rainfall = float(request.form['rainfall'])
            wind_speed = float(request.form['wind_speed'])
            prediction = temperature_model.predict([[rainfall, wind_speed]])[0]
        except ValueError:
            error = "Please enter valid numerical values for rainfall and wind speed."
        except Exception as e:
            error = str(e)
    return render_template('predict_temperature.html', prediction=prediction, error=error)

@app.route('/monitor', methods=['POST'])
@app.route('/monitor', methods=['GET', 'POST'])
def monitor():
    result = None
    if request.method == 'POST':
        try:
            rainfall = float(request.form['rainfall'])
            dam_level = float(request.form['dam_level'])
            temperature = float(request.form['temperature'])

            result = ""
            if rainfall > FLOOD_LEVEL_THRESHOLD:
                result += "Flood warning: Open floodgates.\n"
            if dam_level > DAM_MAX_LEVEL:
                result += "Dam overflow warning: Open floodgates immediately.\n"
            if temperature > DROUGHT_TEMPERATURE_THRESHOLD:
                result += "Drought warning: Temperature is critically high.\n"

            if not result:
                result = "All systems normal."
        except Exception as e:
            result = f"Error: {e}"
    return render_template('monitor.html', result=result)


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
