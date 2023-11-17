# California Housing Price Prediction

## Overview

This machine learning project focuses on predicting housing prices in California based on various features, including median income, house age, and average room count. The dataset used is the California Housing dataset, obtained from the StatLib repository.

## Dataset Information

The California Housing dataset comprises 20,640 instances with 8 numeric predictive attributes and the target variable:

- **Features:**
  - MedInc: Median income in the block group
  - HouseAge: Median house age in the block group
  - AveRooms: Average number of rooms per household
  - AveBedrms: Average number of bedrooms per household
  - Population: Block group population
  - AveOccup: Average number of household members
  - Latitude: Block group latitude
  - Longitude: Block group longitude

- **Target Variable:**
  - Price: Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000).

## Dependencies

Ensure you have the following Python libraries installed:

- `scikit-learn`
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`

Install dependencies using:

pip install scikit-learn pandas numpy seaborn matplotlib
Getting Started
Clone the repository:

Copy code
git clone https://github.com/chandraparsad3/California-Housing-Price-Prediction.git
cd California-Housing-Price-Prediction
Install Dependencies.

Run the Jupyter notebook or Python script:

Execute the California_Housing_Prediction.ipynb Jupyter notebook or Python script.
Project Structure
California_Housing_Prediction.ipynb: Jupyter notebook with project code.
scaler.pkl: Pickle file containing the trained StandardScaler.
regressor.pkl: Pickle file containing the trained Linear Regression model.
Usage
Load the trained scaler and regressor:

import pickle

# Load scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load regressor
regressor = pickle.load(open('regressor.pkl', 'rb'))
Preprocess new data:

# Assuming new_data is your new input data
new_data_scaled = scaler.transform(new_data)
Make predictions:

predictions = regressor.predict(new_data_scaled)
print(predictions)
Model Evaluation
The model has been evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared. Adjusted R-squared is also calculated for more accuracy.

MAE: 0.54
MSE: 0.55
RMSE: 0.74
R-squared: 0.60
Adjusted R-squared: 0.60
Feel free to experiment with the notebook and make improvements based on your specific use case.
