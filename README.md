# COVID-19 Case Predictor

This project implements a COVID-19 case prediction model using machine learning techniques in Python. It leverages historical COVID-19 data to forecast positive case counts in specific regions (Arkansas and California) and overall total positive cases.

## Table of Contents

* [Project Overview](#project-overview)

* [Features](#features)

* [Data Source](#data-source)

* [Methodology](#methodology)

  * [Linear Regression with Polynomial Features](#linear-regression-with-polynomial-features)

  * [Time Series Prediction with Keras](#time-series-prediction-with-keras)

* [Tools and Libraries](#tools-and-libraries)


## Project Overview

The goal of this project is to build and demonstrate a simple COVID-19 case prediction system. It explores two primary approaches:

1. **Polynomial Linear Regression**: To model the relationship between a simple time identifier (ID) and positive case counts.

2. **Time Series Forecasting (Keras)**: To prepare and train a basic neural network for potential future time-dependent predictions.

The project processes a CSV dataset containing COVID-19 case information, trains predictive models, visualizes their performance, and outputs a prediction for a future date.

## Outcome and Results

The project successfully demonstrates two different modeling approaches for COVID-19 case prediction:

**Linear Regression Models:**

* The accuracy of the linear regression models, measured by the $R^2$ score, is as follows:

  * **Total Positive Cases Prediction Accuracy:** $95.918\%$

  * **California (CA) Total Positive Tests Prediction Accuracy:** $92.179\%$

  * **Arkansas (AR) Total Positive Tests Prediction Accuracy:** $97.086\%$

* These high accuracy scores indicate that the polynomial linear regression models were able to capture a significant portion of the variance in the historical data for the respective target variables.

* The plots illustrate how well the polynomial linear regression models fit the historical data, showing the actual case counts against the predicted curves.

**Keras Neural Network:**

* The time series preparation correctly transforms the data into a supervised learning format, suitable for sequence modeling.

* The training loss plot for the Keras model (`plt.plot(history.history['loss'])`) indicates the model's performance during training, showing how the Mean Absolute Error decreases over epochs. A lower loss indicates better performance. The initial output `var1(t-1) var2(t-1) var3(t-1) ... var3(t) var4(t) var5(t)` with sample rows demonstrates the structure of the reframed data, which is crucial for time series input.

**Final Prediction:**
Based on the trained linear regression model for `Total Positive` cases, the prediction for Total Cases 1 day after 12/31/20 (the last date of the dataset) is $1,887,985$. This specific numerical output is a key result of the project's predictive capability.

---

## Features

* **Data Loading and Preparation**: Reads COVID-19 case data from a CSV file.

* **Polynomial Feature Engineering**: Transforms input features to capture non-linear relationships for linear regression.

* **Linear Regression Models**:

  * Predicts `Total Positive` cases.

  * Predicts `CA-Total Positive Tests` (California).

  * Predicts `AR-Total Positive Tests` (Arkansas).

* **Performance Evaluation**: Calculates and displays the accuracy of the linear regression models.

* **Time Series Data Transformation**: Includes a utility function to convert sequential data into a supervised learning format for neural networks.

* **Neural Network Model (Keras)**: Implements a simple sequential model for time series forecasting.

* **Prediction Output**: Provides a concrete prediction for the total positive cases on a specific future day.

* **Visualizations**: Generates plots to show actual vs. predicted values for linear regression and training loss for the neural network.

## Data Source

The dataset used in this project is sourced from **The COVID Tracking Project**. Specifically, it's derived from the `all-states-history.csv` file, which has been pre-filtered and refined to include only information pertaining to the states of Arkansas (AR) and California (CA).

The key input features utilized from the dataset are:

* `ID`: A numerical identifier, likely representing the day or sequence of records.

* `AR-Total Positive Tests`: Total positive cases in Arkansas.

* `AR-Total Tests`: Total tests administered in Arkansas.

* `CA-Total Postive Tests`: Total positive cases in California.

* `CA-Total Tests`: Total tests administered in California.

* `Total Positive`: Overall total positive cases (this is the primary target for one of the linear regression models).

## Methodology

### Linear Regression with Polynomial Features

The first part of the project focuses on using `scikit-learn`'s `LinearRegression` model combined with `PolynomialFeatures`.

1. **Input Feature (`x`)**: The `ID` column from the dataset is used as the independent variable.

2. **Polynomial Transformation**: `PolynomialFeatures(degree=2)` is applied to the `ID` column. This transforms the input `ID` into `[1, ID, ID^2]`, allowing the linear model to fit quadratic relationships with the target variables.

3. **Target Variables**:

   * `Total Positive`

   * `CA-Total Postive Tests`

   * `AR-Total Positve Tests`

4. **Model Training**: Separate `LinearRegression` models are trained for each of the three target variables using the transformed `ID` feature.

5. **Evaluation**: The `model.score()` method is used to determine the $R^2$ accuracy of each linear regression model.

6. **Visualization**: Plots are generated to compare the actual positive case counts against the values predicted by the linear regression models.

### Time Series Prediction with Keras

The second part of the project sets up a basic time series forecasting framework using `Keras` (part of TensorFlow).

1. **Data Preparation**:

   * A helper function `series_to_supervised` is defined. This function transforms a given time series into a supervised learning dataset, where past observations (lagged values) become input features for predicting a future observation. In this implementation, it uses `n_in=1` (one past timestep as input) and `n_out=1` (one future timestep as output).

   * The `LabelEncoder` is used, specifically on the 5th column of the dataset values (`values[:,4]`), which suggests it might be handling a categorical feature, although the data types are subsequently cast to `float32`.

   * `MinMaxScaler` is applied to normalize all features to a range between 0 and 1. This is crucial for neural network training.

2. **Train-Test Split**: The prepared time series data is split into training and testing sets based on a fixed number of "hours" (`n_train_hours = 365 * 24`). This split is then reshaped to fit the input requirements of an LSTM-like neural network (`[samples, timesteps, features]`), even though the subsequent Keras model doesn't explicitly use an LSTM layer.

3. **Model Architecture**: A simple `Sequential` Keras model is defined with a single `Dense` layer. While the input shape is prepared for a recurrent network (like LSTM), the current model is a simple feed-forward network.

4. **Compilation**: The model is compiled with `loss='mae'` (Mean Absolute Error) and `optimizer='adam'`.

5. **Training**: The neural network is trained for 50 epochs using the prepared training and validation data.

6. **Visualization**: A plot showing the training loss history over epochs is displayed to monitor the model's learning progress.

## Tools and Libraries

The project utilizes the following Python libraries:

* **`pandas`**: For data manipulation and analysis, such as reading CSV files and handling DataFrames.

* **`numpy`**: For numerical operations, especially array manipulation.

* **`tensorflow` & `keras`**: For building and training neural networks.

* **`matplotlib.pyplot`**: For creating static, interactive, and animated visualizations in Python.

* **`sklearn.preprocessing.PolynomialFeatures`**: To generate polynomial and interaction features.

* **`sklearn.linear_model.LinearRegression`**: To implement the ordinary least squares Linear Regression model.

* **`sklearn.preprocessing.LabelEncoder`**: To encode target labels with values between 0 and `n_classes-1`.

* **`sklearn.preprocessing.MinMaxScaler`**: To transform features by scaling each feature to a given range (e.g., 0 to 1).

* **`sklearn.model_selection.train_test_split`**: For splitting datasets into training and testing sets (imported but not explicitly used for the final train/test split in the time series part).

* **`seaborn`**: (Imported, but not directly used in the provided code snippet). A library for statistical data visualization.

* **`pylab.rcParams`**: For modifying Matplotlib's runtime configuration parameters.

* **`pandas.plotting.register_matplotlib_converters`**: To ensure Pandas plots work correctly with Matplotlib's date handling.

## Disclaimer

The dataset used in this project was obtained from The Covid Tracking Project, specifically from their `all-states-history.csv` file. This data was subsequently filtered and refined to include only information from the states of Arkansas and California. The input feature vectors included Dates (represented as `ID`), Location, Positive Cases, and Total Tests Administered. The primary output demonstrated in the final prediction is the Total predicted Case count after 1 day from the last recorded date in the dataset.

This project serves as a demonstration of applying machine learning techniques to publicly available health data for predictive purposes. The accuracy and reliability of predictions are dependent on the quality, completeness, and characteristics of the input data, as well as the complexity of the chosen models. For real-world applications, more sophisticated models, continuous data updates, and thorough validation would be required.
