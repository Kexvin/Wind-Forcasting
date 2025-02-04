# Wind Speed Data Visualization and Analysis

This project is a data visualization and analysis program that reads wind speed data from multiple CSV files, processes it, and offers a simple command-line interface for visualizing and analyzing the data.

## Project Overview

The program is designed to work with wind speed data collected from 2007 to 2012. It reads data from six CSV files, concatenates them into a single Pandas DataFrame, and creates a `Datetime` column by combining year, month, day, hour, and minute fields. Users interact with the program via a command-line menu that provides multiple options for data exploration and analysis.

## File Structure

- **main.py**  
  Acts as the primary script for the project. It handles:
  - **Data Loading**: Reads CSV files, concatenates the data, and creates a unified `Datetime` column.
  - **User Interface**: Presents a menu with options for:
    - Viewing a histogram of daily mean wind speeds with a Weibull distribution overlay.
    - Building and viewing a regression model to predict wind speeds for future years.
    - Displaying a list of libraries used in the project.
    - Exiting the program.
  
- **methods.py**  
  Contains the core functions used for data visualization and analysis:
  1. **plot_histogram(wind_speeds)**  
     - Generates a histogram of wind speed data.
     - Fits a Weibull distribution using Scipy and overlays the probability density function (PDF) on the histogram.
     - Uses Matplotlib and NumPy for plotting and numerical calculations.
  
  2. **fit_and_plot_regression(X, y)**  
     - Fits a linear regression model using historical data.
     - Prompts the user to specify a forecast period (2-15 years) for future wind speed predictions.
     - Plots historical wind speeds and overlays the regression line that includes predictions.
     - Calculates performance metrics such as Mean Squared Error (MSE) and R² score.
     - Uses Scikit-learn, Matplotlib, and NumPy.
  
  3. **r2_manualCalc(y_true, y_pred)**  
     - Manually computes the R² score by calculating the residual sum of squares and the total sum of squares.
  
  4. **t_squared_test_manual(X, y, y_pred)**  
     - Performs a manual t-squared test on the regression model.
     - Computes the standard error of the slope and returns the t-squared value.

## Libraries Used

- **Pandas**: Reading CSV files, data manipulation, and creating DataFrames.
- **Matplotlib**: Data visualization (plotting histograms and regression graphs).
- **Scikit-learn**: Building and evaluating the linear regression model.
- **NumPy**: Performing numerical calculations.
- **Scipy.stats**: Fitting the Weibull distribution and calculating probability density functions.

## Program Flow

1. **Data Loading**:
   - The program begins by loading wind speed data from six CSV files.
   - Data is concatenated into a single DataFrame and a `Datetime` column is created.
   
2. **User Interaction**:
   - A command-line menu is presented with the following options:
     - **Histogram**: View a histogram of daily mean wind speeds with an overlaid Weibull distribution.
     - **Regression Model**: Build a linear regression model and predict future wind speeds.
     - **Library List**: Display the libraries used in the project.
     - **Exit**: Quit the program.
     
3. **Execution**:
   - Based on the user’s selection, appropriate functions from `methods.py` are invoked to perform the desired analysis or visualization.

## Getting Started

### Prerequisites

Make sure you have the following Python libraries installed:

- Pandas
- Matplotlib
- Scikit-learn
- NumPy
- Scipy

You can install these dependencies using pip:

```bash
pip install pandas matplotlib scikit-learn numpy scipy
