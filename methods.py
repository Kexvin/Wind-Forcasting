# Copyright 2024 Kevin Do
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import weibull_min


def plot_histogram(wind_speeds):
    plt.figure(figsize=(5, 3))
    # Plot histogram of wind speed data with frequency
    # Purpose is to store raw bins and counts which are the frequencies inside the bins.
    counts, bins, _ = plt.hist(wind_speeds, bins=30, color='blue', alpha=0.7, label='Histogram')

    # Fit a Weibull distribution to the data
    # Floc ensures data is anchored to 0
    shape, loc, scale = weibull_min.fit(wind_speeds, floc=0)

    # Generate Weibull PDF values for plotting
    x = np.linspace(wind_speeds.min(), wind_speeds.max(), 100)
    # Calculates the probability density function at each value of x in curve
    weibull_pdf = weibull_min.pdf(x, shape, loc, scale)

    # Scale the Weibull PDF to match the histogram scale
    bin_width = bins[1] - bins[0]
    scaled_weibull_pdf = weibull_pdf * len(wind_speeds) * bin_width  # Scale by total data points and bin width

    # Plot the Weibull distribution curve
    plt.plot(x, scaled_weibull_pdf, 'r-', lw=2, label=f'Weibull PDF\nshape={shape:.1f}, scale={scale:.1f}')

    # Add titles and labels
    plt.title('Daily Mean Wind Speed at 100m with Weibull Distribution')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Frequency')
    plt.legend(fontsize='small', loc='upper right', bbox_to_anchor=(1, 1), frameon=True, framealpha=0.7)

    # Prevent cutting off labels
    plt.tight_layout()

    # Show the plot
    plt.show()


def fit_and_plot_regression(X, y):
    print("How many years would you like to Predict? (2-15 Years) ")
    years = int(input("Enter the number of years: "))

    if years > 15 or (years <= 1):
        print("Your prediction is too far out\nExiting the method...\n")
        return  # Exits the function

    # Defines the model being used.
    model = LinearRegression()
    model.fit(X, y)

    # Forecast 10 years into the future
    future_years = np.arange(X.max() + 1, X.max() + years).reshape(-1, 1)

    # Predict future wind speeds
    future_predictions = model.predict(future_years)

    # Plot the data (scatter points for monthly wind speeds and linear regression line)
    plt.figure(figsize=(10, 6))

    # Plot scatter points for the historical monthly wind speeds (X as Year, y as wind speed)
    plt.scatter(X, y, color='blue', alpha=0.6, label='Historical Wind Speed (Monthly)')

    # Plot the linear regression line for both historical data and future predictions
    plt.plot(np.concatenate([X.flatten(), future_years.flatten()]),
             np.concatenate([model.predict(X), future_predictions]),
             color='red', label='Linear Regression (Trend)', linestyle='-')

    # Add titles and labels
    plt.title('Wind Speed Forecast (m/s)')
    plt.xlabel('Year')
    plt.ylabel('Wind Speed (m/s)')

    # Set the x-axis to show only full years
    # Predict the wind speeds for the historical data
    plt.xticks(np.arange(X.min(), future_years.max() + 1, 1))
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    r2_manual = r2_manualCalc(y, y_pred)

    # Calculate t-squared value manually
    t_squared, t_stat, SE = t_squared_test_manual(X, y, y_pred)
    print(f"Mean Squared Error (MSE) for historical data: {mse}")
    print(f"R² score for historical data - Sklearn: {r2}")
    print(f"R² score for historical data - Manual Calculation: {r2_manual}")
    print(f"T-statistic: {t_stat}")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def r2_manualCalc(y_true, y_pred):
    # Calculate the residual sum of squares (SS_res)
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Calculate the total sum of squares (SS_tot)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_manual = 1 - (ss_res / ss_tot)
    return r2_manual

def t_squared_test_manual(X, y, y_pred):
    n = len(X)
    # Calculates the residual sum of squares (RSS) & total sum of squares (tss)
    rss = np.sum((y - y_pred) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)

    # Calculates the variance of the residuals (RSS / degrees of freedom)
    residual_var = rss / (n - 2)

    # Standard error of the slope (SE)
    X_mean = np.mean(X)
    SE = np.sqrt(residual_var / np.sum((X - X_mean) ** 2))

    # Get the slope coefficient from the regression model
    beta_hat = np.polyfit(X.flatten(), y, 1)[0]

    # t-statistic for the slope
    t_stat = beta_hat / SE
    t_squared = t_stat ** 2

    return t_squared, t_stat, SE