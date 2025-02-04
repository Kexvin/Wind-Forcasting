# Copyright 2024 Kevin Do
import pandas as pd
import methods


print("Hello, Welcome to the Data Visualization Program!")
data_frame2007 = pd.read_csv('1243941_33.04_-96.86_2007.csv', skiprows=1, header=0)
data_frame2008 = pd.read_csv('1243941_33.04_-96.86_2008.csv', skiprows=1, header=0)
data_frame2009 = pd.read_csv('1243941_33.04_-96.86_2009.csv', skiprows=1, header=0)
data_frame2010 = pd.read_csv('1243941_33.04_-96.86_2010.csv', skiprows=1, header=0)
data_frame2011 = pd.read_csv('1243941_33.04_-96.86_2011.csv', skiprows=1, header=0)
data_frame2012 = pd.read_csv('1243941_33.04_-96.86_2012.csv', skiprows=1, header=0)


# 1st issue encountered - needing to skip row for data to read correctly
combined_df = pd.concat([data_frame2007, data_frame2008, data_frame2009,
                         data_frame2010, data_frame2011, data_frame2012])

combined_df['Datetime'] = pd.to_datetime(combined_df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

choice = True
while choice:
    print("Which option would you like to choose? \n 1. View Histogram \n 2. View Regression Model \n 3. Libraries used \n 4. Exit Program")
    decision = input()
    if decision == '1':
        print("Loading...")
        # Resample the data to daily means
        daily_wind_speed = combined_df.resample('D', on='Datetime').mean()
        # Extract the wind speed data
        wind_speeds = daily_wind_speed['wind speed at 100m (m/s)'].dropna()
        methods.plot_histogram(wind_speeds)

    elif decision == '2':
        print("Loading...")
        # Extracts the year and month from datetime to create new columns
        combined_df['Year'] = combined_df['Datetime'].dt.year
        combined_df['Month'] = combined_df['Datetime'].dt.month

        # Creates a new data frame with year and month as columns removes date time
        monthly_wind_speed = combined_df.groupby(['Year', 'Month'])['wind speed at 100m (m/s)'].mean().reset_index()
        X = monthly_wind_speed[['Year']].values  # Use only Year as the independent variable
        y = monthly_wind_speed['wind speed at 100m (m/s)'].values  # Wind Speed as dependent variable

        # Call the function to fit the model and plot the results
        methods.fit_and_plot_regression(X, y)

    elif decision == '3':
        print("--Libraries--")
        print(' 1. Pandas \n 2. MathPlotLib \n 3. Sklearn \n 4. Numpy \n 5. Scipy.stats \n')

    elif decision == '4':
        print("Thanks for using my program")
        print("Exiting....")
        choice = False