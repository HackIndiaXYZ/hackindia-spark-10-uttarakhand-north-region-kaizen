# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:19:25 2022
@author: USER

Demmin	Original Data
Sheet1	Total 63 samples with different measurments
Sheet2	62 samples with all 2151 wavelengths
Sheet3	62 samples with all 2151 wavelengths with SOC and groupings for classification
Sheet4	62 samples with all 2151 wavelengths with grouping only for classification
Sheet5	62 samples with 401 bands; rangining from 350 - 750 nm for grouping classification
Sheet6	62 samples with 401 bnads; ranging from 350 - 750 nm for SOC Regdression
Sheet7	57 samples (5 samples are removed based on very high organic content) with 401 bands; ranging from 350 - 750 nm for SOC regression
Sheet8	62 samples with 1488 bands; ranging from 400 - 2349 nm for SOC regression
Sheet9	57 samples with all 2151 bands; rangining from 350 - 2500 nm for SOC regression

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

filename = r'\\172.20.125.61\mtech\Major Jarmal Singh\SOC Project\Demmin_data.xlsx'
data = pd.read_excel(filename, sheet_name='Sheet9', engine='openpyxl')

column_names_feature = data.columns.difference(['Org_Car_g_per_kg'])
x = data[column_names_feature[100:2050]].values
y = data['Org_Car_g_per_kg'].values

# All Bands plotting
with plt.style.context('ggplot'):
    plt.plot(column_names_feature[100:2050], x.T)

# create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

# fit the regressor with x and y data
regressor.fit(x, y)

type(x)
test_data = x[56]
type(test_data)
test_data.shape

Y_pred = regressor.predict(test_data.reshape(1, -1)) # test the output by changing values

### Visualising the Random Forest Regression results ###
# arrange for creating a range of values from min value of x to max
# value of x with a difference of 0.01 between two consecutive values
X_grid = np.arange(np.min(x), np.max(x), 0.01)

# reshape for reshaping the data into a len(X_grid)*1 array, i.e. to make a column out of the X_grid value				
X_grid = X_grid.reshape((len(X_grid), 1))

# Scatter plot for original data
plt.scatter(x, column_names_feature, color = 'blue')

# plot predicted data
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()











