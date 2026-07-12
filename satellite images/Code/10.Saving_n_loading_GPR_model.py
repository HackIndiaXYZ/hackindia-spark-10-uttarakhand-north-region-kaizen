# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:36:07 2025

@author: Manoj Kaushik
"""

# saving the ML Regression model for further use by external loading
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

# loading data
filename = r'\\172.20.125.61\mtech\Major Jarmal Singh\Demmin_data.xlsx'
data = pd.read_excel(filename, sheet_name='Sheet14', engine='openpyxl')

column_names_feature = data.columns.difference(['Org_Car_g_per_kg'])

x = data[column_names_feature].values
y = data['Org_Car_g_per_kg'].values

# making of model
# Define the kernel (RBF kernel)
kernel = 1.0 * RBF(length_scale=1.0)
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
model.fit(x, y)
y_pred = model.predict(x[0].reshape(1, -1))
    
model_path = r"\\172.20.125.61\mtech\Major Jarmal Singh\Data_Demmin_2015_10_01\Saved_model\GPR_model.sav"
pickle.dump(model, open(model_path, 'wb'))

# loading model for testing again
loaded_model = pickle.load(open(model_path, 'rb'))
result = loaded_model.score(x, y)
print(result) # 1.0


# to plot the graph between actual data points and predicted line:
y_pred = []
for i in x:
    pred = loaded_model.predict(i.reshape(1, -1))
    y_pred.append(pred[0])

y_pred = np.asarray(y_pred)


plt.figure(figsize=(8, 8))
with plt.style.context('bmh'):
    plt.scatter(y, y_pred, color='red')
    plt.plot(y, y, '-g', label='Expected regression line')
    z = np.polyfit(y, y_pred, 1)
    plt.plot(np.polyval(z, y), y, color='blue', label='Predicted regression line')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.plot()

