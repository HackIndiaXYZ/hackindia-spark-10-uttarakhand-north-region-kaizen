# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:31:38 2022

@author: Manoj Kaushik
"""

# saving the ML Regression model for further use by external loading
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

# loading data
filename = r'\\172.20.125.61\mtech\Major Jarmal Singh\Demmin_data.xlsx'
data = pd.read_excel(filename, sheet_name='Sheet14', engine='openpyxl')

matirial = 'Org_Car_g_per_kg' # Clay
column_names_feature = data.columns.difference([matirial])

x = data[column_names_feature].values
y = data[matirial].values

# making of model
pls_model = PLSRegression(n_components = 8)
pls_model.fit(x, y)
y_pred = pls_model.predict(x[0].reshape(1, -1))

# where to save the loaded model
model_path = r"\\172.20.125.61\mtech\Major Jarmal Singh\Data_Demmin_2015_10_01\Saved_model\PLS_model.sav"
pickle.dump(pls_model, open(model_path, 'wb'))

# loading model for testing again
loaded_model = pickle.load(open(model_path, 'rb'))
result = loaded_model.score(x, y)
print(result) # 0.8352299966970731


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


























