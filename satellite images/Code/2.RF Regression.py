# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 21:43:51 2022
@author: Manoj Kaushik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

filename = r'\\172.20.125.61\mtech\Major Jarmal Singh\Demmin_data.xlsx'
data = pd.read_excel(filename, sheet_name='Sheet14', engine='openpyxl')

matirial = 'Org_Car_g_per_kg' # Org_Car_g_per_kg, Clay
column_names_feature = data.columns.difference([matirial])

x = data[column_names_feature].values
y = data[matirial].values

# Set font family
plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'Times New Roman'

# Set DPI
plt.rcParams['figure.dpi'] = 150


# # All Bands plotting
# plt.figure(figsize=(10, 8))
# with plt.style.context('bmh'):
#     plt.plot(column_names_feature, x.T)
#     plt.margins(x=0)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance')


def cv_regressor(X, y):
    regressor = RandomForestRegressor(bootstrap=True, max_depth=None, max_features='sqrt', 
                                      min_samples_leaf=4, min_samples_split=10, n_estimators=100)
    y_cv = cross_val_predict(regressor, X, y, cv=10)
    
    print("y:", y)
    print("y_cv", y_cv)
    
    # --- Basic metrics ---
    r2 = r2_score(y, y_cv)
    rmse = np.sqrt(mean_squared_error(y, y_cv))
    rpd = y.std()/rmse
    ME = np.mean(y_cv - y)
    
    # --- Concordance Correlation Coefficient (CCC) ---
    r, _ = pearsonr(y, y_cv)
    mean_true = np.mean(y)
    mean_pred = np.mean(y_cv)
    sigma_pred = np.std(y_cv)
    sigma_obs = np.std(y)
    CCC = (2 * r * sigma_obs * sigma_pred) / (sigma_obs**2 + sigma_pred**2 + (mean_true - mean_pred)**2)
    
    return (y_cv, r2, rmse, rpd, ME, CCC)


y_cv, r2, rmse, rpd, ME, CCC = cv_regressor(x, y)
print('R2: %0.4f, RMSE: %0.4f, RPD: %0.4f, ME: %0.4f, CCC: %0.4f' %(r2, rmse, rpd, ME, CCC))
# before grid search: R2: 0.2276, MSE: 8.8859, RPD: 1.1378
# after grid search : R2: 0.2362, RMSE: 2.9641, RPD: 1.1443, ME: -0.0987, CCC: 0.4795


plt.figure(figsize=(8, 8))
with plt.style.context('bmh'):
    plt.scatter(y, y_cv, color='red', label='Estimated SOC(g/Kg)')
    # plt.plot(y, y, '-g', label='Expected regression line')
    
    # Add text with a bounding box
    x_pos = 8.5  # Adjust position based on data
    y_pos = 20  # Adjust position based on data
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    # plt.text(x_pos, y_pos, r'${R^2}$ = 0.71' + '\nMSE = 3.30\nRPD = 1.86', ha='center', va='bottom', bbox=bbox)
    plt.text(x_pos, y_pos, f'${{R^2}}$ = {r2:.2f}\nMSE = {rmse:.2f}\nRPD = {rpd:.2f}', ha='center', va='bottom', bbox=bbox)
    
    # Plot regression line
    z = np.polyfit(y, y_cv, 1)
    plt.plot(np.polyval(z, y), y, color='blue', linestyle=':', label='Predicted regression line') # 
    
    # Add labels
    plt.xlabel('Measured SOC (g/Kg)', color='royalblue')
    plt.ylabel('Estimated SOC (g/Kg)', color='royalblue')
    
    # Customize legend box
    plt.legend(
        loc='lower right',           # Position of the legend
        # bbox_to_anchor=(1, 1),     # Move legend outside the plot (adjust as needed)
        frameon=True,              # Show legend box
        edgecolor='black',         # Border color of the legend box
        facecolor='lightgray',     # Background color of the legend box
        shadow=True,               # Add shadow to the legend box
        # fontsize='medium',        # Font size of the legend text
        # title='Legend',            # Add a title to the legend
        # title_fontsize='large'     # Font size of the legend title
    )
    plt.plot()


###############-------------------####################

# Experimental results for Sheet15  (Chilka AVIRIS_NG Data)  for SOC
# without wavelet SVR: R2: 0.1875, MSE: 9.3464, RPD: 1.1094
# with wavelet    SVR: R2: 0.1812, MSE: 9.4190, RPD: 1.1051

# without wavelet RF: R2: 0.2143, MSE: 9.0386, RPD: 1.1282
# with wavelet    RF: R2: 0.2251, MSE: 8.9141, RPD: 1.1360


# Experimental results for Sheet18  (PRISM Data)  for SOC
# without wavelet SVR: R2: 0.1743, MSE: 9.4991, RPD: 1.1005
# with wavelet    SVR: R2: 0.1754, MSE: 9.4856, RPD: 1.1013

# without wavelet RF: R2: 0.2228, MSE: 8.9403, RPD: 1.1344
# with wavelet    RF: R2: 0.2427, MSE: 8.7116, RPD: 1.1491


# Grid Search for RFR | Deepseek
# Import required libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
    'bootstrap': [True, False]  # Whether bootstrap samples are used
}

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, scoring='neg_mean_squared_error', 
                           n_jobs=-1, verbose=2)

# Fit the grid search
grid_search.fit(x, y)

# Evaluate the best model
print("Best parameters found: ", grid_search.best_params_)

best_model = grid_search.best_estimator_




























