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
from sklearn.linear_model import ElasticNet
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
    regressor = ElasticNet(alpha=0.1, l1_ratio=0.1)
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
# before grid search: R2: 0.3298, MSE: 7.7102, RPD: 1.2215
# after grid search : R2: 0.3276, RMSE: 2.7811, RPD: 1.2195, ME: -0.0655, CCC: 0.4757


plt.figure(figsize=(8, 8))
with plt.style.context('bmh'):
    plt.scatter(y, y_cv, color='red', label='Estimated SOC(g/Kg)')
    # plt.plot(y, y, '-g', label='Expected regression line')
    
    # Add text with a bounding box
    x_pos = 8.5  # Adjust position based on data
    y_pos = 20  # Adjust position based on data
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    # plt.text(x_pos, y_pos, r'${R^2}$ = 0.71' + '\nMSE = 3.30\nRPD = 1.86', ha='center', va='bottom', bbox=bbox)
    plt.text(x_pos, y_pos, f'${{R^2}}$ = {r2:.2f}\nRMSE = {rmse:.2f}\nRPD = {rpd:.2f}', ha='center', va='bottom', bbox=bbox)
    
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
# grid search hyperparameters for the elastic net | DeepSeek
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression


# Define the ElasticNet model
elastic_net = ElasticNet()

# Define the hyperparameter grid
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0],  # Regularization strength
    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]  # Mixing parameter
}

# Set up the grid search
grid_search = GridSearchCV(
    estimator=elastic_net,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # Use negative MSE for scoring
    cv=5,  # 5-fold cross-validation
    verbose=1,
    n_jobs=-1  # Use all available CPU cores
)

# Perform the grid search
grid_search.fit(x, y)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Best Hyperparameters: {alpha=0.1, l1_ratio=0.1}

















