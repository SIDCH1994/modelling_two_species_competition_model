# pip install pandas 
# pip install matplotlib 
# pip install numpy
# pip install scikit-learn
# pip install statsmodels
# pip install torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from statsmodels.tsa.api import VAR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from two_species_differential_equation import mixture_data, result_global, lotka_volterra, solve_ivp


# ----------------------------------------------------------------------------------------
# PART-2: Machine Learning Modelling
# ----------------------------------------------------------------------------------------
# Phase-1: Multi- output Linear Regression
# Phase-2: Ridge Regression
# Phase-3: Polynomial Regression (degree ≤ 2)
# Phase-4: Gaussian Process Regression (GPR)
# Phase-5: Vector AutoRegressive Model (VAR)
# Phase-6: kNN Regressor (Multi-output)
# Phase-7: Physics-Informed Neural Network (PINN)
# ----------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------
# Phase-1: Multi- output Linear Regression
# --------------------------------------------------------------------------------
# Step-1: Importing the dataset
# Step-2: Defining the RMSE and MAE functions
# Step-3: Train multi-output linear regression
# Step-4: Model Evaluation
# Step-5: Compare with IVP model visually vs IVP model
# ---------------------------------------------------------------------------------


# Step-1: Importing the dataset
# ------------------------------------------------------------------
data = pd.read_excel('gauseR.xlsx', sheet_name='Sheet1')
# print(data.head())
print("\n\n")
mixture_data = data[data['Treatment'] == 'Mixture']
# print(mixture_data.head())
X = mixture_data['Day'].values.reshape(-1, 1)  # time as input
Y = mixture_data[['Volume_Species1', 'Volume_Species2']].values  # target: both species


# Step-2: Defining the RMSE and MAE functions
# ------------------------------------------------------------------
def compute_rmse_ml(y_true, y_pred):
    rmse_x = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    return rmse_x, rmse_y

def compute_mae_ml(y_true, y_pred):
    mae_x = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    return mae_x, mae_y


# Step 3: Train multi-output linear regression
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)

# Step 4: Model Evaluation
# ---------------------------------------------------------
rmse_species1 = np.sqrt(mean_squared_error(Y[:, 0], Y_pred[:, 0]))
rmse_species2 = np.sqrt(mean_squared_error(Y[:, 1], Y_pred[:, 1]))
mae_species1 = mean_absolute_error(Y[:, 0], Y_pred[:, 0])
mae_species2 = mean_absolute_error(Y[:, 1], Y_pred[:, 1])

print(f"RMSE (Species 1): {rmse_species1:.2f}")
print(f"RMSE (Species 2): {rmse_species2:.2f}")
print(f"MAE (Species 1): {mae_species1:.2f}")
print(f"MAE (Species 2): {mae_species2:.2f}")

# Step 5: Compare with IVP model visually
# ---------------------------------------------------------
t_data = mixture_data['Day'].values
x0 = Y[0]

ivp_solution = solve_ivp(lotka_volterra, (t_data[0], t_data[-1]), x0, args=tuple(result_global.x), t_eval=t_data)
ivp_species1, ivp_species2 = ivp_solution.y

plt.figure(figsize=(10, 6))
plt.plot(t_data, Y[:, 0], 'o', label='Observed Species 1')
plt.plot(t_data, Y[:, 1], 's', label='Observed Species 2')
plt.plot(t_data, Y_pred[:, 0], '--', label='Linear Regression Species 1')
plt.plot(t_data, Y_pred[:, 1], '--', label='Linear Regression Species 2')
plt.plot(t_data, ivp_species1, ':', label='LV solve_ivp Species 1')
plt.plot(t_data, ivp_species2, ':', label='LV solve_ivp Species 2')

plt.title("Multi-Output Linear Regression vs LV Model")
plt.xlabel("Day")
plt.ylabel("Population Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("linear_regression_vs_lv.png")


# ---------------------------------------------------------------------------------
# Phase-1 CONCLUDED
# ---------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# Phase-2: Ridge Regression
# --------------------------------------------------------------------------------
# Step-1: Define model with regularization strength alpha
# Step-2: Model fitting for the ridge regression
# Step-3: Evaluate the model using RMSE and MAE
# Step-4: Compare with IVP model visually vs IVP model
# --------------------------------------------------------------------------------


# Step-1: Define model with regularization strength alpha
# ------------------------------------------------------------------
ridge_model = Ridge(alpha=0.4)  # you can later tune alpha


# Step-2: Model fitting for the ridge regression
# ------------------------------------------------------------------
ridge_model.fit(X, Y)
Y_ridge_pred = ridge_model.predict(X)


# Step-3: Evaluate the model using RMSE and MAE
# ------------------------------------------------------------------
rmse_ridge_x, rmse_ridge_y = compute_rmse_ml(Y, Y_ridge_pred)
mae_ridge_x, mae_ridge_y = compute_mae_ml(Y, Y_ridge_pred)

print(f"Ridge RMSE (Species 1): {rmse_ridge_x:.2f}")
print(f"Ridge RMSE (Species 2): {rmse_ridge_y:.2f}")
print(f"Ridge MAE (Species 1): {mae_ridge_x:.2f}")
print(f"Ridge MAE (Species 2): {mae_ridge_y:.2f}")


# Step-4: Compare with IVP model visually
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_data, Y[:, 0], 'o', label='Observed Species 1')
plt.plot(t_data, Y[:, 1], 's', label='Observed Species 2')
plt.plot(t_data, Y_ridge_pred[:, 0], '--', label='Ridge Regression Species 1')
plt.plot(t_data, Y_ridge_pred[:, 1], '--', label='Ridge Regression Species 2')
plt.plot(t_data, ivp_species1, ':', label='LV solve_ivp Species 1')
plt.plot(t_data, ivp_species2, ':', label='LV solve_ivp Species 2')

plt.title("Ridge Regression vs LV Model")
plt.xlabel("Day")
plt.ylabel("Population Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ridge_regression_vs_lv.png")


# --------------------------------------------------------------------------------
# Phase-2 CONCLUDED
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Phase-3: Polynomial Regression (degree ≤ 2)
# --------------------------------------------------------------------------------
# Step-1: Define pipeline: polynomial features + linear regression
# Step-2: Fit model for polynomial regression
# Step-3: Evaluate the model using RMSE and MAE
# Step-4: Plotting the polynomial regression results vs IVP model
# ---------------------------------------------------------------------------------


# Step-1: Define pipeline: polynomial features + linear regression
# ------------------------------------------------------------------
poly_degree = 3
poly_model = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())

# Step-2: Fit model for polynomial regression
# ------------------------------------------------------------------
poly_model.fit(X, Y)
Y_poly_pred = poly_model.predict(X)

# Step-3: Evaluate the model using RMSE and MAE
# ------------------------------------------------------------------
rmse_poly_x, rmse_poly_y = compute_rmse_ml(Y, Y_poly_pred)
mae_poly_x, mae_poly_y = compute_mae_ml(Y, Y_poly_pred)

print(f"Polynomial RMSE (Species 1): {rmse_poly_x:.2f}")
print(f"Polynomial RMSE (Species 2): {rmse_poly_y:.2f}")
print(f"Polynomial MAE (Species 1): {mae_poly_x:.2f}")
print(f"Polynomial MAE (Species 2): {mae_poly_y:.2f}")

# Step-4: Plotting the polynomial regression results
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_data, Y[:, 0], 'o', label='Observed Species 1')
plt.plot(t_data, Y[:, 1], 's', label='Observed Species 2')
plt.plot(t_data, Y_poly_pred[:, 0], '--', label='Polynomial Regression Species 1')
plt.plot(t_data, Y_poly_pred[:, 1], '--', label='Polynomial Regression Species 2')
plt.plot(t_data, ivp_species1, ':', label='LV solve_ivp Species 1')
plt.plot(t_data, ivp_species2, ':', label='LV solve_ivp Species 2')

plt.title(f"Polynomial Regression (Degree {poly_degree}) vs LV Model")
plt.xlabel("Day")
plt.ylabel("Population Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("polynomial_regression_vs_lv.png")


# ---------------------------------------------------------------------------------
# Phase-3 CONCLUDED
# ---------------------------------------------------------------------------------



# --------------------------------------------------------------------------------
# Phase-4: Gaussian Process Regression (GPR)
# --------------------------------------------------------------------------------
# Step-1: Define the GPR model
# Step-2: Fit the GPR model
# Step-3: Evaluate the GPR model using RMSE and MAE
# Step-4: Plotting the GPR results vs IVP model
# --------------------------------------------------------------------------------


# Step-1: Define the GPR model
# ------------------------------------------------------------------
# Phase-4: Gaussian Process Regression (GPR)
# ---------------------------------------------------------
# Step-1: Define kernel (constant * RBF)
# Step-2: Define and fit separate GPR models
# Step-3: Predict using the GPR models
# Step-4: Evaluate the GPR model using RMSE and MAE
# Step-5: Plotting the GPR results vs IVP model
# ---------------------------------------------------------------------------------


# Step-1: Define kernel (constant * RBF)
# ------------------------------------------------------------------
#kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e2))
# kernel = DotProduct() + WhiteKernel(noise_level=5.0)
# kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=5.0, length_scale_bounds=(1, 50)) + WhiteKernel(noise_level=5.0)
kernel = C(10.0, (1e-2, 1e3)) * RBF(length_scale=3.0, length_scale_bounds=(0.1, 20.0)) + WhiteKernel(noise_level=2.0, noise_level_bounds=(1e-1, 50))

# Step-2: Define and fit separate GPR models
# ------------------------------------------------------------------
gpr_species1 = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=20, normalize_y=True)
gpr_species2 = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=20, normalize_y=True)

gpr_species1.fit(X, Y[:, 0])
gpr_species2.fit(X, Y[:, 1])


# Step-3: Predict using the GPR models
# ------------------------------------------------------------------
y1_pred, y1_std = gpr_species1.predict(X, return_std=True)
y2_pred, y2_std = gpr_species2.predict(X, return_std=True)
Y_gpr_pred = np.column_stack([y1_pred, y2_pred])


# Step-4: Evaluate the GPR model using RMSE and MAE
# ------------------------------------------------------------------
rmse_gpr_x, rmse_gpr_y = compute_rmse_ml(Y, Y_gpr_pred)
mae_gpr_x, mae_gpr_y = compute_mae_ml(Y, Y_gpr_pred)

print(f"GPR RMSE (Species 1): {rmse_gpr_x:.2f}")
print(f"GPR RMSE (Species 2): {rmse_gpr_y:.2f}")
print(f"GPR MAE (Species 1): {mae_gpr_x:.2f}")
print(f"GPR MAE (Species 2): {mae_gpr_y:.2f}")


# Step-5: Plotting the GPR results vs IVP model
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_data, Y[:, 0], 'o', label='Observed Species 1')
plt.plot(t_data, Y[:, 1], 's', label='Observed Species 2')
plt.plot(t_data, y1_pred, '--', label='GPR Species 1')
plt.plot(t_data, y2_pred, '--', label='GPR Species 2')
plt.fill_between(t_data.flatten(), y1_pred - y1_std, y1_pred + y1_std, alpha=0.2, label='GPR ±1σ Species 1')
plt.fill_between(t_data.flatten(), y2_pred - y2_std, y2_pred + y2_std, alpha=0.2, label='GPR ±1σ Species 2')
plt.plot(t_data, ivp_species1, ':', label='LV solve_ivp Species 1')
plt.plot(t_data, ivp_species2, ':', label='LV solve_ivp Species 2')

plt.title("Gaussian Process Regression vs LV Model")
plt.xlabel("Day")
plt.ylabel("Population Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gpr_vs_lv.png")

# ---------------------------------------------------------------------------------
# Phase-4 CONCLUDED
# ---------------------------------------------------------------------------------



# --------------------------------------------------------------------------------
# Phase-5: Vector AutoRegressive Model (VAR)
# --------------------------------------------------------------------------------
# Step-1: Prepare data for VAR
# Step-2: Fit VAR model
# Step-3: Forecast over original time horizon
# Step-4: Evaluation using RMSE and MAE
# Step-5: Plot comparison with IVP model
# ---------------------------------------------------------------------------------


# Step-1: Prepare data for VAR
# ------------------------------------------------------------------
var_data = mixture_data[['Volume_Species1', 'Volume_Species2']]
var_data.index = mixture_data['Day']


# Step-2: Fit VAR model
# ------------------------------------------------------------------
var_model = VAR(var_data)
var_result = var_model.fit(4)


# Step-3: Forecast over original time horizon
# ------------------------------------------------------------------
forecast = var_result.forecast(var_data.values, steps=len(var_data))
forecast_df = pd.DataFrame(forecast, columns=['Species1_Pred', 'Species2_Pred'])
forecast_df.index = var_data.index


# Step-4: Evaluation using RMSE and MAE
# ------------------------------------------------------------------
Y_true = var_data.values
Y_pred = forecast_df.values

rmse_var_x, rmse_var_y = compute_rmse_ml(Y_true, Y_pred)
mae_var_x, mae_var_y = compute_mae_ml(Y_true, Y_pred)

print(f"VAR RMSE (Species 1): {rmse_var_x:.2f}")
print(f"VAR RMSE (Species 2): {rmse_var_y:.2f}")
print(f"VAR MAE (Species 1): {mae_var_x:.2f}")
print(f"VAR MAE (Species 2): {mae_var_y:.2f}")


# Step-5: Plot comparison with IVP model
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(var_data.index, Y_true[:, 0], 'o', label='Observed Species 1')
plt.plot(var_data.index, Y_true[:, 1], 's', label='Observed Species 2')
plt.plot(var_data.index, Y_pred[:, 0], '--', label='VAR Species 1')
plt.plot(var_data.index, Y_pred[:, 1], '--', label='VAR Species 2')
plt.plot(t_data, ivp_species1, ':', label='LV solve_ivp Species 1')
plt.plot(t_data, ivp_species2, ':', label='LV solve_ivp Species 2')

plt.title("VAR Forecast vs LV Model")
plt.xlabel("Day")
plt.ylabel("Population Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("var_vs_lv.png")
plt.show()

# ---------------------------------------------------------------------------------
# Phase-5 CONCLUDED
# ---------------------------------------------------------------------------------



# --------------------------------------------------------------------------------
# Phase-6: kNN Regressor (Multi-output)
# --------------------------------------------------------------------------------
# Step-1: Prepare data
# Step-2: Fit multi-output kNN model
# Step-3: Evaluate predictions
# Step-4: Plot vs LV model
# ---------------------------------------------------------------------------------


# Step-1: Prepare data
# ------------------------------------------------------------------
X = mixture_data['Day'].values.reshape(-1, 1)
Y = mixture_data[['Volume_Species1', 'Volume_Species2']].values


# Step-2: Fit multi-output kNN model
# ------------------------------------------------------------------
knn_base = KNeighborsRegressor(n_neighbors=2)  # try k=3, tweakable
knn_model = MultiOutputRegressor(knn_base)
knn_model.fit(X, Y)
Y_knn_pred = knn_model.predict(X)


# Step-3: Evaluate predictions
# ------------------------------------------------------------------
rmse_knn_x, rmse_knn_y = compute_rmse_ml(Y, Y_knn_pred)
mae_knn_x, mae_knn_y = compute_mae_ml(Y, Y_knn_pred)

print(f"kNN RMSE (Species 1): {rmse_knn_x:.2f}")
print(f"kNN RMSE (Species 2): {rmse_knn_y:.2f}")
print(f"kNN MAE (Species 1): {mae_knn_x:.2f}")
print(f"kNN MAE (Species 2): {mae_knn_y:.2f}")


# Step-4: Plot vs LV model
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(X, Y[:, 0], 'o', label='Observed Species 1')
plt.plot(X, Y[:, 1], 's', label='Observed Species 2')
plt.plot(X, Y_knn_pred[:, 0], '--', label='kNN Species 1')
plt.plot(X, Y_knn_pred[:, 1], '--', label='kNN Species 2')
plt.plot(t_data, ivp_species1, ':', label='LV solve_ivp Species 1')
plt.plot(t_data, ivp_species2, ':', label='LV solve_ivp Species 2')

plt.title("k-Nearest Neighbors vs LV Model")
plt.xlabel("Day")
plt.ylabel("Population Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_vs_lv.png")
plt.show()

# ---------------------------------------------------------------------------------
# Phase-6 CONCLUDED
# ---------------------------------------------------------------------------------



# --------------------------------------------------------------------------------
# Phase-7: Physics-Informed Neural Network (PINN)
# --------------------------------------------------------------------------------












