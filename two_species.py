# pip install pandas 
# pip install matplotlib 
# pip install openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution, Bounds

# Loading the data from the Excel file
data = pd.read_excel('gauseR.xlsx', sheet_name='Sheet1')
# print(data.head())
print("\n\n")


# Filtering for "Mixture" treatment
mixture_data = data[data['Treatment'] == 'Mixture']
# print(mixture_data.head())


# plotting the Mixture data

plt.figure(figsize=(10, 6))
plt.plot(mixture_data['Day'], mixture_data['Volume_Species1'], label='Paramecium caudatum', marker='o')
plt.plot(mixture_data['Day'], mixture_data['Volume_Species2'], label='Paramecium aurelia', marker='s')

plt.title("Paramecium Competition Over Time (Gause Experiment)")
plt.xlabel("Day")
plt.ylabel("Population Volume")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("paramecium_population_line_graph.png")
plt.show()


# The Lotka-Volterra equations for two species with interspecific competition are given by:

def lotka_volterra(t, z, r1, r2, K1, K2, alpha, beta):
    x, y = z
    dxdt = r1 * x * (1 - (x + alpha * y) / K1)
    dydt = r2 * y * (1 - (y + beta * x) / K2)
    return [dxdt, dydt]


# Define Loss Function

def loss_function(params, t_data, x_data, y_data):
    r1, r2, K1, K2, alpha, beta = params
    z0 = [x_data[0], y_data[0]]  # initial populations
    sol = solve_ivp(lotka_volterra, (t_data[0], t_data[-1]), z0,
                    args=(r1, r2, K1, K2, alpha, beta), t_eval=t_data)
    if not sol.success:
        return np.inf
    x_pred, y_pred = sol.y
    return np.sum((x_data - x_pred)**2 + (y_data - y_pred)**2)


# Preparing Data for Optimization

t_data = mixture_data['Day'].values
x_data = mixture_data['Volume_Species1'].values
y_data = mixture_data['Volume_Species2'].values


# Initial Parameter Guess and Bounds

initial_guess = [0.5, 0.3, 100, 100, 0.5, 0.5]
bounds = Bounds([0.01, 0.01, 10, 10, 0.01, 0.01], [5, 5, 500, 500, 5, 5])


# Step 1: Minimize with L-BFGS-B (Local Optimizer)

result_local = minimize(loss_function, initial_guess,
                        args=(t_data, x_data, y_data),
                        method='L-BFGS-B', bounds=bounds)

print("\nOptimized Parameters (L-BFGS-B with bounds):")
print("r1 = {:.3f}, r2 = {:.3f}, K1 = {:.2f}, K2 = {:.2f}, alpha = {:.2f}, beta = {:.2f}".format(*result_local.x))


# Step 2: Refine with Global Optimization (Differential Evolution)

de_bounds = [(0.01, 5), (0.01, 5), (10, 500), (10, 500), (0.01, 5), (0.01, 5)]
result_global = differential_evolution(loss_function, de_bounds, args=(t_data, x_data, y_data), strategy='best1bin')

print("\nOptimized Parameters (Differential Evolution):")
print("r1 = {:.3f}, r2 = {:.3f}, K1 = {:.2f}, K2 = {:.2f}, alpha = {:.2f}, beta = {:.2f}".format(*result_global.x))


# Step 3: Plot Model Fits vs Real Data

def simulate_and_plot(params, label_prefix):
    sol = solve_ivp(lotka_volterra, (t_data[0], t_data[-1]), [x_data[0], y_data[0]],
                    args=tuple(params), t_eval=t_data)
    plt.plot(t_data, sol.y[0], '--', label=f'{label_prefix} Species 1')
    plt.plot(t_data, sol.y[1], '--', label=f'{label_prefix} Species 2')

plt.figure(figsize=(10, 6))
plt.plot(t_data, x_data, 'o', label='Observed Species 1')
plt.plot(t_data, y_data, 's', label='Observed Species 2')
simulate_and_plot(result_local.x, 'L-BFGS-B Fit')
simulate_and_plot(result_global.x, 'DE Fit')

plt.title("Model Fit vs Real Data")
plt.xlabel("Day")
plt.ylabel("Population Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model_fit_comparison.png")
plt.show()


# Step 4: Goodness-of-fit: RMSE Function, Print RMSE for Both Fits

def compute_rmse(params, t_data, x_data, y_data):
    sol = solve_ivp(lotka_volterra, (t_data[0], t_data[-1]), [x_data[0], y_data[0]],
                    args=tuple(params), t_eval=t_data)
    if not sol.success:
        return np.nan
    x_pred, y_pred = sol.y
    rmse_x = np.sqrt(np.mean((x_data - x_pred)**2))
    rmse_y = np.sqrt(np.mean((y_data - y_pred)**2))
    return rmse_x, rmse_y

rmse_local = compute_rmse(result_local.x, t_data, x_data, y_data)
rmse_global = compute_rmse(result_global.x, t_data, x_data, y_data)

print("\nRMSE (L-BFGS-B): Species 1 = {:.2f}, Species 2 = {:.2f}".format(*rmse_local))
print("RMSE (Differential Evolution): Species 1 = {:.2f}, Species 2 = {:.2f}".format(*rmse_global))