# pip install pandas 
# pip install matplotlib 
# pip install openpyxl
# pip install scipy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution, Bounds


# -----------------------------------------------------------------------
# Phase-1: Data preparation and visualization
# -----------------------------------------------------------------------
# Step-1: Load the data from the Excel file
# Step-2: Filter the data for "Mixture" treatment
# Step-3: Plot the data for "Mixture" treatment
# -----------------------------------------------------------------------


# Step 1: Load the data from the Excel file
# ---------------------------------------------------------
data = pd.read_excel('gauseR.xlsx', sheet_name='Sheet1')
# print(data.head())
print("\n\n")


# Step 2: Filter the data for "Mixture" treatment
# ---------------------------------------------------------
# The data contains two treatments: "Species1" and "Species2" (single species) and "Mixture" (both species together)
# We will filter the data for "Mixture" treatment to analyze the competition between the two species.
mixture_data = data[data['Treatment'] == 'Mixture']
# print(mixture_data.head())


# Step 3: Plot the data for "Mixture" treatment
# ---------------------------------------------------------
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

# Phase-1 CONCLUDED
# -----------------------------------------------------------------------   
# -----------------------------------------------------------------------



# -----------------------------------------------------------------------
# Phase-2: Model fitting and model evaluation functions
# -----------------------------------------------------------------------
# Step-1: Define the Lotka-Volterra equations
# Step-2: Define the loss function for optimization
# Step-3: Define the RMSE function
# Step-4: Define the MAE function
# -----------------------------------------------------------------------


# Step 1: Define the Lotka-Volterra equations
# ---------------------------------------------------------
# The Lotka-Volterra equations for two species with interspecific competition are given by:
def lotka_volterra(t, z, r1, r2, K1, K2, alpha, beta):
    x, y = z
    dxdt = r1 * x * (1 - (x + alpha * y) / K1)
    dydt = r2 * y * (1 - (y + beta * x) / K2)
    return [dxdt, dydt]


# Step 2: Define the loss function for optimization
# -----------------------------------------------------------
# The loss function calculates the sum of squared differences between observed and predicted values
def loss_function(params, t_data, x_data, y_data):
    r1, r2, K1, K2, alpha, beta = params
    z0 = [x_data[0], y_data[0]]  # initial populations
    sol = solve_ivp(lotka_volterra, (t_data[0], t_data[-1]), z0,
                    args=(r1, r2, K1, K2, alpha, beta), t_eval=t_data)
    if not sol.success:
        return np.inf
    x_pred, y_pred = sol.y
    return np.sum((x_data - x_pred)**2 + (y_data - y_pred)**2)


# Step 3: Define the RMSE function
# -----------------------------------------------------------
# The RMSE function calculates the root mean square error between observed and predicted values
def compute_rmse(params, t_data, x_data, y_data):
    sol = solve_ivp(lotka_volterra, (t_data[0], t_data[-1]), [x_data[0], y_data[0]],
                    args=tuple(params), t_eval=t_data)
    if not sol.success:
        return np.nan
    x_pred, y_pred = sol.y
    rmse_x = np.sqrt(np.mean((x_data - x_pred)**2))
    rmse_y = np.sqrt(np.mean((y_data - y_pred)**2))
    return rmse_x, rmse_y


# Step 4: Define the MAE function
# -----------------------------------------------------------
# The MAE function calculates the mean absolute error between observed and predicted values
def compute_mae(params, t_data, x_data, y_data):
    sol = solve_ivp(lotka_volterra, (t_data[0], t_data[-1]), [x_data[0], y_data[0]],
                    args=tuple(params), t_eval=t_data)
    if not sol.success:
        return np.nan
    x_pred, y_pred = sol.y
    mae_x = np.mean(np.abs(x_data - x_pred))
    mae_y = np.mean(np.abs(y_data - y_pred))
    return mae_x, mae_y

#  Phase-2 CONCLUDED
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------



# -----------------------------------------------------------------------
# Phase-3: Optimization and model fitting
# -----------------------------------------------------------------------
# Step-1: Prepare data for optimization
# Step-2: Define initial parameter guess and bounds
# Step-3: Minimize the loss function using L-BFGS-B (local optimizer)
# Step-4: Refine the optimization using Differential Evolution (global optimizer)
# Step-5: Plot the model fits against the real data
# Step-6: Compute RMSE for both fits
# Step-7: Compute MAE for both fits
# -----------------------------------------------------------------------


# Step 1: Prepare data for optimization
# ---------------------------------------------------------
t_data = mixture_data['Day'].values
x_data = mixture_data['Volume_Species1'].values
y_data = mixture_data['Volume_Species2'].values


# Step 2: Define initial parameter guess and bounds
# ---------------------------------------------------------
initial_guess = [0.5, 0.3, 100, 100, 0.5, 0.5]
bounds = Bounds([0.01, 0.01, 10, 10, 0.01, 0.01], [5, 5, 500, 500, 5, 5])


# Step 3: Minimize the loss function using L-BFGS-B (local optimizer)
# ---------------------------------------------------------
result_local = minimize(loss_function, initial_guess,
                        args=(t_data, x_data, y_data),
                        method='L-BFGS-B', bounds=bounds)

print("\nOptimized Parameters (L-BFGS-B with bounds):")
print("r1 = {:.3f}, r2 = {:.3f}, K1 = {:.2f}, K2 = {:.2f}, alpha = {:.2f}, beta = {:.2f}".format(*result_local.x))


# Step 4: Refine the optimization using Differential Evolution (global optimizer)
# ---------------------------------------------------------
de_bounds = [(0.01, 5), (0.01, 5), (10, 500), (10, 500), (0.01, 5), (0.01, 5)]
result_global = differential_evolution(loss_function, de_bounds, args=(t_data, x_data, y_data), strategy='best1bin')

print("\nOptimized Parameters (Differential Evolution):")
print("r1 = {:.3f}, r2 = {:.3f}, K1 = {:.2f}, K2 = {:.2f}, alpha = {:.2f}, beta = {:.2f}".format(*result_global.x))


# Step 5: Plot the model fits against the real data
# ---------------------------------------------------------
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
plt.savefig("lotka_volterra_diffrential_equations_optimization_solution.png")
plt.show()


# Step 6: Compute RMSE for both fits
# ---------------------------------------------------------
rmse_local = compute_rmse(result_local.x, t_data, x_data, y_data)
rmse_global = compute_rmse(result_global.x, t_data, x_data, y_data)

print("\nRMSE (L-BFGS-B): Species 1 = {:.2f}, Species 2 = {:.2f}".format(*rmse_local))
print("RMSE (Differential Evolution): Species 1 = {:.2f}, Species 2 = {:.2f}".format(*rmse_global))


# Step 7: Compute MAE for both fits
# ---------------------------------------------------------
mae_local = compute_mae(result_local.x, t_data, x_data, y_data)
mae_global = compute_mae(result_global.x, t_data, x_data, y_data)

print("MAE (L-BFGS-B): Species 1 = {:.2f}, Species 2 = {:.2f}".format(*mae_local))
print("MAE (Differential Evolution): Species 1 = {:.2f}, Species 2 = {:.2f}".format(*mae_global))

# Phase-3 CONCLUDED
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------



# -----------------------------------------------------------------------
# Phase-4: Approximation Methods - Euler's Method and RK4 Method
# -----------------------------------------------------------------------
# Step-1: Define the Euler's method function
# Step-2: Implement the Euler's method for the Lotka-Volterra equations with step size h=0.5
# Step-3: Implement the Euler's method for the Lotka-Volterra equations with step size h=0.25
# Step-4: Plot the results of Euler's methods vs IVP solution
# Step-5: Compute RMSE and MAE for Euler's method
# Step-6: Define the RK4 method function
# Step-7: Implement the RK4 method for the Lotka-Volterra equations with step size h=0.5
# Step-8: Implement the RK4 method for the Lotka-Volterra equations with step size h=0.25
# Step-9: Plot the results of RK4 methods vs IVP solution
# Step-10: Compute RMSE and MAE for RK4 method
# Step-11: Comparision of all three methods
# -----------------------------------------------------------------------


# Step 1: Define the Euler's method function
# ---------------------------------------------------------
def euler_method(f, t_span, z0, h, params):
    t0, tf = t_span
    t_values = np.arange(t0, tf + h, h)
    z_values = np.zeros((len(t_values), len(z0)))
    z_values[0] = z0
    
    for i in range(1, len(t_values)):
        z_values[i] = z_values[i-1] + h * np.array(f(t_values[i-1], z_values[i-1], *params))

    return t_values, z_values


# Step 2: Implement the Euler's method for the Lotka-Volterra equations with step size h=0.5
# ---------------------------------------------------------
h1 = 0.5  # step size h1 for Euler method
euler_t, euler_sol = euler_method(lotka_volterra, (t_data[0], t_data[-1]), [x_data[0], y_data[0]], h1, result_global.x)


# Step-3: Implement the Euler's method for the Lotka-Volterra equations with step size h=0.25
# ---------------------------------------------------------
h2 = 0.25  # step size h2 for Euler method
alt_euler_t, alt_euler_sol = euler_method(lotka_volterra, (t_data[0], t_data[-1]), [x_data[0], y_data[0]], h2, result_global.x)


# Step 4: Plot the results of Euler's methods vs IVP solution
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_data, x_data, 'o', label='Observed Species 1')
plt.plot(t_data, y_data, 's', label='Observed Species 2')
plt.plot(euler_t, euler_sol[:, 0], '--', label='Euler (h=0.50) Species 1')
plt.plot(euler_t, euler_sol[:, 1], '--', label='Euler (h=0.50) Species 2')
plt.plot(alt_euler_t, alt_euler_sol[:, 0], ':', label='Euler (h=0.25) Species 1')
plt.plot(alt_euler_t, alt_euler_sol[:, 1], ':', label='Euler (h=0.25) Species 2')

simulate_and_plot(result_local.x, 'L-BFGS-B Fit')
simulate_and_plot(result_global.x, 'DE Fit')

plt.title("Euler's Method Simulation with Two Step Sizes vs Real Data")
plt.xlabel("Day")
plt.ylabel("Population Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("euler_method_comparison.png")
plt.show()


# Step 5: Compute the RMSE and MAE for Euler's method
# ---------------------------------------------------------
# Interpolate Euler predictions to match observed time points
interp_euler_x = np.interp(t_data, euler_t, euler_sol[:, 0])
interp_euler_y = np.interp(t_data, euler_t, euler_sol[:, 1])
interp_alt_x = np.interp(t_data, alt_euler_t, alt_euler_sol[:, 0])
interp_alt_y = np.interp(t_data, alt_euler_t, alt_euler_sol[:, 1])

# Compute RMSE
rmse_euler_h1_x = np.sqrt(np.mean((x_data - interp_euler_x) ** 2))
rmse_euler_h1_y = np.sqrt(np.mean((y_data - interp_euler_y) ** 2))
rmse_euler_h2_x = np.sqrt(np.mean((x_data - interp_alt_x) ** 2))
rmse_euler_h2_y = np.sqrt(np.mean((y_data - interp_alt_y) ** 2))

# Compute MAE
mae_euler_h1_x = np.mean(np.abs(x_data - interp_euler_x))
mae_euler_h1_y = np.mean(np.abs(y_data - interp_euler_y))
mae_euler_h2_x = np.mean(np.abs(x_data - interp_alt_x))
mae_euler_h2_y = np.mean(np.abs(y_data - interp_alt_y))

print("RMSE (Euler h1=0.5): Species 1 = {:.2f}, Species 2 = {:.2f}".format(rmse_euler_h1_x, rmse_euler_h1_y))
print("RMSE (Euler h2=0.25): Species 1 = {:.2f}, Species 2 = {:.2f}".format(rmse_euler_h2_x, rmse_euler_h2_y))
print("MAE (Euler h1=0.5): Species 1 = {:.2f}, Species 2 = {:.2f}".format(mae_euler_h1_x, mae_euler_h1_y))
print("MAE (Euler h2=0.25): Species 1 = {:.2f}, Species 2 = {:.2f}".format(mae_euler_h2_x, mae_euler_h2_y))


# Step 6: Define the RK4 method function
# ---------------------------------------------------------



