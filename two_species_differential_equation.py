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
# PART-1: Differential equation modelling
# -----------------------------------------------------------------------   
# Phase-1: Data preparation and visualization
# Phase-2: Model fitting and model evaluation functions
# Phase-3: Optimization and model fitting
# Phase-4: Approximation Methods - Euler's Method and RK4 Method
# Phase-5: Stability Analysis & Null clines
# -----------------------------------------------------------------------



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
# Step 12: Summary table of RMSE and MAE for all methods
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
def rk4_method(f, t_span, z0, h, params):
    t0, tf = t_span
    t_values = np.arange(t0, tf + h, h)
    z_values = np.zeros((len(t_values), len(z0)))
    z_values[0] = z0

    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        z = z_values[i - 1]

        k1 = np.array(f(t, z, *params))
        k2 = np.array(f(t + h / 2, z + h * k1 / 2, *params))
        k3 = np.array(f(t + h / 2, z + h * k2 / 2, *params))
        k4 = np.array(f(t + h, z + h * k3, *params))

        z_values[i] = z + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t_values, z_values


# Step 7: Implement the RK4 method for the Lotka-Volterra equations with step size h=0.5
# ---------------------------------------------------------
rk4_t1, rk4_sol1 = rk4_method(lotka_volterra, (t_data[0], t_data[-1]), [x_data[0], y_data[0]], h1, result_global.x)


# Step 8: Implement the RK4 method for the Lotka-Volterra equations with step size h=0.25
# ---------------------------------------------------------
rk4_t2, rk4_sol2 = rk4_method(lotka_volterra, (t_data[0], t_data[-1]), [x_data[0], y_data[0]], h2, result_global.x)


# Step 9: Plot the results of RK4 methods vs IVP solution
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_data, x_data, 'o', label='Observed Species 1')
plt.plot(t_data, y_data, 's', label='Observed Species 2')
plt.plot(rk4_t1, rk4_sol1[:, 0], '--', label='RK4 (h=0.50) Species 1')
plt.plot(rk4_t1, rk4_sol1[:, 1], '--', label='RK4 (h=0.50) Species 2')
plt.plot(rk4_t2, rk4_sol2[:, 0], ':', label='RK4 (h=0.25) Species 1')
plt.plot(rk4_t2, rk4_sol2[:, 1], ':', label='RK4 (h=0.25) Species 2')
simulate_and_plot(result_global.x, 'solve_ivp')

plt.title("RK4 Method Simulation with Two Step Sizes vs Real Data")
plt.xlabel("Day")
plt.ylabel("Population Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rk4_method_comparison.png")
plt.show()


# Step 10: Compute RMSE and MAE for RK4 method
# ---------------------------------------------------------
interp_rk4_h1_x = np.interp(t_data, rk4_t1, rk4_sol1[:, 0])
interp_rk4_h1_y = np.interp(t_data, rk4_t1, rk4_sol1[:, 1])
interp_rk4_h2_x = np.interp(t_data, rk4_t2, rk4_sol2[:, 0])
interp_rk4_h2_y = np.interp(t_data, rk4_t2, rk4_sol2[:, 1])

rmse_rk4_h1_x = np.sqrt(np.mean((x_data - interp_rk4_h1_x) ** 2))
rmse_rk4_h1_y = np.sqrt(np.mean((y_data - interp_rk4_h1_y) ** 2))
rmse_rk4_h2_x = np.sqrt(np.mean((x_data - interp_rk4_h2_x) ** 2))
rmse_rk4_h2_y = np.sqrt(np.mean((y_data - interp_rk4_h2_y) ** 2))

mae_rk4_h1_x = np.mean(np.abs(x_data - interp_rk4_h1_x))
mae_rk4_h1_y = np.mean(np.abs(y_data - interp_rk4_h1_y))
mae_rk4_h2_x = np.mean(np.abs(x_data - interp_rk4_h2_x))
mae_rk4_h2_y = np.mean(np.abs(y_data - interp_rk4_h2_y))

print("RMSE (RK4 h1=0.5): Species 1 = {:.2f}, Species 2 = {:.2f}".format(rmse_rk4_h1_x, rmse_rk4_h1_y))
print("RMSE (RK4 h2=0.25): Species 1 = {:.2f}, Species 2 = {:.2f}".format(rmse_rk4_h2_x, rmse_rk4_h2_y))
print("MAE (RK4 h1=0.5): Species 1 = {:.2f}, Species 2 = {:.2f}".format(mae_rk4_h1_x, mae_rk4_h1_y))
print("MAE (RK4 h2=0.25): Species 1 = {:.2f}, Species 2 = {:.2f}".format(mae_rk4_h2_x, mae_rk4_h2_y))


# Step 11: Comparison of all three methods
# ---------------------------------------------------------
# Plot for step size h = 0.5
plt.figure(figsize=(10, 6))
plt.plot(t_data, x_data, 'o', label='Observed Species 1')
plt.plot(t_data, y_data, 's', label='Observed Species 2')
plt.plot(euler_t, euler_sol[:, 0], '--', label='Euler (h=0.50) Species 1')
plt.plot(euler_t, euler_sol[:, 1], '--', label='Euler (h=0.50) Species 2')
plt.plot(rk4_t1, rk4_sol1[:, 0], ':', label='RK4 (h=0.50) Species 1')
plt.plot(rk4_t1, rk4_sol1[:, 1], ':', label='RK4 (h=0.50) Species 2')
simulate_and_plot(result_global.x, 'solve_ivp')
plt.title("Comparison of Methods (Step Size = 0.5)")
plt.xlabel("Day")
plt.ylabel("Population Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison_step_0.5.png")
plt.show()

# Plot for step size h = 0.25
plt.figure(figsize=(10, 6))
plt.plot(t_data, x_data, 'o', label='Observed Species 1')
plt.plot(t_data, y_data, 's', label='Observed Species 2')
plt.plot(alt_euler_t, alt_euler_sol[:, 0], '--', label='Euler (h=0.25) Species 1')
plt.plot(alt_euler_t, alt_euler_sol[:, 1], '--', label='Euler (h=0.25) Species 2')
plt.plot(rk4_t2, rk4_sol2[:, 0], ':', label='RK4 (h=0.25) Species 1')
plt.plot(rk4_t2, rk4_sol2[:, 1], ':', label='RK4 (h=0.25) Species 2')
simulate_and_plot(result_global.x, 'solve_ivp')
plt.title("Comparison of Methods (Step Size = 0.25)")
plt.xlabel("Day")
plt.ylabel("Population Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison_step_0.25.png")
plt.show()


# Step 12: Summary table of RMSE and MAE for all methods
# ---------------------------------------------------------
summary_data = {
    'Method': [
        'Euler h=0.50', 'Euler h=0.25',
        'RK4 h=0.50', 'RK4 h=0.25',
        'solve_ivp (L-BFGS-B)', 'solve_ivp (DE)'
    ],
    'RMSE Species 1': [
        rmse_euler_h1_x, rmse_euler_h2_x,
        rmse_rk4_h1_x, rmse_rk4_h2_x,
        rmse_local[0], rmse_global[0]
    ],
    'RMSE Species 2': [
        rmse_euler_h1_y, rmse_euler_h2_y,
        rmse_rk4_h1_y, rmse_rk4_h2_y,
        rmse_local[1], rmse_global[1]
    ],
    'MAE Species 1': [
        mae_euler_h1_x, mae_euler_h2_x,
        mae_rk4_h1_x, mae_rk4_h2_x,
        mae_local[0], mae_global[0]
    ],
    'MAE Species 2': [
        mae_euler_h1_y, mae_euler_h2_y,
        mae_rk4_h1_y, mae_rk4_h2_y,
        mae_local[1], mae_global[1]
    ]
}

summary_df = pd.DataFrame(summary_data)
print("Summary of RMSE and MAE for All Methods:")
print(summary_df.to_string(index=False))
summary_df.to_csv("model_comparison_summary.csv", index=False)



# Phase-4 CONCLUDED
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------



# ----------------------------------------------------------------------
# Phase-5: Stability Analysis & Null clines
# ----------------------------------------------------------------------
# Step-1: Compute Equilibrium Points (fixed points) from LV equations
# Step-2: Compute Jacobian matrix at equilibrium points
# Step-3: Compute Eigenvalues of the Jacobian matrix and analyze stability
# Step-4: Plot the nullclines and Phase plane trajectories
# Step-5: Overlay Model Trajectories on Phase Plane
# -----------------------------------------------------------------------


# Step-1: Compute Equilibrium Points (fixed points) from LV equations
# ---------------------------------------------------------
def compute_equilibrium_points(r1, r2, K1, K2, alpha, beta):
    # Assuming interior equilibrium exists and is stable:
    x_star = (K1 - alpha * K2) / (1 - alpha * beta)
    y_star = (K2 - beta * K1) / (1 - alpha * beta)
    return x_star, y_star

# Use the parameters from global optimization
r1, r2, K1, K2, alpha, beta = result_global.x
x_star, y_star = compute_equilibrium_points(r1, r2, K1, K2, alpha, beta)

print("Equilibrium Point (Interior):")
print("x* = {:.2f}, y* = {:.2f}".format(x_star, y_star))


# Step-2: Compute Jacobian matrix at equilibrium points
# ---------------------------------------------------------
def compute_jacobian(x_star, y_star, r1, r2, K1, K2, alpha, beta):
    df1_dx = r1 * (1 - (2 * x_star + alpha * y_star) / K1)
    df1_dy = r1 * (-alpha * x_star / K1)
    df2_dx = r2 * (-beta * y_star / K2)
    df2_dy = r2 * (1 - (2 * y_star + beta * x_star) / K2)

    J = np.array([[df1_dx, df1_dy],
                  [df2_dx, df2_dy]])
    return J

jacobian_matrix = compute_jacobian(x_star, y_star, r1, r2, K1, K2, alpha, beta)
print("Jacobian Matrix at Equilibrium:")
print(jacobian_matrix)


# Step-3: Compute Eigenvalues of the Jacobian matrix and analyze stability
# ---------------------------------------------------------
eigenvalues, _ = np.linalg.eig(jacobian_matrix)
trace = np.trace(jacobian_matrix)
det = np.linalg.det(jacobian_matrix)

print("Eigenvalues of the Jacobian:")
print(eigenvalues)

# Eigenvalue-based analysis
if np.all(np.real(eigenvalues) < 0):
    print("=> Eigenvalue: The equilibrium is locally stable (attractor).")
elif np.all(np.real(eigenvalues) > 0):
    print("=> Eigenvalue: The equilibrium is unstable (repellor).")
else:
    print("=> Eigenvalue: The equilibrium is a saddle point or has mixed stability.")

print("Jacobian Matrix Trace: {:.4f}".format(trace))
print("Jacobian Matrix Determinant: {:.4f}".format(det))

# Trace-Determinant analysis
if det > 0 and trace < 0:
    print("=> Trace-Det: The equilibrium is locally stable (attractor).")
elif det > 0 and trace > 0:
    print("=> Trace-Det: The equilibrium is unstable (repellor).")
elif det < 0:
    print("=> Trace-Det: The equilibrium is a saddle point (unstable).")
else:
    print("=> Trace-Det: Mixed or undefined behavior.")
    print("=> Trace-Det: Mixed or undefined behavior.")


# Step-4: Plot the nullclines and equilibrium points
# ---------------------------------------------------------
x_vals = np.linspace(0, K1 * 1.1, 300)
y_vals = np.linspace(0, K2 * 1.1, 300)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute vector field
U = r1 * X * (1 - (X + alpha * Y) / K1)
V = r2 * Y * (1 - (Y + beta * X) / K2)

# Species 1 nullcline: dx/dt = 0 => x + alpha*y = K1 => y = (K1 - x)/alpha
y_nullcline1 = (K1 - x_vals) / alpha
# Species 2 nullcline: dy/dt = 0 => y + beta*x = K2 => y = K2 - beta*x
y_nullcline2 = K2 - beta * x_vals

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_nullcline1, label='Species 1 Nullcline (dx/dt=0)')
plt.plot(x_vals, y_nullcline2, label='Species 2 Nullcline (dy/dt=0)')
plt.plot(x_star, y_star, 'ro', label='Equilibrium Point')

# Add vector field to show phase plane flow
plt.streamplot(X, Y, U, V, density=1.0, color='gray', arrowsize=1)

plt.xlim(0, K1 * 1.05)
plt.ylim(0, K2 * 1.05)
plt.xlabel('Species 1 Population')
plt.ylabel('Species 2 Population')
plt.title('Nullclines and Phase Plane Trajectories')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("nullclines_phase_plane.png")
plt.show()


# Step-5: Overlay Model Trajectories on Phase Plane
# ---------------------------------------------------------
x_vals = np.linspace(0, K1 * 1.1, 300)
y_vals = np.linspace(0, K2 * 1.1, 300)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute vector field
U = r1 * X * (1 - (X + alpha * Y) / K1)
V = r2 * Y * (1 - (Y + beta * X) / K2)

plt.figure(figsize=(10, 6))
plt.streamplot(X, Y, U, V, density=1.0, color='gray', arrowsize=1)
plt.plot(x_star, y_star, 'ro', label='Equilibrium Point')
plt.plot(euler_sol[:, 0], euler_sol[:, 1], '--', label='Euler Trajectory (h=0.5)')
plt.plot(rk4_sol1[:, 0], rk4_sol1[:, 1], ':', label='RK4 Trajectory (h=0.5)')
sol_ivp_overlay = solve_ivp(lotka_volterra, (t_data[0], t_data[-1]), [x_data[0], y_data[0]],
                             args=tuple(result_global.x), t_eval=t_data)
plt.plot(sol_ivp_overlay.y[0], sol_ivp_overlay.y[1], '-', label='solve_ivp Trajectory')

plt.xlim(0, K1 * 1.05)
plt.ylim(0, K2 * 1.05)
plt.xlabel('Species 1 Population')
plt.ylabel('Species 2 Population')
plt.title('Phase Plane with Model Trajectories')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("phase_plane_trajectories.png")
plt.show()



# Phase-5 CONCLUDED
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------



# End of the PART-1 (Diffrential equation modelling)
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------