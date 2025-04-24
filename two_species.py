# pip install pandas 
# pip install matplotlib 
# pip install openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

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

def lotka_volterra(t, z, r1, r2, K1, K2, alpha, beta):
    x, y = z
    dxdt = r1 * x * (1 - (x + alpha * y) / K1)
    dydt = r2 * y * (1 - (y + beta * x) / K2)
    return [dxdt, dydt]

# -------------------------------
# Define Loss Function
# -------------------------------
def loss_function(params, t_data, x_data, y_data):
    r1, r2, K1, K2, alpha, beta = params
    z0 = [x_data[0], y_data[0]]  # initial populations
    sol = solve_ivp(lotka_volterra, (t_data[0], t_data[-1]), z0,
                    args=(r1, r2, K1, K2, alpha, beta), t_eval=t_data)
    if not sol.success:
        return np.inf
    x_pred, y_pred = sol.y
    return np.sum((x_data - x_pred)**2 + (y_data - y_pred)**2)

# -------------------------------
# Prepare Data for Optimization
# -------------------------------
t_data = mixture_data['Day'].values
x_data = mixture_data['Volume_Species1'].values
y_data = mixture_data['Volume_Species2'].values

# -------------------------------
# Initial Parameter Guess
# -------------------------------
initial_guess = [0.5, 0.3, 100, 100, 0.5, 0.5]

# -------------------------------
# Minimize the Loss
# -------------------------------
result = minimize(loss_function, initial_guess,
                  args=(t_data, x_data, y_data),
                  method='Nelder-Mead')

print("\nOptimized Parameters:")
print("r1 = {:.3f}, r2 = {:.3f}, K1 = {:.2f}, K2 = {:.2f}, alpha = {:.2f}, beta = {:.2f}".format(*result.x))
