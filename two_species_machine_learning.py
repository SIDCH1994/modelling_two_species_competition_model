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
# Phase-5: kNN Regressor (Multi-output)
# Phase-6: Physics-Informed Neural Network (PINN)
# Phase-7: Model Comparison using tables and Visulization
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
plt.show()

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
plt.show()

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
plt.show()

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
plt.show()

# ---------------------------------------------------------------------------------
# Phase-4 CONCLUDED
# ---------------------------------------------------------------------------------



# --------------------------------------------------------------------------------
# Phase-5: kNN Regressor (Multi-output)
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
# Phase-5 CONCLUDED
# ---------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------
# Phase-6: Physics-Informed Neural Network (PINN)
# ---------------------------------------------------------------------------------
# Step-1: Prepare data
# Step-2: Define LV parameters (from your solve_ivp fit)
# Step-3: Define PINN
# Step-4: Define loss function
# Step-5: Train model
# Step-6: Predict and plot
# ---------------------------------------------------------------------------------


# Step-1: Prepare data
# ------------------------------------------------------------------
t = torch.tensor(mixture_data['Day'].values, dtype=torch.float32).view(-1, 1)
species = torch.tensor(mixture_data[['Volume_Species1', 'Volume_Species2']].values, dtype=torch.float32)


# Step-2: Lotka-Volterra parameters (fixed)
# ------------------------------------------------------------------
r1, r2 = 1.010, 0.828
K1, K2 = 254.35, 146.40
alpha, beta = 1.63, 0.24


# Step-3: Define PINN
# ------------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Softplus(),
            nn.Linear(32, 32), nn.Softplus(),
            nn.Linear(32, 2)
        )

    def forward(self, t):
        return self.net(t)

# Step-4: Combined loss function
def pinn_loss(model, t, y_true):
    t.requires_grad_(True)
    y_pred = model(t)
    x, y = y_pred[:, 0:1], y_pred[:, 1:2]

    # Derivatives
    dxdt = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dydt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    # Lotka-Volterra physics
    dxdt_lv = r1 * x * (1 - (x + alpha * y) / K1)
    dydt_lv = r2 * y * (1 - (y + beta * x) / K2)

    # Loss
    data_loss = torch.mean((x - y_true[:, 0:1])**2 + (y - y_true[:, 1:2])**2)
    ode_loss = torch.mean((dxdt - dxdt_lv)**2 + (dydt - dydt_lv)**2)
    total_loss = data_loss + 0.1 * ode_loss
    return total_loss, data_loss.detach(), ode_loss.detach()

# Step-5: Train model
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
epochs = 5000

for epoch in range(epochs):
    loss, data_l, ode_l = pinn_loss(model, t, species)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Total Loss={loss.item():.2f}, Data={data_l.item():.2f}, ODE={ode_l.item():.2f}")

# Step-6: Predict and Evaluate
with torch.no_grad():
    t_test = torch.linspace(t.min(), t.max(), 100).view(-1, 1)
    pred_test = model(t_test).detach().numpy()

    pred_obs = model(t).detach().numpy()
    y_true = species.detach().numpy()

    rmse_pinn_x = np.sqrt(mean_squared_error(y_true[:, 0], pred_obs[:, 0]))
    rmse_pinn_y = np.sqrt(mean_squared_error(y_true[:, 1], pred_obs[:, 1]))
    mae_pinn_x = np.mean(np.abs(y_true[:, 0] - pred_obs[:, 0]))
    mae_pinn_y = np.mean(np.abs(y_true[:, 1] - pred_obs[:, 1]))

    print("\n✅ PINN Final Evaluation:")
    print(f"RMSE (Species 1): {rmse_pinn_x:.2f}")
    print(f"RMSE (Species 2): {rmse_pinn_y:.2f}")
    print(f"MAE (Species 1): {mae_pinn_x:.2f}")
    print(f"MAE (Species 2): {mae_pinn_y:.2f}")

# Step-7: Plot
plt.figure(figsize=(10, 6))
plt.plot(t.detach().numpy(), y_true[:, 0], 'o', label='Observed Species 1')
plt.plot(t.detach().numpy(), y_true[:, 1], 's', label='Observed Species 2')
plt.plot(t_test.detach().numpy(), pred_test[:, 0], '--', label='PINN Species 1')
plt.plot(t_test.detach().numpy(), pred_test[:, 1], '--', label='PINN Species 2')
plt.plot(t_data, ivp_species1, ':', label='LV solve_ivp Species 1')
plt.plot(t_data, ivp_species2, ':', label='LV solve_ivp Species 2')

plt.title("Physics-Informed Neural Network vs LV Model")
plt.xlabel("Day")
plt.ylabel("Population Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pinn_vs_lv.png")
plt.show()

# ---------------------------------------------------------------------------------
# Phase-6 CONCLUDED
# ---------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------
# Phase-7: Model Comparison using tables and Visulization
# ---------------------------------------------------------------------------------
# Step-1: Create a DataFrame for model evaluation metrics
# Step-2: Bar chart comparing IVP with all 6 ML models using RMSE
# Step-3: Bar chart comparing IVP with all 6 ML models using MAE
# ---------------------------------------------------------------------------------


# Step-1: Create a DataFrame for model evaluation metrics
# ------------------------------------------------------------------
comparison_df = pd.DataFrame({
    "Model": ["LV IVP", "Linear", "Ridge", "Polynomial", "GPR", "kNN", "PINN"],
    "RMSE (Species 1)": [9.72, 30.73, 30.73, 16.29, 6.73, 9.43, 6.45],
    "RMSE (Species 2)": [10.05, 23.62, 23.62, 10.45, 9.52, 9.25, 6.59],
    "MAE (Species 1)": [8.07, 23.10, 23.10, 13.16, 5.57, 7.42, 4.82],
    "MAE (Species 2)": [7.23, 20.78, 20.78, 8.56, 7.48, 7.90, 5.15]
})

# Bar plot parameters
models = comparison_df["Model"].tolist()
x = np.arange(len(models))
width = 0.35


# Step-2: Bar chart comparing IVP with all 6 ML models using RMSE
# ------------------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, comparison_df["RMSE (Species 1)"], width, label='Species 1 RMSE')
plt.bar(x + width/2, comparison_df["RMSE (Species 2)"], width, label='Species 2 RMSE')
plt.xticks(x, models, rotation=30)
plt.ylabel("RMSE")
plt.title("RMSE Comparison (All Models + LV IVP)")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Step-3: Bar chart comparing IVP with all 6 ML models using MAE
# ------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, comparison_df["MAE (Species 1)"], width, label='Species 1 MAE')
plt.bar(x + width/2, comparison_df["MAE (Species 2)"], width, label='Species 2 MAE')
plt.xticks(x, models, rotation=30)
plt.ylabel("MAE")
plt.title("MAE Comparison (All Models + LV IVP)")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

comparison_df.to_csv("model_comparison_table.csv", index=False)

# ----------------------------------------------------------------------------------
# Phase-7 CONCLUDED
# ----------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------
# END OF THE PART_2
# ---------------------------------------------------------------------------------



