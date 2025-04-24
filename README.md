# Modelling Two-Species Competition

This project models and visualizes the classical **Lotka-Volterra competition** between two species using real biological data from *Gause's Paramecium* experiments. The goal is to simulate population dynamics, analyze stability, and understand outcomes like coexistence or competitive exclusion.

## 📁 Project Structure

. ├── data/ # Raw dataset (e.g., gauseR.xlsx) ├── src/ # Python scripts for simulation, plotting ├── figures/ # Output graphs ├── two_species.py # Starting script ├── README.md # You're here! └── requirements.txt # Python dependencies

## 📊 Objective

- Model population dynamics between two competing species.
- Fit or simulate **Lotka-Volterra competition equations**.
- Visualize time series and phase-plane behavior.
- Analyze biological interpretations and stability.

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
▶️ Run the Code
bash
Copy
Edit
python two_species.py
Output plots will be saved in the figures/ directory.

📚 Dataset
Based on digitized historical data from:

Gause’s 1930s experiments

Source: gauseR.xlsx or the gauseR package in R

📈 What's Coming
Numerical simulation using Euler/RK4 methods

Nullcline and phase-plane analysis

Parameter estimation and sensitivity analysis