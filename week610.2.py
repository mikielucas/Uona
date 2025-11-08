"""
Case Study Assignment 4 - Report 10.2: India & China Population Analysis
Author: Statistical Analysis
Date: November 2025

This script analyzes the India and China population dataset using:
- Trend Regression Models (Linear, Quadratic, Exponential)
- Holt Exponential Smoothing (with trend)
- Model comparison and selection
- 5-year population forecasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Using numpy for statistical calculations

# Load the dataset
df = pd.read_excel('/home/ubuntu/upload/IndiaChinaDataset.xlsx')

print("=" * 80)
print("INDIA & CHINA POPULATION FORECASTING ANALYSIS")
print("=" * 80)
print("\nDataset Overview:")
print(df.head(10))
print(f"\nDataset Shape: {df.shape}")
print(f"\nDescriptive Statistics:")
print(df.describe())

# Create time variable (years since 1960)
df['Time'] = df['Year'] - 1960

# ============================================================================
# LINEAR TREND REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("LINEAR TREND REGRESSION")
print("=" * 80)

# Linear regression for India using numpy
india_linear_coef = np.polyfit(df['Time'], df['India'], 1)
india_linear_poly = np.poly1d(india_linear_coef)
df['India_Linear'] = india_linear_poly(df['Time'])

# Calculate R-squared for India linear model
ss_res_india_linear = np.sum((df['India'] - df['India_Linear'])**2)
ss_tot_india = np.sum((df['India'] - np.mean(df['India']))**2)
r2_india_linear = 1 - (ss_res_india_linear / ss_tot_india)

print(f"\nIndia Linear Model:")
print(f"  Equation: Population = {india_linear_coef[1]:.4f} + {india_linear_coef[0]:.4f} * Time")
print(f"  R-squared: {r2_india_linear:.6f}")

# Linear regression for China using numpy
china_linear_coef = np.polyfit(df['Time'], df['China'], 1)
china_linear_poly = np.poly1d(china_linear_coef)
df['China_Linear'] = china_linear_poly(df['Time'])

# Calculate R-squared for China linear model
ss_res_china_linear = np.sum((df['China'] - df['China_Linear'])**2)
ss_tot_china = np.sum((df['China'] - np.mean(df['China']))**2)
r2_china_linear = 1 - (ss_res_china_linear / ss_tot_china)

print(f"\nChina Linear Model:")
print(f"  Equation: Population = {china_linear_coef[1]:.4f} + {china_linear_coef[0]:.4f} * Time")
print(f"  R-squared: {r2_china_linear:.6f}")

# ============================================================================
# QUADRATIC TREND REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("QUADRATIC TREND REGRESSION")
print("=" * 80)

# Quadratic regression for India
india_quad_coef = np.polyfit(df['Time'], df['India'], 2)
india_quad_poly = np.poly1d(india_quad_coef)
df['India_Quadratic'] = india_quad_poly(df['Time'])

# Calculate R-squared for quadratic model
ss_res_india_quad = np.sum((df['India'] - df['India_Quadratic'])**2)
ss_tot_india = np.sum((df['India'] - np.mean(df['India']))**2)
r2_india_quad = 1 - (ss_res_india_quad / ss_tot_india)

print(f"\nIndia Quadratic Model:")
print(f"  Equation: Population = {india_quad_coef[0]:.6f} * Time² + {india_quad_coef[1]:.4f} * Time + {india_quad_coef[2]:.4f}")
print(f"  R-squared: {r2_india_quad:.6f}")

# Quadratic regression for China
china_quad_coef = np.polyfit(df['Time'], df['China'], 2)
china_quad_poly = np.poly1d(china_quad_coef)
df['China_Quadratic'] = china_quad_poly(df['Time'])

# Calculate R-squared for quadratic model
ss_res_china_quad = np.sum((df['China'] - df['China_Quadratic'])**2)
ss_tot_china = np.sum((df['China'] - np.mean(df['China']))**2)
r2_china_quad = 1 - (ss_res_china_quad / ss_tot_china)

print(f"\nChina Quadratic Model:")
print(f"  Equation: Population = {china_quad_coef[0]:.6f} * Time² + {china_quad_coef[1]:.4f} * Time + {china_quad_coef[2]:.4f}")
print(f"  R-squared: {r2_china_quad:.6f}")

# ============================================================================
# EXPONENTIAL TREND REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("EXPONENTIAL TREND REGRESSION")
print("=" * 80)

# Exponential regression for India (using log transformation)
india_log = np.log(df['India'])
india_exp_coef = np.polyfit(df['Time'], india_log, 1)
df['India_Exponential'] = np.exp(india_exp_coef[1]) * np.exp(india_exp_coef[0] * df['Time'])

# Calculate R-squared for exponential model
ss_res_india_exp = np.sum((df['India'] - df['India_Exponential'])**2)
r2_india_exp = 1 - (ss_res_india_exp / ss_tot_india)

print(f"\nIndia Exponential Model:")
print(f"  Equation: Population = {np.exp(india_exp_coef[1]):.4f} * e^({india_exp_coef[0]:.6f} * Time)")
print(f"  R-squared: {r2_india_exp:.6f}")

# Exponential regression for China
china_log = np.log(df['China'])
china_exp_coef = np.polyfit(df['Time'], china_log, 1)
df['China_Exponential'] = np.exp(china_exp_coef[1]) * np.exp(china_exp_coef[0] * df['Time'])

# Calculate R-squared for exponential model
ss_res_china_exp = np.sum((df['China'] - df['China_Exponential'])**2)
r2_china_exp = 1 - (ss_res_china_exp / ss_tot_china)

print(f"\nChina Exponential Model:")
print(f"  Equation: Population = {np.exp(china_exp_coef[1]):.4f} * e^({china_exp_coef[0]:.6f} * Time)")
print(f"  R-squared: {r2_china_exp:.6f}")

# ============================================================================
# HOLT'S EXPONENTIAL SMOOTHING (WITH TREND)
# ============================================================================

print("\n" + "=" * 80)
print("HOLT'S EXPONENTIAL SMOOTHING (WITH TREND)")
print("=" * 80)

def holt_smoothing(data, alpha, beta):
    """
    Holt's Exponential Smoothing with trend
    Level: L_t = alpha * Y_t + (1 - alpha) * (L_(t-1) + T_(t-1))
    Trend: T_t = beta * (L_t - L_(t-1)) + (1 - beta) * T_(t-1)
    """
    n = len(data)
    level = np.zeros(n)
    trend = np.zeros(n)
    forecast = np.zeros(n)
    
    # Initialize
    level[0] = data.iloc[0]
    trend[0] = data.iloc[1] - data.iloc[0]
    forecast[0] = level[0]
    
    for t in range(1, n):
        level[t] = alpha * data.iloc[t] + (1 - alpha) * (level[t-1] + trend[t-1])
        trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
        forecast[t] = level[t-1] + trend[t-1]
    
    return forecast, level, trend

# Test different alpha and beta combinations
alpha_beta_pairs = [(0.2, 0.1), (0.3, 0.1), (0.5, 0.2), (0.7, 0.3)]

holt_results_india = {}
holt_results_china = {}

for alpha, beta in alpha_beta_pairs:
    forecast_india, level_india, trend_india = holt_smoothing(df['India'], alpha, beta)
    holt_results_india[f'Holt_α{alpha}_β{beta}'] = {
        'forecast': forecast_india,
        'level': level_india,
        'trend': trend_india
    }
    df[f'India_Holt_α{alpha}_β{beta}'] = forecast_india
    
    forecast_china, level_china, trend_china = holt_smoothing(df['China'], alpha, beta)
    holt_results_china[f'Holt_α{alpha}_β{beta}'] = {
        'forecast': forecast_china,
        'level': level_china,
        'trend': trend_china
    }
    df[f'China_Holt_α{alpha}_β{beta}'] = forecast_china

print(f"\nTested Holt's method with different α and β combinations:")
for alpha, beta in alpha_beta_pairs:
    print(f"  α = {alpha}, β = {beta}")

# ============================================================================
# MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("MODEL EVALUATION - INDIA")
print("=" * 80)

metrics_india = []

# Linear
mae_linear = np.mean(np.abs(df['India'] - df['India_Linear']))
rmse_linear = np.sqrt(np.mean((df['India'] - df['India_Linear'])**2))
metrics_india.append({'Method': 'Linear Trend', 'MAE': mae_linear, 'RMSE': rmse_linear, 'R²': r2_india_linear})

# Quadratic
mae_quad = np.mean(np.abs(df['India'] - df['India_Quadratic']))
rmse_quad = np.sqrt(np.mean((df['India'] - df['India_Quadratic'])**2))
metrics_india.append({'Method': 'Quadratic Trend', 'MAE': mae_quad, 'RMSE': rmse_quad, 'R²': r2_india_quad})

# Exponential
mae_exp = np.mean(np.abs(df['India'] - df['India_Exponential']))
rmse_exp = np.sqrt(np.mean((df['India'] - df['India_Exponential'])**2))
metrics_india.append({'Method': 'Exponential Trend', 'MAE': mae_exp, 'RMSE': rmse_exp, 'R²': r2_india_exp})

# Holt's methods
for alpha, beta in alpha_beta_pairs:
    col_name = f'India_Holt_α{alpha}_β{beta}'
    mae_holt = np.mean(np.abs(df['India'] - df[col_name]))
    rmse_holt = np.sqrt(np.mean((df['India'] - df[col_name])**2))
    ss_res = np.sum((df['India'] - df[col_name])**2)
    r2_holt = 1 - (ss_res / ss_tot_india)
    metrics_india.append({'Method': f'Holt (α={alpha}, β={beta})', 'MAE': mae_holt, 'RMSE': rmse_holt, 'R²': r2_holt})

metrics_india_df = pd.DataFrame(metrics_india)
print(metrics_india_df.to_string(index=False))

best_india = metrics_india_df.loc[metrics_india_df['MAE'].idxmin()]
print(f"\n*** BEST METHOD (Lowest MAE): {best_india['Method']} ***")
print(f"MAE: {best_india['MAE']:.4f}, RMSE: {best_india['RMSE']:.4f}, R²: {best_india['R²']:.6f}")

print("\n" + "=" * 80)
print("MODEL EVALUATION - CHINA")
print("=" * 80)

metrics_china = []

# Linear
mae_linear_c = np.mean(np.abs(df['China'] - df['China_Linear']))
rmse_linear_c = np.sqrt(np.mean((df['China'] - df['China_Linear'])**2))
metrics_china.append({'Method': 'Linear Trend', 'MAE': mae_linear_c, 'RMSE': rmse_linear_c, 'R²': r2_china_linear})

# Quadratic
mae_quad_c = np.mean(np.abs(df['China'] - df['China_Quadratic']))
rmse_quad_c = np.sqrt(np.mean((df['China'] - df['China_Quadratic'])**2))
metrics_china.append({'Method': 'Quadratic Trend', 'MAE': mae_quad_c, 'RMSE': rmse_quad_c, 'R²': r2_china_quad})

# Exponential
mae_exp_c = np.mean(np.abs(df['China'] - df['China_Exponential']))
rmse_exp_c = np.sqrt(np.mean((df['China'] - df['China_Exponential'])**2))
metrics_china.append({'Method': 'Exponential Trend', 'MAE': mae_exp_c, 'RMSE': rmse_exp_c, 'R²': r2_china_exp})

# Holt's methods
for alpha, beta in alpha_beta_pairs:
    col_name = f'China_Holt_α{alpha}_β{beta}'
    mae_holt_c = np.mean(np.abs(df['China'] - df[col_name]))
    rmse_holt_c = np.sqrt(np.mean((df['China'] - df[col_name])**2))
    ss_res_c = np.sum((df['China'] - df[col_name])**2)
    r2_holt_c = 1 - (ss_res_c / ss_tot_china)
    metrics_china.append({'Method': f'Holt (α={alpha}, β={beta})', 'MAE': mae_holt_c, 'RMSE': rmse_holt_c, 'R²': r2_holt_c})

metrics_china_df = pd.DataFrame(metrics_china)
print(metrics_china_df.to_string(index=False))

best_china = metrics_china_df.loc[metrics_china_df['MAE'].idxmin()]
print(f"\n*** BEST METHOD (Lowest MAE): {best_china['Method']} ***")
print(f"MAE: {best_china['MAE']:.4f}, RMSE: {best_china['RMSE']:.4f}, R²: {best_china['R²']:.6f}")

# ============================================================================
# FORECASTING NEXT 5 YEARS (2020-2024)
# ============================================================================

print("\n" + "=" * 80)
print("FORECASTING NEXT 5 YEARS (2020-2024)")
print("=" * 80)

forecast_years = [2020, 2021, 2022, 2023, 2024]
forecast_time = [year - 1960 for year in forecast_years]

# Determine best model for each country
print(f"\nUsing best models:")
print(f"  India: {best_india['Method']}")
print(f"  China: {best_china['Method']}")

# Function to forecast based on model type
def forecast_population(method_name, country, forecast_time):
    forecasts = []
    
    if 'Linear' in method_name:
        if country == 'India':
            for t in forecast_time:
                forecasts.append(india_linear_poly(t))
        else:
            for t in forecast_time:
                forecasts.append(china_linear_poly(t))
    
    elif 'Quadratic' in method_name:
        if country == 'India':
            for t in forecast_time:
                forecasts.append(india_quad_poly(t))
        else:
            for t in forecast_time:
                forecasts.append(china_quad_poly(t))
    
    elif 'Exponential' in method_name:
        if country == 'India':
            for t in forecast_time:
                forecasts.append(np.exp(india_exp_coef[1]) * np.exp(india_exp_coef[0] * t))
        else:
            for t in forecast_time:
                forecasts.append(np.exp(china_exp_coef[1]) * np.exp(china_exp_coef[0] * t))
    
    elif 'Holt' in method_name:
        # Extract alpha and beta from method name
        import re
        match = re.search(r'α=([\d.]+), β=([\d.]+)', method_name)
        alpha = float(match.group(1))
        beta = float(match.group(2))
        
        if country == 'India':
            last_level = holt_results_india[f'Holt_α{alpha}_β{beta}']['level'][-1]
            last_trend = holt_results_india[f'Holt_α{alpha}_β{beta}']['trend'][-1]
        else:
            last_level = holt_results_china[f'Holt_α{alpha}_β{beta}']['level'][-1]
            last_trend = holt_results_china[f'Holt_α{alpha}_β{beta}']['trend'][-1]
        
        for h in range(1, len(forecast_time) + 1):
            forecasts.append(last_level + h * last_trend)
    
    return forecasts

india_forecasts = forecast_population(best_india['Method'], 'India', forecast_time)
china_forecasts = forecast_population(best_china['Method'], 'China', forecast_time)

print(f"\nIndia Population Forecast (millions):")
for year, pop in zip(forecast_years, india_forecasts):
    print(f"  {year}: {pop:.2f}")

print(f"\nChina Population Forecast (millions):")
for year, pop in zip(forecast_years, china_forecasts):
    print(f"  {year}: {pop:.2f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

# Plot 1: India - Actual vs Trend Models
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(df['Year'], df['India'], 'o-', label='Actual', linewidth=2, markersize=3)
plt.plot(df['Year'], df['India_Linear'], '--', label='Linear', linewidth=1.5)
plt.plot(df['Year'], df['India_Quadratic'], '--', label='Quadratic', linewidth=1.5)
plt.plot(df['Year'], df['India_Exponential'], '--', label='Exponential', linewidth=1.5)
plt.xlabel('Year', fontsize=11)
plt.ylabel('Population (millions)', fontsize=11)
plt.title('India: Actual vs Trend Models', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: China - Actual vs Trend Models
plt.subplot(2, 2, 2)
plt.plot(df['Year'], df['China'], 'o-', label='Actual', linewidth=2, markersize=3)
plt.plot(df['Year'], df['China_Linear'], '--', label='Linear', linewidth=1.5)
plt.plot(df['Year'], df['China_Quadratic'], '--', label='Quadratic', linewidth=1.5)
plt.plot(df['Year'], df['China_Exponential'], '--', label='Exponential', linewidth=1.5)
plt.xlabel('Year', fontsize=11)
plt.ylabel('Population (millions)', fontsize=11)
plt.title('China: Actual vs Trend Models', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: India - Actual vs Holt's Methods
plt.subplot(2, 2, 3)
plt.plot(df['Year'], df['India'], 'o-', label='Actual', linewidth=2, markersize=3)
for alpha, beta in alpha_beta_pairs:
    plt.plot(df['Year'], df[f'India_Holt_α{alpha}_β{beta}'], '--', 
             label=f'Holt (α={alpha}, β={beta})', linewidth=1.5)
plt.xlabel('Year', fontsize=11)
plt.ylabel('Population (millions)', fontsize=11)
plt.title("India: Actual vs Holt's Methods", fontsize=12, fontweight='bold')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# Plot 4: China - Actual vs Holt's Methods
plt.subplot(2, 2, 4)
plt.plot(df['Year'], df['China'], 'o-', label='Actual', linewidth=2, markersize=3)
for alpha, beta in alpha_beta_pairs:
    plt.plot(df['Year'], df[f'China_Holt_α{alpha}_β{beta}'], '--', 
             label=f'Holt (α={alpha}, β={beta})', linewidth=1.5)
plt.xlabel('Year', fontsize=11)
plt.ylabel('Population (millions)', fontsize=11)
plt.title("China: Actual vs Holt's Methods", fontsize=12, fontweight='bold')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/assignment4/india_china_models.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: india_china_models.png")

# Plot 5: Model Comparison - Error Metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# India
axes[0].barh(metrics_india_df['Method'], metrics_india_df['MAE'], color='steelblue')
axes[0].set_xlabel('Mean Absolute Error (MAE)', fontsize=11)
axes[0].set_title('India: Model Comparison (MAE)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# China
axes[1].barh(metrics_china_df['Method'], metrics_china_df['MAE'], color='coral')
axes[1].set_xlabel('Mean Absolute Error (MAE)', fontsize=11)
axes[1].set_title('China: Model Comparison (MAE)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/ubuntu/assignment4/india_china_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: india_china_comparison.png")

# Plot 6: Forecast visualization
plt.figure(figsize=(14, 6))

# India forecast
plt.subplot(1, 2, 1)
plt.plot(df['Year'], df['India'], 'o-', label='Historical', linewidth=2, markersize=4)
plt.plot(forecast_years, india_forecasts, 's-', label='Forecast', linewidth=2, markersize=6, color='red')
plt.axvline(x=2019, color='gray', linestyle='--', alpha=0.5, label='Forecast Start')
plt.xlabel('Year', fontsize=11)
plt.ylabel('Population (millions)', fontsize=11)
plt.title('India: Population Forecast (2020-2024)', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# China forecast
plt.subplot(1, 2, 2)
plt.plot(df['Year'], df['China'], 'o-', label='Historical', linewidth=2, markersize=4)
plt.plot(forecast_years, china_forecasts, 's-', label='Forecast', linewidth=2, markersize=6, color='red')
plt.axvline(x=2019, color='gray', linestyle='--', alpha=0.5, label='Forecast Start')
plt.xlabel('Year', fontsize=11)
plt.ylabel('Population (millions)', fontsize=11)
plt.title('China: Population Forecast (2020-2024)', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/assignment4/india_china_forecast.png', dpi=300, bbox_inches='tight')
print("✓ Saved: india_china_forecast.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
