"""
Case Study Assignment 4 - Report 10.1: Fried Dough Analysis
Author: Statistical Analysis
Date: November 2025

This script analyzes the Fried Dough dataset using various forecasting methods:
- Moving Averages (3-day and 5-day)
- Simple Exponential Smoothing
- Comparison of forecast accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Using numpy for error calculations instead of sklearn

# Load the dataset
df = pd.read_excel('/home/ubuntu/upload/FriedDoughDataset.xlsx')

print("=" * 80)
print("FRIED DOUGH SALES FORECASTING ANALYSIS")
print("=" * 80)
print("\nDataset Overview:")
print(df.head(10))
print(f"\nDataset Shape: {df.shape}")
print(f"\nDescriptive Statistics:")
print(df.describe())

# ============================================================================
# MOVING AVERAGE METHODS
# ============================================================================

def moving_average(data, window):
    """Calculate moving average with specified window size"""
    return data.rolling(window=window).mean()

# Calculate 3-day and 5-day moving averages for Fried Dough
df['FD_MA3'] = moving_average(df['Fried Dough'], 3)
df['FD_MA5'] = moving_average(df['Fried Dough'], 5)

# Calculate 3-day and 5-day moving averages for Soft Drinks
df['SD_MA3'] = moving_average(df['Soft Drinks'], 3)
df['SD_MA5'] = moving_average(df['Soft Drinks'], 5)

print("\n" + "=" * 80)
print("MOVING AVERAGE RESULTS")
print("=" * 80)
print(df[['Day', 'Fried Dough', 'FD_MA3', 'FD_MA5', 'Soft Drinks', 'SD_MA3', 'SD_MA5']])

# ============================================================================
# SIMPLE EXPONENTIAL SMOOTHING
# ============================================================================

def simple_exponential_smoothing(data, alpha):
    """
    Simple Exponential Smoothing
    S_t = alpha * Y_t + (1 - alpha) * S_(t-1)
    """
    result = [data.iloc[0]]  # Initialize with first value
    for i in range(1, len(data)):
        result.append(alpha * data.iloc[i] + (1 - alpha) * result[i-1])
    return result

# Test different alpha values for Fried Dough
alphas = [0.2, 0.3, 0.5, 0.7]
ses_results_fd = {}

for alpha in alphas:
    ses_results_fd[f'FD_SES_{alpha}'] = simple_exponential_smoothing(df['Fried Dough'], alpha)
    df[f'FD_SES_{alpha}'] = ses_results_fd[f'FD_SES_{alpha}']

# Test different alpha values for Soft Drinks
ses_results_sd = {}
for alpha in alphas:
    ses_results_sd[f'SD_SES_{alpha}'] = simple_exponential_smoothing(df['Soft Drinks'], alpha)
    df[f'SD_SES_{alpha}'] = ses_results_sd[f'SD_SES_{alpha}']

print("\n" + "=" * 80)
print("SIMPLE EXPONENTIAL SMOOTHING RESULTS (Alpha = 0.2, 0.3, 0.5, 0.7)")
print("=" * 80)
print(df[['Day', 'Fried Dough', 'FD_SES_0.2', 'FD_SES_0.3', 'FD_SES_0.5', 'FD_SES_0.7']].head(15))

# ============================================================================
# MODEL EVALUATION - Calculate Error Metrics
# ============================================================================

print("\n" + "=" * 80)
print("MODEL EVALUATION - FRIED DOUGH")
print("=" * 80)

# For Moving Averages, we need to exclude NaN values
ma3_valid = df.dropna(subset=['FD_MA3'])
ma5_valid = df.dropna(subset=['FD_MA5'])

# Calculate MAE and RMSE for each method
metrics = []

# MA3
mae_ma3 = np.mean(np.abs(ma3_valid['Fried Dough'] - ma3_valid['FD_MA3']))
rmse_ma3 = np.sqrt(np.mean((ma3_valid['Fried Dough'] - ma3_valid['FD_MA3'])**2))
metrics.append({'Method': '3-Day MA', 'MAE': mae_ma3, 'RMSE': rmse_ma3})

# MA5
mae_ma5 = np.mean(np.abs(ma5_valid['Fried Dough'] - ma5_valid['FD_MA5']))
rmse_ma5 = np.sqrt(np.mean((ma5_valid['Fried Dough'] - ma5_valid['FD_MA5'])**2))
metrics.append({'Method': '5-Day MA', 'MAE': mae_ma5, 'RMSE': rmse_ma5})

# SES with different alphas
for alpha in alphas:
    mae = np.mean(np.abs(df['Fried Dough'] - df[f'FD_SES_{alpha}']))
    rmse = np.sqrt(np.mean((df['Fried Dough'] - df[f'FD_SES_{alpha}'])**2))
    metrics.append({'Method': f'SES (α={alpha})', 'MAE': mae, 'RMSE': rmse})

metrics_df = pd.DataFrame(metrics)
print(metrics_df.to_string(index=False))

# Find best method
best_method = metrics_df.loc[metrics_df['MAE'].idxmin()]
print(f"\n*** BEST METHOD (Lowest MAE): {best_method['Method']} ***")
print(f"MAE: {best_method['MAE']:.4f}, RMSE: {best_method['RMSE']:.4f}")

# ============================================================================
# MODEL EVALUATION - SOFT DRINKS
# ============================================================================

print("\n" + "=" * 80)
print("MODEL EVALUATION - SOFT DRINKS")
print("=" * 80)

metrics_sd = []

# MA3
mae_ma3_sd = np.mean(np.abs(ma3_valid['Soft Drinks'] - ma3_valid['SD_MA3']))
rmse_ma3_sd = np.sqrt(np.mean((ma3_valid['Soft Drinks'] - ma3_valid['SD_MA3'])**2))
metrics_sd.append({'Method': '3-Day MA', 'MAE': mae_ma3_sd, 'RMSE': rmse_ma3_sd})

# MA5
mae_ma5_sd = np.mean(np.abs(ma5_valid['Soft Drinks'] - ma5_valid['SD_MA5']))
rmse_ma5_sd = np.sqrt(np.mean((ma5_valid['Soft Drinks'] - ma5_valid['SD_MA5'])**2))
metrics_sd.append({'Method': '5-Day MA', 'MAE': mae_ma5_sd, 'RMSE': rmse_ma5_sd})

# SES with different alphas
for alpha in alphas:
    mae_sd = np.mean(np.abs(df['Soft Drinks'] - df[f'SD_SES_{alpha}']))
    rmse_sd = np.sqrt(np.mean((df['Soft Drinks'] - df[f'SD_SES_{alpha}'])**2))
    metrics_sd.append({'Method': f'SES (α={alpha})', 'MAE': mae_sd, 'RMSE': rmse_sd})

metrics_sd_df = pd.DataFrame(metrics_sd)
print(metrics_sd_df.to_string(index=False))

# Find best method for soft drinks
best_method_sd = metrics_sd_df.loc[metrics_sd_df['MAE'].idxmin()]
print(f"\n*** BEST METHOD (Lowest MAE): {best_method_sd['Method']} ***")
print(f"MAE: {best_method_sd['MAE']:.4f}, RMSE: {best_method_sd['RMSE']:.4f}")

# ============================================================================
# FORECASTING NEXT FEW DAYS
# ============================================================================

print("\n" + "=" * 80)
print("FORECASTING NEXT FEW DAYS")
print("=" * 80)

# Determine the best alpha for SES (based on lowest MAE)
best_alpha_fd = float(best_method['Method'].split('α=')[1].rstrip(')'))
best_alpha_sd = float(best_method_sd['Method'].split('α=')[1].rstrip(')'))

print(f"\nUsing best method for forecasting:")
print(f"  Fried Dough: {best_method['Method']}")
print(f"  Soft Drinks: {best_method_sd['Method']}")

# Forecast next 5 days using the best SES model
forecast_days = 5
last_value_fd = df[f'FD_SES_{best_alpha_fd}'].iloc[-1]
last_value_sd = df[f'SD_SES_{best_alpha_sd}'].iloc[-1]

print(f"\nFried Dough Forecast (Next {forecast_days} days):")
for day in range(1, forecast_days + 1):
    print(f"  Day {30 + day}: {last_value_fd:.2f} plates")

print(f"\nSoft Drinks Forecast (Next {forecast_days} days):")
for day in range(1, forecast_days + 1):
    print(f"  Day {30 + day}: {last_value_sd:.2f} drinks")

# ============================================================================
# VISUALIZATION
# ============================================================================

# Plot 1: Fried Dough - Actual vs Moving Averages
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(df['Day'], df['Fried Dough'], 'o-', label='Actual', linewidth=2, markersize=4)
plt.plot(df['Day'], df['FD_MA3'], 's-', label='3-Day MA', linewidth=1.5, markersize=3)
plt.plot(df['Day'], df['FD_MA5'], '^-', label='5-Day MA', linewidth=1.5, markersize=3)
plt.xlabel('Day', fontsize=11)
plt.ylabel('Fried Dough Sales (plates)', fontsize=11)
plt.title('Fried Dough: Actual vs Moving Averages', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Soft Drinks - Actual vs Moving Averages
plt.subplot(1, 2, 2)
plt.plot(df['Day'], df['Soft Drinks'], 'o-', label='Actual', linewidth=2, markersize=4)
plt.plot(df['Day'], df['SD_MA3'], 's-', label='3-Day MA', linewidth=1.5, markersize=3)
plt.plot(df['Day'], df['SD_MA5'], '^-', label='5-Day MA', linewidth=1.5, markersize=3)
plt.xlabel('Day', fontsize=11)
plt.ylabel('Soft Drinks Sales', fontsize=11)
plt.title('Soft Drinks: Actual vs Moving Averages', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/assignment4/fried_dough_moving_averages.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: fried_dough_moving_averages.png")

# Plot 3: Fried Dough - Actual vs Exponential Smoothing
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(df['Day'], df['Fried Dough'], 'o-', label='Actual', linewidth=2, markersize=4)
for alpha in alphas:
    plt.plot(df['Day'], df[f'FD_SES_{alpha}'], label=f'SES (α={alpha})', linewidth=1.5)
plt.xlabel('Day', fontsize=11)
plt.ylabel('Fried Dough Sales (plates)', fontsize=11)
plt.title('Fried Dough: Actual vs Exponential Smoothing', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Soft Drinks - Actual vs Exponential Smoothing
plt.subplot(1, 2, 2)
plt.plot(df['Day'], df['Soft Drinks'], 'o-', label='Actual', linewidth=2, markersize=4)
for alpha in alphas:
    plt.plot(df['Day'], df[f'SD_SES_{alpha}'], label=f'SES (α={alpha})', linewidth=1.5)
plt.xlabel('Day', fontsize=11)
plt.ylabel('Soft Drinks Sales', fontsize=11)
plt.title('Soft Drinks: Actual vs Exponential Smoothing', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/assignment4/fried_dough_exponential_smoothing.png', dpi=300, bbox_inches='tight')
print("✓ Saved: fried_dough_exponential_smoothing.png")

# Plot 5: Model Comparison - Error Metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.barh(metrics_df['Method'], metrics_df['MAE'], color='steelblue')
plt.xlabel('Mean Absolute Error (MAE)', fontsize=11)
plt.title('Fried Dough: Model Comparison (MAE)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

plt.subplot(1, 2, 2)
plt.barh(metrics_sd_df['Method'], metrics_sd_df['MAE'], color='coral')
plt.xlabel('Mean Absolute Error (MAE)', fontsize=11)
plt.title('Soft Drinks: Model Comparison (MAE)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/ubuntu/assignment4/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
