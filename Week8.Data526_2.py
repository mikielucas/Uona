import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_excel('/home/ubuntu/upload/HousePrice.xlsx')

# Prepare features
# Features: Beds, Baths, Sqft_home, Sqft_lot, Build_year
X = df[['Beds', 'Baths', 'Sqft_home', 'Sqft_lot', 'Build_year']]
y = df['Sale_amount']

# Dictionary to store models and results
models = {}
results = {}

# Get unique universities (campuses)
universities = df['University'].value_counts()
# Only use universities with at least 50 samples for reliable models
universities = universities[universities >= 50].index.tolist()

print("Building Decision Tree Models for House Price Prediction\n")
print("=" * 80)
print(f"\nAnalyzing {len(universities)} campuses with sufficient data (50+ houses)\n")

for university in universities:
    print(f"\n{university}")
    print("-" * 80)
    
    # Filter data for this university
    uni_df = df[df['University'] == university].copy()
    X_uni = uni_df[['Beds', 'Baths', 'Sqft_home', 'Sqft_lot', 'Build_year']]
    y_uni = uni_df['Sale_amount']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_uni, y_uni, test_size=0.3, random_state=42)
    
    # Build decision tree
    dt_model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = dt_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store model and results
    models[university] = dt_model
    results[university] = {
        'total_houses': len(uni_df),
        'avg_price': y_uni.mean(),
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    print(f"Total Houses: {results[university]['total_houses']}")
    print(f"Average Price: ${results[university]['avg_price']:,.0f}")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"MAE: ${mae:,.0f}")
    print(f"R² Score: {r2:.3f}")

print("\n" + "=" * 80)
print("\nComparison of Models (Top 10 by R² Score):")
print("-" * 80)

# Sort by R² score
sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)

print(f"\n{'University':<50} {'R² Score':<12} {'RMSE':<15} {'Avg Price':<15}")
print("-" * 80)
for university, metrics in sorted_results[:10]:
    print(f"{university:<50} {metrics['r2']:<12.3f} ${metrics['rmse']:<14,.0f} ${metrics['avg_price']:<14,.0f}")

print("\n" + "=" * 80)
print("\nBottom 5 Models by R² Score:")
print("-" * 80)
print(f"\n{'University':<50} {'R² Score':<12} {'RMSE':<15} {'Avg Price':<15}")
print("-" * 80)
for university, metrics in sorted_results[-5:]:
    print(f"{university:<50} {metrics['r2']:<12.3f} ${metrics['rmse']:<14,.0f} ${metrics['avg_price']:<14,.0f}")

# Function to predict house price for a new house
def predict_price(house_data, university):
    """
    Predict house price for a specific university
    house_data: dict with keys Beds, Baths, Sqft_home, Sqft_lot, Build_year
    university: name of the university
    """
    if university not in models:
        return None
    
    features = [[house_data['Beds'], house_data['Baths'], house_data['Sqft_home'], 
                 house_data['Sqft_lot'], house_data['Build_year']]]
    
    predicted_price = models[university].predict(features)[0]
    
    return predicted_price

# Example prediction
print("\n" + "=" * 80)
print("\nExample Prediction:")
print("-" * 80)
example_house = {
    'Beds': 3,
    'Baths': 2.0,
    'Sqft_home': 1800,
    'Sqft_lot': 10000,
    'Build_year': 1990
}

print(f"House: {example_house['Beds']} beds, {example_house['Baths']} baths, "
      f"{example_house['Sqft_home']} sqft home, {example_house['Sqft_lot']} sqft lot, built {example_house['Build_year']}")
print("\nPredicted Prices (Top 5 Universities):")
for university, _ in sorted_results[:5]:
    price = predict_price(example_house, university)
    if price:
        print(f"  {university}: ${price:,.0f}")
