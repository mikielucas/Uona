import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_excel('/home/ubuntu/upload/CollegeAdmission1.xlsx')

# Clean the data - remove the weird duplicate row
df = df[df['College'] != 'Math & ScienceMath & Science&']

# Prepare features
# Features: Gender, White, Asian, HSGPA, SAT/ACT, Edu_Parent1, Edu_Parent2
df['Gender_Numeric'] = df['Gender'].map({'M': 0, 'F': 1})
X = df[['Gender_Numeric', 'White', 'Asian', 'HSGPA', 'SAT/ACT', 'Edu_Parent1', 'Edu_Parent2']]
df['Admitted_Numeric'] = df['Admitted'].map({'No': 0, 'Yes': 1})

# Dictionary to store models and results
models = {}
results = {}

# Get unique colleges
colleges = df['College'].unique()

print("Building Decision Tree Models for College Admissions\n")
print("=" * 60)

for college in colleges:
    print(f"\n{college}")
    print("-" * 60)
    
    # Filter data for this college
    college_df = df[df['College'] == college].copy()
    X_college = college_df[['Gender_Numeric', 'White', 'Asian', 'HSGPA', 'SAT/ACT', 'Edu_Parent1', 'Edu_Parent2']]
    y_college = college_df['Admitted_Numeric']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_college, y_college, test_size=0.3, random_state=42)
    
    # Build decision tree
    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=50, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = dt_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store model and results
    models[college] = dt_model
    results[college] = {
        'accuracy': accuracy,
        'total_applicants': len(college_df),
        'admitted': sum(y_college),
        'admission_rate': sum(y_college) / len(y_college)
    }
    
    print(f"Total Applicants: {results[college]['total_applicants']}")
    print(f"Admitted: {results[college]['admitted']}")
    print(f"Admission Rate: {results[college]['admission_rate']:.2%}")
    print(f"Model Accuracy: {accuracy:.2%}")

print("\n" + "=" * 60)
print("\nSummary of Results:")
print("-" * 60)
for college in colleges:
    print(f"{college}: {results[college]['accuracy']:.2%} accuracy")

# Function to predict admission for a new applicant
def predict_admission(applicant_data):
    """
    Predict admission probability for each college
    applicant_data: dict with keys Gender, White, Asian, HSGPA, SAT/ACT, Edu_Parent1, Edu_Parent2
    """
    gender_num = 0 if applicant_data['Gender'] == 'M' else 1
    features = [[gender_num, applicant_data['White'], applicant_data['Asian'], 
                 applicant_data['HSGPA'], applicant_data['SAT/ACT'], 
                 applicant_data['Edu_Parent1'], applicant_data['Edu_Parent2']]]
    
    predictions = {}
    for college, model in models.items():
        prob = model.predict_proba(features)[0][1]  # Probability of admission (class 1)
        predictions[college] = prob
    
    # Find most likely college
    best_college = max(predictions, key=predictions.get)
    
    return predictions, best_college

# Example prediction
print("\n" + "=" * 60)
print("\nExample Prediction:")
print("-" * 60)
example_applicant = {
    'Gender': 'F',
    'White': 1,
    'Asian': 0,
    'HSGPA': 3.8,
    'SAT/ACT': 1350,
    'Edu_Parent1': 6,
    'Edu_Parent2': 6
}

predictions, best_college = predict_admission(example_applicant)
print(f"Applicant: Female, White, GPA: 3.8, SAT: 1350, Parents Edu: 6, 6")
print("\nAdmission Probabilities:")
for college, prob in predictions.items():
    print(f"  {college}: {prob:.2%}")
print(f"\nMost likely to be accepted: {best_college}")
