import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    TweedieRegressor,
    SGDRegressor,
    LogisticRegression  # Note: LogisticRegression is a classifier, included for completeness
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Load your dataset
df = pd.read_csv('your_dataset.csv')  # Replace with your actual CSV file

# Define features and target
X = df.drop('target_column', axis=1)  # Replace 'target_column' with your actual target
y = df['target_column']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define regression models
models = {
    'Linear Regression': LinearRegression(),
    'Huber Regressor': HuberRegressor(),
    'Tweedie Regressor': TweedieRegressor(power=1, alpha=0.5),  # power=1 → Poisson-like
    'SGD Regressor': SGDRegressor(max_iter=1000, tol=1e-3),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Support Vector Regressor': SVR(),
    'Logistic Regression (for regression use not recommended)': LogisticRegression()  # Used here for illustration only
}

# Evaluate each model
results = []

print("Model Performance:\n")

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, scoring='r2', cv=5)

        print(f"{name}")
        print(f"  R² Score       : {r2:.4f}")
        print(f"  MSE            : {mse:.4f}")
        print(f"  CV R² Mean     : {cv_scores.mean():.4f}")
        print(f"  CV R² Std Dev  : {cv_scores.std():.4f}\n")

        results.append({
            'Model': name,
            'MSE': mse,
            'R2 Score': r2,
            'CV R2 Mean': cv_scores.mean(),
            'CV R2 Std': cv_scores.std()
        })
    except Exception as e:
        print(f"{name} failed: {e}\n")

# Show comparison as a sorted DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='R2 Score', ascending=False)

print("\nModel Comparison Table:")
print(results_df.to_string(index=False))
