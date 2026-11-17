import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV, KFold, cross_val_score, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

concrete = pd.read_csv("/concrete_strength.csv").drop_duplicates()
target = "Concrete compressive strength"
X = concrete.drop(columns=target)
y = concrete[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"
baseline = LinearRegression()
baseline_scores = -cross_val_score(
    baseline, X_train, y_train, cv=folds, scoring=scoring
)
baseline_mean = baseline_scores.mean()
print("Baseline fold RMSEs:", baseline_scores.round(2))
print(f"Baseline mean CV RMSE: {baseline_mean} MPa")

ridge_pipeline = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge())
])

param_grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

search = GridSearchCV(
    ridge_pipeline,
    param_grid,
    cv=folds,
    scoring=scoring,
    refit=True,
    return_train_score=False
)
search.fit(X_train, y_train)

results = pd.DataFrame(search.cv_results_)
top_three = results.sort_values("rank_test_score").head(3)

print("\nThree highest-ranked parameter settings:")
for _, row in top_three.iterrows():
    print(
        f"Rank {int(row['rank_test_score'])}: "
        f"degree={row['param_poly__degree']}, "
        f"alpha={row['param_ridge__alpha']}, "
        f"mean CV RMSE={-row['mean_test_score']} MPa"
    )

print("\nBest parameters:", search.best_params_)
tuned_mean = -search.best_score_
print(f"Best tuned mean CV RMSE: {tuned_mean} MPa")

test_predictions = search.predict(X_test)
test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5
test_mae = mean_absolute_error(y_test, test_predictions)

print(f"Test RMSE: {test_rmse:.2f} MPa")
print(f"Test MAE: {test_mae:.2f} MPa")

# Actual values on x-axis; predicted values on y-axis
plt.figure(figsize=(7, 6))
plt.scatter(y_test, test_predictions, alpha=0.65, label="Test predictions")

low = min(y_test.min(), test_predictions.min())
high = max(y_test.max(), test_predictions.max())
plt.plot([low, high], [low, high], "r--", label="Ideal: y = x")

plt.xlabel("Actual concrete strength (MPa)")
plt.ylabel("Predicted concrete strength (MPa)")
plt.title("Actual vs. Predicted Concrete Strength")
plt.legend()
plt.tight_layout()
plt.savefig("task3_concrete.png", dpi=300)
plt.show()

print("\nComparison:")
print(f"Baseline mean CV RMSE = {baseline_mean} MPa")
print(f"Tuned mean CV RMSE = {tuned_mean} MPa")
