import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
homes = pd.read_csv("real_estate.csv")
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]
model = LinearRegression()
model.fit(X, y)
print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
new_home = pd.DataFrame({"distance_100m": [5]})
prediction = model.predict(new_home)[0]
print(f"Predicted price at 500 m: {prediction:.2f} price units")
import matplotlib.pyplot as plt
feature = "distance_100m"
plt.figure(figsize=(8, 5))
plt.scatter(X[feature], y, alpha=0.6, label="Observed houses")
x_line = pd.DataFrame({feature: [X[feature].min(), X[feature].max()]})
y_line = model.predict(x_line)
plt.plot(x_line[feature], y_line, color="red", linewidth=2, label="Linear regression")
plt.xlabel("Distance to nearest MRT station (per 100 m)")
plt.ylabel("House price of unit area (10,000 NTD per ping)")
plt.title("Housing Price vs. Distance to MRT Station")
plt.legend()
plt.tight_layout()
plt.savefig("task1_housing.png", dpi=300)
plt.show()
