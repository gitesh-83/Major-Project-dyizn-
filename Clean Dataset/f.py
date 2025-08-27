import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Load the cleaned dataset
df = pd.read_csv("titanic_cleaned.csv")

# ---------------------------
# 1. CLASSIFICATION MODEL
# ---------------------------
# Features (exclude Survived, PassengerId)
X = df.drop(columns=["Survived", "PassengerId"])
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

print("🔹 Classification Results (Survival):")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------------------
# 2. REGRESSION MODEL
# ---------------------------
# Example: Predict Fare using other features
X_reg = df.drop(columns=["Fare", "PassengerId"])
y_reg = df["Fare"]

# Train-test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Linear Regression model
reg = LinearRegression()
reg.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_reg = reg.predict(X_test_reg)

print("\n🔹 Regression Results (Fare):")
print("MSE:", mean_squared_error(y_test_reg, y_pred_reg))
print("R² Score:", r2_score(y_test_reg, y_pred_reg))
