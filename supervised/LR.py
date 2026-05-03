
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

diabetes = datasets.load_diabetes()

X = diabetes.data    # 10 medical measurements (442 patients)
y = diabetes.target  # disease progression score (a number)

print("=" * 60)
print("           DATASET INFORMATION")
print("=" * 60)
print(f"Feature Names  : {diabetes.feature_names}")
print(f"X Shape        : {X.shape}")    
print(f"y Shape        : {y.shape}")   
print(f"y Min          : {y.min()}")    
print(f"y Max          : {y.max()}")    
print(f"y Mean         : {y.mean():.2f}")  

# ─────────────────────────────────────────
# STEP 3: SIMPLE LINEAR REGRESSION
#         (Only 1 Feature — BMI)
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("       SIMPLE LINEAR REGRESSION (1 Feature)")
print("=" * 60)

X_simple = X[:, 2].reshape(-1, 1)  # shape (442,) → (442, 1)
print(f"X_simple shape : {X_simple.shape}")  # (442, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y,
    test_size=0.3,
    random_state=42
)
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

print(f"\nSlope (m)      : {model_simple.coef_[0]:.4f}")
print(f"Intercept (b)  : {model_simple.intercept_:.4f}")
print(f"Equation       : y = {model_simple.coef_[0]:.2f} * BMI + {model_simple.intercept_:.2f}")

y_pred_simple = model_simple.predict(X_test)

mse  = mean_squared_error(y_test, y_pred_simple)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred_simple)
r2   = r2_score(y_test, y_pred_simple)

print(f"\nMSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R2   : {r2:.4f}  → model explains {r2*100:.1f}% of variation")

print(f"\nActual vs Predicted (first 5):")
for i in range(5):
    print(f"  Actual: {y_test[i]:.1f}   Predicted: {y_pred_simple[i]:.1f}")

