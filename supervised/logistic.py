import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression       # ← changed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # ← changed

# ── Dataset ──
iris = datasets.load_iris()                              # ← changed dataset
X = iris.data
y = iris.target

# print("=" * 55)
# print("         DATASET INFORMATION")
# print("=" * 55)
# print(f"Feature Names : {iris.feature_names}")
# print(f"Target Names  : {iris.target_names}")
# print(f"X Shape       : {X.shape}")
# print(f"y Shape       : {y.shape}")
# print(f"Classes       : {np.unique(y)}")
# print(f"Samples/Class : {np.bincount(y)}")

# ── Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)
print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# ── Train Model ──
model = LogisticRegression()                 # ← changed
model.fit(X_train, y_train)

# ── Predict ──
y_pred = model.predict(X_test)

# ── Evaluate ──   (all different from Linear Regression)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {acc:.4f} ({acc*100:.1f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print(f"Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Probability (unique to Logistic) ──
proba = model.predict_proba(X_test)
print(f"\nFirst 3 Predictions with Probabilities:")
for i in range(3):
    print(f"  Predicted: {y_pred[i]} Probabilities: {proba[i]}")

# ── Predict New Sample ──
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction  = model.predict(new_flower)
probability = model.predict_proba(new_flower)
print(f"\nNew Flower Prediction : {iris.target_names[prediction[0]]}")
print(f"Probabilities         : {probability[0]}")