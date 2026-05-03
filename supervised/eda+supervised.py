# ============================================================
#     EDA + SUPERVISED LEARNING — COMBINED SIMPLE CODE
# ============================================================

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────
iris    = datasets.load_iris()
df      = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
# converting sklearn dataset to pandas DataFrame
# so we can use pandas EDA functions on it

# ─────────────────────────────────────────
# STEP 2: EDA (Exploratory Data Analysis)
# ─────────────────────────────────────────
print("=" * 55)
print("              EDA")
print("=" * 55)

print("\n--- First 5 Rows ---")
print(df.head())
# quick look at data structure and values

print("\n--- Data Info ---")
print(df.info())
# column names, data types, non-null counts

print("\n--- Missing Values ---")
print(df.isnull().sum())
# count missing values per column

print("\n--- Basic Statistics ---")
print(df.describe())
# mean, min, max, std for each column

print("\n--- Class Distribution ---")
print(df['target'].value_counts())
# how many samples per class
# checks if data is balanced or imbalanced

# ─────────────────────────────────────────
# STEP 3: HANDLE MISSING VALUES
# ─────────────────────────────────────────
df = df.fillna(df.mean(numeric_only=True))
# fill missing values with column mean
# iris has no missing values but this is good practice
# always include this for real datasets

# ─────────────────────────────────────────
# STEP 4: SEPARATE X AND y
# ─────────────────────────────────────────
X = df.drop('target', axis=1)   # all columns except target
y = df['target']                 # only target column

print("\n" + "=" * 55)
print("         AFTER EDA — DATA READY")
print("=" * 55)
print(f"X Shape : {X.shape}")
print(f"y Shape : {y.shape}")

# ─────────────────────────────────────────
# STEP 5: TRAIN TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"Train   : {len(X_train)} samples")
print(f"Test    : {len(X_test)} samples")

# ─────────────────────────────────────────
# STEP 6: FEATURE SCALING
# ─────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
# fit_transform on train → learns mean and std then scales
# transform on test      → uses same mean and std to scale

# ─────────────────────────────────────────
# STEP 7: TRAIN MODEL (SVM)
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("         SUPERVISED LEARNING (SVM)")
print("=" * 55)

model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_train, y_train)

# ─────────────────────────────────────────
# STEP 8: PREDICT AND EVALUATE
# ─────────────────────────────────────────
y_pred = model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ─────────────────────────────────────────
# STEP 9: PREDICT NEW SAMPLE
# ─────────────────────────────────────────
new_flower  = [[5.1, 3.5, 1.4, 0.2]]
new_scaled  = scaler.transform(new_flower)
# IMPORTANT: scale new sample using SAME scaler
# never use raw values after scaling training data

prediction  = model.predict(new_scaled)
probability = model.predict_proba(new_scaled)
print(f"New Flower Prediction : {iris.target_names[prediction[0]]}")
print(f"Probabilities         : {probability[0]}")