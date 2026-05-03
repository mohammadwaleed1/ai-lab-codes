# =========================
# ML PREPROCESSING TEMPLATE
# Use for: classification / regression datasets
# =========================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
df = pd.read_csv("your_file.csv")

# 2. BASIC DATA CHECK
print(df.head())
print(df.info())
print(df.isnull().sum())

# 3. HANDLE MISSING VALUES
# Option 1: remove rows with missing values (simple but loses data)
df = df.dropna()

# Option 2 (alternative): fill missing numeric values with mean
# df = df.fillna(df.mean(numeric_only=True))

# 4. ENCODE CATEGORICAL COLUMNS (text → numbers)
# safer approach: one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# 5. SPLIT FEATURES AND TARGET
target_column = "target_column"   # <-- change this

X = df.drop(target_column, axis=1)
y = df[target_column]

# 6. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 7. FEATURE SCALING (important for distance-based models)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)  # learn + transform
X_test = scaler.transform(X_test)        # only transform

# =========================
# DONE: READY FOR MODEL TRAINING
# =========================