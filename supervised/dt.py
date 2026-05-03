# ============================================================
#         DECISION TREE CLASSIFIER — SIMPLE COMPLETE CODE
# ============================================================

# STEP 1: IMPORT
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# STEP 2: LOAD DATASET
iris    = datasets.load_iris()
X       = iris.data
y       = iris.target

print(f"X Shape       : {X.shape}")
print(f"y Shape       : {y.shape}")
print(f"Feature Names : {iris.feature_names}")
print(f"Target Names  : {iris.target_names}")
print(f"Classes       : {np.unique(y)}")
print(f"Samples/Class : {np.bincount(y)}")

# STEP 3: SPLIT DATA
# Corrected Split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"\nTrain Samples : {len(X_train)}")
print(f"Test Samples  : {len(X_test)}")

# STEP 4: CREATE AND TRAIN MODEL
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# STEP 5: PREDICT
y_pred = dt.predict(X_test)

# STEP 6: EVALUATE
print(f"\nAccuracy          : {accuracy_score(y_test, y_pred):.4f}")
print(f"Tree Depth        : {dt.get_depth()}")
print(f"Number of Leaves  : {dt.get_n_leaves()}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print(f"Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# STEP 8: PREDICT NEW SAMPLE
new_flower  = [[5.1, 3.5, 1.4, 0.2]]
prediction  = dt.predict(new_flower)
probability = dt.predict_proba(new_flower)
print(f"\nNew Flower Prediction : {iris.target_names[prediction[0]]}")
print(f"Probabilities         : {probability[0]}")

# STEP 9: VISUALIZE TREE
plt.figure(figsize=(14, 6))
plot_tree(dt,feature_names=iris.feature_names,class_names=iris.target_names,filled=True,rounded=True)
plt.title("Decision Tree")
plt.show()

# STEP 10: PRINT TREE AS TEXT
print("\nTree Structure:")
print(export_text(dt, feature_names=iris.feature_names))