
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


# STEP 2: LOAD DATASET

iris = datasets.load_iris()

X = iris.data        
y = iris.target      


print(f"Feature Names  : {iris.feature_names}")
print(f"Target Names   : {iris.target_names}")
print(f"Data Shape     : {X.shape}  → (samples, features)")
print(f"Target Shape   : {y.shape}  → (samples,)")
print(f"Classes        : {np.unique(y)} → {iris.target_names}")
print(f"Samples/Class  : {np.bincount(y)}")  # How many per class


# STEP 4: BINARY CLASSIFICATION
y_binary = (y == 0).astype(int)   # 1 = Setosa, 0 = Not Setosa
print(f"Original Target  : {np.unique(y)}       → 3 classes")
print(f"Binary Target    : {np.unique(y_binary)} → 2 classes")
print(f"Class counts     : {np.bincount(y_binary)} (0=NotSetosa, 1=Setosa)")


# STEP 5: SPLIT DATA (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary,
    test_size=0.3,      # 30% for testing
    random_state=42     # Fixed shuffle for reproducibility
)

print(f"Total Samples   : {len(X)}")
print(f"Training Samples: {len(X_train)}  (70%)")
print(f"Testing Samples : {len(X_test)}   (30%)")
print(f"\nX_train shape   : {X_train.shape}")
print(f"X_test shape    : {X_test.shape}")
print(f"y_train shape   : {y_train.shape}")
print(f"y_test shape    : {y_test.shape}")


# STEP 6: TRAIN ALL 3 KERNELS
#         Linear, Polynomial, RBF

# ── Kernel 1: Linear ──
svm_linear = SVC(
    kernel='linear',
    C=1.0,
    probability=True,       # Regularization
)
svm_linear.fit(X_train, y_train)
pred_linear = svm_linear.predict(X_test)
acc_linear = accuracy_score(y_test, pred_linear)
print(f"\n1. LINEAR Kernel")
print(f"   Arguments      : kernel='linear', C=1.0")
print(f"   Accuracy       : {acc_linear:.4f} ({acc_linear*100:.2f}%)")

# ── Kernel 2: Polynomial ──
svm_poly = SVC(
    kernel='poly',
    C=1.0,          # Regularization
    degree=3,       # Degree of polynomial curve
    gamma='scale',  # Influence of each point
    coef0=0.0,      # Independent term
    probability=True,  
)
svm_poly.fit(X_train, y_train)
pred_poly = svm_poly.predict(X_test)
acc_poly = accuracy_score(y_test, pred_poly)
#-----to find prob of sample
new_flower  = [[5.1, 3.5, 1.4, 0.2]]
prediction=svm_poly.predict(new_flower)
probability=svm_poly.predict_proba(new_flower)
print(prediction[0])
print(probability[0])
#------------------------
# ── Kernel 3: RBF ──
svm_rbf = SVC(
    kernel='rbf',
    C=1.0,          # Regularization
    gamma='scale',  # Auto-calculated gamma
    probability=True,  
)
svm_rbf.fit(X_train, y_train)
pred_rbf = svm_rbf.predict(X_test)
acc_rbf = accuracy_score(y_test, pred_rbf)
print(f"   Accuracy       : {acc_rbf:.4f} ({acc_rbf*100:.2f}%)")
for i in range(5):
    print(f"  Actual: {y_test[i]}   Predicted: {pred_rbf[i]:.1f}")

