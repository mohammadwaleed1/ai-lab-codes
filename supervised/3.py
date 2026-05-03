import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_classes=2,        
    random_state=42
)

feature_names = ['total_spending', 'age', 'num_visits', 'purchase_freq']

df              = pd.DataFrame(X, columns=feature_names)
df['customer']  = y
print(df.head())
df = df.fillna(df.mean(numeric_only=True))
x = df.drop('customer', axis=1)
y = df['customer']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)
scaler  = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(x_train, y_train)
pre_svm = svm.predict(x_test)
acc_svm = accuracy_score(y_test, pre_svm)

print(f"Accuracy : {acc_svm*100:.2f}%")
print(classification_report(y_test, pre_svm,
      target_names=['Low Value', 'High Value']))
print(confusion_matrix(y_test, pre_svm))

dt = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt.fit(x_train, y_train)
pre_dt = dt.predict(x_test)
acc_dt = accuracy_score(y_test, pre_dt)

print(f"Accuracy : {acc_dt*100:.2f}%")
print(classification_report(y_test, pre_dt,
      target_names=['Low Value', 'High Value']))
print(confusion_matrix(y_test, pre_dt))

print("\nDecision Tree Rules:")
print(export_text(dt, feature_names=feature_names))

# STEP 9: VISUALIZE TREE
plt.figure(figsize=(14, 6))
plot_tree(dt, feature_names=feature_names, class_names=['Low Value', 'High Value'], filled=True, rounded=True)
plt.show()

new_customer = [[5000, 35, 20, 15]]
new_customer = scaler.transform(new_customer)

pred_svm  = svm.predict(new_customer)
prob_svm  = svm.predict_proba(new_customer)
print(f"SVM Prediction  : {'High Value' if pred_svm[0]==1 else 'Low Value'}")
print(f"Probabilities   : Low={prob_svm[0][0]:.2f}  High={prob_svm[0][1]:.2f}")

pred_dt   = dt.predict(new_customer)
prob_dt   = dt.predict_proba(new_customer)
print(f"\nDT  Prediction  : {'High Value' if pred_dt[0]==1 else 'Low Value'}")
print(f"Probabilities   : Low={prob_dt[0][0]:.2f}  High={prob_dt[0][1]:.2f}")