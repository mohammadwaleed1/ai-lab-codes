import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── LOAD DATASET ──
X, y = make_classification(
    n_samples=1000,     # 1000 emails
    n_features=10,      # 10 features (word freq, length, links etc)
    n_classes=2,        # 2 classes → spam or not spam
    random_state=42
)

# ── CONVERT TO DATAFRAME FOR EDA ──
feature_names = ['word_freq', 'email_length', 'num_links',
                 'sender_score', 'capital_ratio', 'exclamation',
                 'dollar_signs', 'reply_to', 'html_content', 'attachments']

df      = pd.DataFrame(X, columns=feature_names)
df['spam'] = y
df.head()
df.isnull().sum()
df.fillna(df.mean(numeric_only=True))

x=df.drop('spam',axis=1)
y=df['spam']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=LogisticRegression()
model.fit(x_train,y_train)

pre=model.predict(x_test)
acc=accuracy_score(y_test,pre)
print(f"accuracy:  {acc*100}")
print(classification_report(y_test, pre, target_names=['not spam','spam']))
print(confusion_matrix(y_test,pre))
new_email = [[0.9, 500, 15, 0.1, 0.9, 10, 8, 0, 1, 3]]
new_email=scaler.transform(new_email)
prediction=model.predict(new_email)
prob=model.predict_proba(new_email)


print(f"Prediction    : {'SPAM' if prediction[0]==1 else 'NOT SPAM'}")
print(f"Probabilities : Not Spam={prob[0][0]:.2f}  Spam={prob[0][1]:.2f}")
