import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

home=datasets.fetch_california_housing()
df= pd.DataFrame(home.data, columns= home.feature_names)
df['price']=home.target
df.head()
df.info()
df.isnull().sum()

df.fillna(df.mean(numeric_only=True))
x=df.drop('price',axis=1)
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=43,test_size=0.3)
Scaler=StandardScaler()
x_train=Scaler.fit_transform(x_train)
x_test=Scaler.transform(x_test)

model=LinearRegression()
model.fit(x_train,y_train)

print(f"slope: {model.coef_[0]}")
print(f"intercept: {model.intercept_:.4f}")

pre=model.predict(x_test)
print(mean_squared_error(y_test,pre))
print(np.sqrt(mean_squared_error(y_test,pre)))
print(mean_absolute_error(y_test,pre))
print(r2_score(y_test,pre))

new_house  = [[3.5, 20.0, 5.0, 1.0, 800.0, 3.0, 37.0, -120.0]]
new_house=Scaler.transform(new_house)
pre2=model.predict(new_house)