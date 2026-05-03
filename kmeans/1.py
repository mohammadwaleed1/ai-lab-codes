import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Mall_Customers.csv')

print("=" * 55)
print("              EDA")
print("=" * 55)
print(df.head())
print(f"\nColumns : {df.columns.tolist()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
df = df.fillna(df.mean(numeric_only=True))

X = df.drop('CustomerID', axis=1)


X = pd.get_dummies(X, drop_first=True)
print(f"\nFeatures Used : {X.columns.tolist()}")
print(f"X Shape       : {X.shape}")

wcss_no_scale = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss_no_scale.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss_no_scale)
plt.title('Elbow Method')
plt.show()
kmeans_no_scale    = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred_no_scale    = kmeans_no_scale.fit_predict(X)


X_age          = X[['Age']]                    # age column kept as is
X_to_scale     = X.drop('Age', axis=1)         # all other columns
scaler         = StandardScaler()
X_scaled       = scaler.fit_transform(X_to_scale)
X_final = np.hstack([X_age.values, X_scaled])

wcss_scaled = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_final)
    wcss_scaled.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss_scaled)
plt.title('Elbow Method')
plt.show()
kmeans_scaled  = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred_scaled  = kmeans_scaled.fit_predict(X_final)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("K-Means Clustering Comparison", fontsize=14, fontweight='bold')

colors = ['blue', 'green', 'red', 'black', 'purple']

axes[0].plot(range(1, 11), wcss_no_scale,
             marker='o', label='Without Scaling', color='red')
axes[0].plot(range(1, 11), wcss_scaled,
             marker='s', label='With Scaling', color='blue')
axes[0].set_title("Elbow Method Comparison")
axes[0].set_xlabel("Number of Clusters")
axes[0].set_ylabel("WCSS")
axes[0].legend()
axes[0].grid(True)

X_arr = X.values
for i in range(5):
    axes[1].scatter(X_arr[y_pred_no_scale==i, 2],
                    X_arr[y_pred_no_scale==i, 3],
                    s=80, c=colors[i], label=f'Cluster {i+1}')
axes[1].set_title("Without Scaling")
axes[1].set_xlabel("Annual Income")
axes[1].set_ylabel("Spending Score")
axes[1].legend()

X_final_arr = X_final.values
for i in range(5):
    axes[2].scatter(X_final_arr[y_pred_scaled==i, 1],
                    X_final_arr[y_pred_scaled==i, 2],
                    s=80, c=colors[i], label=f'Cluster {i+1}')
axes[2].set_title("With Scaling (except Age)")
axes[2].set_xlabel("Annual Income (scaled)")
axes[2].set_ylabel("Spending Score (scaled)")
axes[2].legend()

plt.tight_layout()
plt.show()

