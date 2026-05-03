import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── LOAD ──
df = pd.read_csv('file.csv')        # ← change file name

# ── EDA ──
print(df.head())
print(df.isnull().sum())
df = df.fillna(df.mean(numeric_only=True))

# ── EXTRACT FEATURES (no y!) ──
X = df.iloc[:, [3, 4]].values      # ← change column numbers
# ── SCALE ──
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ── ELBOW METHOD (find best K) ──
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.show()
# look at graph → find elbow point → that is your K

# ── TRAIN ──
kmeans    = KMeans(n_clusters=5, init='k-means++', random_state=42)
# ↑ change 5 to whatever elbow method shows
y_predict = kmeans.fit_predict(X_scaled)

# ── VISUALIZE ──
colors = ['blue', 'green', 'red', 'black', 'purple']
for i in range(5):                  # ← change 5 to your K
    plt.scatter(X[y_predict==i, 0],X[y_predict==i, 1],s=100, c=colors[i],label=f'Cluster {i+1}')

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroid')
plt.title('Customer Clusters')
plt.legend()
plt.show()