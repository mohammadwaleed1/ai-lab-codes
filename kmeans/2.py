import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data = {
    'vehicle_serial_no': [5, 3, 8, 2, 4, 7, 6, 10, 1, 9],
    'mileage':           [150000, 120000, 250000, 80000, 100000,
                          220000, 180000, 300000, 75000, 280000],
    'fuel_efficiency':   [15, 18, 10, 22, 20, 12, 16, 8, 24, 9],
    'maintenance_cost':  [5000, 4000, 7000, 2000, 3000,
                          6500, 5500, 8000, 1500, 7500],
    'vehicle_type':      ['SUV','Sedan','Truck','Hatchback','Sedan',
                          'Truck','SUV','Truck','Hatchback','SUV']
}
df = pd.DataFrame(data)

print(df.head(10))
print(f"\nMissing Values:\n{df.isnull().sum()}")
df = df.fillna(df.mean(numeric_only=True))
X = pd.get_dummies(df, drop_first=True)
print(f"\nAll Features : {X.columns.tolist()}")

wcss_no_scale = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss_no_scale.append(kmeans.inertia_)

kmeans_no_scale = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_pred_no_scale = kmeans_no_scale.fit_predict(X)

type= data[['vehicle_type']]
to_scale= data.drop('vehicle_type',axis=1)
scaler=StandardScaler()
scaled=scaler.fit_transform(to_scale)

data_final=np.hstack([data.values,scaled])

wcss_scaled = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data_final)
    wcss_scaled.append(kmeans.inertia_)

kmeans_scaled = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_pred_scaled = kmeans_scaled.fit_predict(data_final)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Vehicle Clustering Comparison", fontsize=14, fontweight='bold')

colors = ['blue', 'green', 'red']
X_arr  = X.values

axes[0].plot(range(1,11), wcss_no_scale,
             marker='o', label='No Scaling', color='red')
axes[0].plot(range(1,11), wcss_scaled,
             marker='s', label='With Scaling', color='blue')
axes[0].set_title("Elbow Method Comparison")
axes[0].set_xlabel("Number of Clusters")
axes[0].set_ylabel("WCSS")
axes[0].legend()
axes[0].grid(True)

for i in range(3):
    axes[1].scatter(X_arr[y_pred_no_scale==i, 1],
                    X_arr[y_pred_no_scale==i, 2],
                    s=100, c=colors[i], label=f'Cluster {i+1}')
axes[1].set_title("Without Scaling")
axes[1].set_xlabel("Mileage")
axes[1].set_ylabel("Fuel Efficiency")
axes[1].legend()

for i in range(3):
    axes[2].scatter(data_final[y_pred_scaled==i, 1],
                    data_final[y_pred_scaled==i, 2],
                    s=100, c=colors[i], label=f'Cluster {i+1}')
axes[2].set_title("With Scaling (except vehicle_type)")
axes[2].set_xlabel("Mileage (scaled)")
axes[2].set_ylabel("Fuel Efficiency (scaled)")
axes[2].legend()

plt.tight_layout()
plt.show()

