import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("kaggle_Interests_group.csv")

# Step 1: Data Preprocessing
interest_columns = df.columns[2:]  # Drop 'group' and 'grand_tot_interests'
df_interests = df[interest_columns]
df_interests.fillna(0, inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_interests)

# Step 2: Data Splitting
X_train, X_temp = train_test_split(X_scaled, test_size=0.2, random_state=42)
X_eval, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

# Step 3 & 4: KMeans Implementation & Training
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_train)

# Step 5: Tuning KMeans
sil_scores = {}
for k in range(2, 10):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_eval)
    sil_scores[k] = silhouette_score(X_eval, labels)

best_k = max(sil_scores, key=sil_scores.get)
kmeans_best = KMeans(n_clusters=best_k, random_state=42)
kmeans_best.fit(X_train)

# Step 6: Evaluation for KMeans
kmeans_labels_test = kmeans_best.predict(X_test)
kmeans_silhouette = silhouette_score(X_test, kmeans_labels_test)

# Step 3–6 for Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=4)
hierarchical_labels_test = hierarchical.fit_predict(X_test)
hierarchical_silhouette = silhouette_score(X_test, hierarchical_labels_test)

# Output results
print("KMeans Clustering:")
print(f"Best K: {best_k}")
print(f"Silhouette Score: {kmeans_silhouette}")

print("\nHierarchical Clustering:")
print(f"Silhouette Score: {hierarchical_silhouette}")