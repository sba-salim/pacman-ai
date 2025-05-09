from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os  # Ajouté pour la gestion du dossier

# === Créer le dossier 'figures' s'il n'existe pas ===
os.makedirs("figures", exist_ok=True)

# === 1. Chargement des données ===
cancer = load_breast_cancer(as_frame=True)
X = cancer['data']
true_labels = cancer['target']

# === 2. Split train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, true_labels, test_size=0.2, random_state=42, stratify=true_labels
)

# === 3. Mise à l’échelle ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 4. Initialisation des listes pour stocker les scores ===
k_values = list(range(2, 6))
silhouette_train_scores = []
silhouette_test_scores = []
db_train_scores = []
db_test_scores = []
ari_train_scores = []
ari_test_scores = []
silhouette_train_knn_scores = []
silhouette_test_knn_scores = []
db_train_knn_scores = []
db_test_knn_scores = []
ari_train_knn_scores = []
ari_test_knn_scores = []

# === 5. Affichage des entêtes pour les métriques dans la console ===
print("k | Silhouette Train | Silhouette Test | Davies-Bouldin Train | Davies-Bouldin Test | ARI Train | ARI Test | Silhouette Train (KNN) | Silhouette Test (KNN) | Davies-Bouldin Train (KNN) | Davies-Bouldin Test (KNN) | ARI Train (KNN) | ARI Test (KNN)")
print("-" * 120)

# === 6. Boucle sur les valeurs de k ===
for k in k_values:
    # Clustering avec KMeans
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    train_labels = kmeans.fit_predict(X_train_scaled)
    test_labels = kmeans.predict(X_test_scaled)

    silhouette_train = silhouette_score(X_train_scaled, train_labels)
    silhouette_test = silhouette_score(X_test_scaled, test_labels)
    db_train = davies_bouldin_score(X_train_scaled, train_labels)
    db_test = davies_bouldin_score(X_test_scaled, test_labels)
    ari_train = adjusted_rand_score(y_train, train_labels)
    ari_test = adjusted_rand_score(y_test, test_labels)

    # Stockage pour KMeans sans KNN
    silhouette_train_scores.append(silhouette_train)
    silhouette_test_scores.append(silhouette_test)
    db_train_scores.append(db_train)
    db_test_scores.append(db_test)
    ari_train_scores.append(ari_train)
    ari_test_scores.append(ari_test)

    # Clustering avec KMeans et KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_train_labels = knn.predict(X_train_scaled)
    knn_test_labels = knn.predict(X_test_scaled)

    silhouette_train_knn = silhouette_score(X_train_scaled, knn_train_labels)
    silhouette_test_knn = silhouette_score(X_test_scaled, knn_test_labels)
    db_train_knn = davies_bouldin_score(X_train_scaled, knn_train_labels)
    db_test_knn = davies_bouldin_score(X_test_scaled, knn_test_labels)
    ari_train_knn = adjusted_rand_score(y_train, knn_train_labels)
    ari_test_knn = adjusted_rand_score(y_test, knn_test_labels)

    # Stockage pour KMeans avec KNN
    silhouette_train_knn_scores.append(silhouette_train_knn)
    silhouette_test_knn_scores.append(silhouette_test_knn)
    db_train_knn_scores.append(db_train_knn)
    db_test_knn_scores.append(db_test_knn)
    ari_train_knn_scores.append(ari_train_knn)
    ari_test_knn_scores.append(ari_test_knn)

    # Affichage des résultats
    print(f"{k:<2} | {silhouette_train:.3f}           | {silhouette_test:.3f}         | {db_train:.3f}                 | {db_test:.3f}               | {ari_train:.3f}     | {ari_test:.3f}      | {silhouette_train_knn:.3f}              | {silhouette_test_knn:.3f}             | {db_train_knn:.3f}                 | {db_test_knn:.3f}               | {ari_train_knn:.3f}      | {ari_test_knn:.3f}")

# === 7. Tracer les courbes ===
plt.figure(figsize=(15, 12))

# Silhouette
plt.subplot(3, 1, 1)
plt.plot(k_values, silhouette_train_scores, marker='o', label="Silhouette Train (KMeans)")
plt.plot(k_values, silhouette_test_scores, marker='o', label="Silhouette Test (KMeans)")
plt.plot(k_values, silhouette_train_knn_scores, marker='x', label="Silhouette Train (KMeans + KNN)")
plt.plot(k_values, silhouette_test_knn_scores, marker='x', label="Silhouette Test (KMeans + KNN)")
plt.title("Silhouette Score vs k")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid(True)

# Davies-Bouldin
plt.subplot(3, 1, 2)
plt.plot(k_values, db_train_scores, marker='o', label="Davies-Bouldin Train (KMeans)")
plt.plot(k_values, db_test_scores, marker='o', label="Davies-Bouldin Test (KMeans)")
plt.plot(k_values, db_train_knn_scores, marker='x', label="Davies-Bouldin Train (KMeans + KNN)")
plt.plot(k_values, db_test_knn_scores, marker='x', label="Davies-Bouldin Test (KMeans + KNN)")
plt.title("Davies-Bouldin Index vs k")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Davies-Bouldin Index")
plt.legend()
plt.grid(True)

# ARI
plt.subplot(3, 1, 3)
plt.plot(k_values, ari_train_scores, marker='o', label="ARI Train (KMeans)")
plt.plot(k_values, ari_test_scores, marker='o', label="ARI Test (KMeans)")
plt.plot(k_values, ari_train_knn_scores, marker='x', label="ARI Train (KMeans + KNN)")
plt.plot(k_values, ari_test_knn_scores, marker='x', label="ARI Test (KMeans + KNN)")
plt.title("Adjusted Rand Index vs k")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("ARI")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("figures/breast_cancer_kmeans_knn_metrics.png")
plt.show()
