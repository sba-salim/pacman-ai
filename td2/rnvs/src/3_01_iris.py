from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import matplotlib.pyplot as plt
import os  # Ajouté pour la gestion du dossier

# === Créer le dossier 'figures' s'il n'existe pas ===
os.makedirs("figures", exist_ok=True)

# === 1. Chargement des données ===
iris = load_iris(as_frame=True)
X = iris['data']
true_labels = iris['target']  # Labels réels (pour l'ARI)

# === 2. Mise à l’échelle ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Initialisation des listes pour stocker les scores ===
k_values = list(range(2, 6))  # Plage de k de 2 à 5
silhouette_lloyd_scores = []
silhouette_pp_scores = []
db_lloyd_scores = []
db_pp_scores = []
inertia_lloyd_scores = []
inertia_pp_scores = []
ari_lloyd_scores = []
ari_pp_scores = []

# === 4. Boucle sur les valeurs de k ===
for k in k_values:
    # 4.1 KMeans avec initialisation aléatoire (Lloyd)
    kmeans_lloyd = KMeans(n_clusters=k, init='random',
                          n_init=10, random_state=42)
    labels_lloyd = kmeans_lloyd.fit_predict(X_scaled)

    # 4.2 KMeans avec k-means++ (amélioré)
    kmeans_pp = KMeans(n_clusters=k, init='k-means++',
                       n_init=10, random_state=42)
    labels_pp = kmeans_pp.fit_predict(X_scaled)

    # Calcul des métriques pour Lloyd
    inertia_lloyd = kmeans_lloyd.inertia_
    silhouette_lloyd = silhouette_score(X_scaled, labels_lloyd)
    db_lloyd = davies_bouldin_score(X_scaled, labels_lloyd)
    ari_lloyd = adjusted_rand_score(
        true_labels, labels_lloyd)  # ARI pour Lloyd

    # Calcul des métriques pour k-means++
    inertia_pp = kmeans_pp.inertia_
    silhouette_pp = silhouette_score(X_scaled, labels_pp)
    db_pp = davies_bouldin_score(X_scaled, labels_pp)
    ari_pp = adjusted_rand_score(true_labels, labels_pp)  # ARI pour k-means++

    # Ajout des résultats dans les listes
    inertia_lloyd_scores.append(inertia_lloyd)
    silhouette_lloyd_scores.append(silhouette_lloyd)
    db_lloyd_scores.append(db_lloyd)
    ari_lloyd_scores.append(ari_lloyd)

    inertia_pp_scores.append(inertia_pp)
    silhouette_pp_scores.append(silhouette_pp)
    db_pp_scores.append(db_pp)
    ari_pp_scores.append(ari_pp)

    # Affichage des résultats dans la console pour chaque k
    print(f"K = {k}")
    print(
        f"  Lloyd -> Inertie : {inertia_lloyd:.3f}, Silhouette : {silhouette_lloyd:.3f}, Davies-Bouldin : {db_lloyd:.3f}, ARI : {ari_lloyd:.3f}")
    print(
        f"  k-means++ -> Inertie : {inertia_pp:.3f}, Silhouette : {silhouette_pp:.3f}, Davies-Bouldin : {db_pp:.3f}, ARI : {ari_pp:.3f}")
    print("-" * 50)

# === 5. Tracer les courbes ===
plt.figure(figsize=(15, 12))

# Inertie
plt.subplot(4, 1, 1)
plt.plot(k_values, inertia_lloyd_scores, marker='o', label="Inertie Lloyd")
plt.plot(k_values, inertia_pp_scores, marker='o', label="Inertie k-means++")
plt.title("Inertie vs k")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie")
plt.legend()
plt.grid(True)

# Silhouette
plt.subplot(4, 1, 2)
plt.plot(k_values, silhouette_lloyd_scores,
         marker='o', label="Silhouette Lloyd")
plt.plot(k_values, silhouette_pp_scores,
         marker='o', label="Silhouette k-means++")
plt.title("Silhouette Score vs k")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid(True)

# Davies-Bouldin
plt.subplot(4, 1, 3)
plt.plot(k_values, db_lloyd_scores, marker='o', label="Davies-Bouldin Lloyd")
plt.plot(k_values, db_pp_scores, marker='o', label="Davies-Bouldin k-means++")
plt.title("Davies-Bouldin Index vs k")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Davies-Bouldin Index")
plt.legend()
plt.grid(True)

# ARI (Adjusted Rand Index)
plt.subplot(4, 1, 4)
plt.plot(k_values, ari_lloyd_scores, marker='o', label="ARI Lloyd")
plt.plot(k_values, ari_pp_scores, marker='o', label="ARI k-means++")
plt.title("Adjusted Rand Index vs k")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("ARI")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("figures/iris_kmeans_metrics.png")

plt.show()
