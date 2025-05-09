import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import os  # Ajouté pour la gestion du dossier

# === Créer le dossier 'figures' s'il n'existe pas ===
os.makedirs("figures", exist_ok=True)

# === Chargement et mise à l'échelle ===
wine = load_wine(as_frame=True)
X = wine['data']
y = wine['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Initialisation ===
k_values = list(range(2, 6))
methods = ['random', 'k-means++']

# Dictionnaires pour stocker les scores
scores = {method: {
    'inertia': [], 'silhouette': [], 'davies': [], 'ari': []
} for method in methods}

# === Boucle sur k et les méthodes ===
print("k | Méthode   | Inertie | Silhouette | Davies-Bouldin | ARI")
print("-" * 60)

for k in k_values:
    for method in methods:
        kmeans = KMeans(n_clusters=k, init=method, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_scaled, labels)
        db_index = davies_bouldin_score(X_scaled, labels)
        ari = adjusted_rand_score(y, labels)

        # Stockage
        scores[method]['inertia'].append(inertia)
        scores[method]['silhouette'].append(silhouette)
        scores[method]['davies'].append(db_index)
        scores[method]['ari'].append(ari)

        # Console
        print(f"{k:<2} | {method:<9} | {inertia:8.2f} | {silhouette:.3f}     | {db_index:.3f}          | {ari:.3f}")

# === Tracés ===
plt.figure(figsize=(14, 10))

# Silhouette
plt.subplot(2, 2, 1)
for method in methods:
    plt.plot(k_values, scores[method]['silhouette'], marker='o', label=f"{method}")
plt.title("Silhouette Score vs k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid(True)

# Davies-Bouldin
plt.subplot(2, 2, 2)
for method in methods:
    plt.plot(k_values, scores[method]['davies'], marker='o', label=f"{method}")
plt.title("Davies-Bouldin Index vs k")
plt.xlabel("k")
plt.ylabel("Davies-Bouldin Index")
plt.legend()
plt.grid(True)

# Inertie
plt.subplot(2, 2, 3)
for method in methods:
    plt.plot(k_values, scores[method]['inertia'], marker='o', label=f"{method}")
plt.title("Inertie vs k")
plt.xlabel("k")
plt.ylabel("Inertie (Erreur quadratique)")
plt.legend()
plt.grid(True)

# ARI
plt.subplot(2, 2, 4)
for method in methods:
    plt.plot(k_values, scores[method]['ari'], marker='o', label=f"{method}")
plt.title("Adjusted Rand Index vs k")
plt.xlabel("k")
plt.ylabel("ARI")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("figures/wine_kmeans_metrics.png")

plt.show()
