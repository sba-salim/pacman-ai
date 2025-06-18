import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os

# Créer le dossier 'figures' s'il n'existe pas
os.makedirs('figures', exist_ok=True)

# === 1. Chargement du dataset ===
df = pd.read_csv("../data/Mall_Customers.csv")
df_encoded = df.copy()
df_encoded['Genre'] = df_encoded['Genre'].map({'Male': 0, 'Female': 1})
X = df_encoded.drop(columns=['CustomerID'])

# === 2. Mise à l’échelle ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Méthodes à tester ===
methods = {
    'Lloyd (random)': 'random',
    'KMeans++': 'k-means++',
    # 'Lloyd (random)': 'random',
}

datasets = {
    'Sans échelle': X.values,
    'Avec échelle': X_scaled    # rnvs : décommenté
}

k_values = list(range(2, 6))

# === 4. Calcul des scores ===
results = {}

for dataset_name, data in datasets.items():
    results[dataset_name] = {}
    for method_name, init_method in methods.items():
        silhouette_scores = []
        db_scores = []
        inertias = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, init=init_method,
                            n_init=10, random_state=42)
            labels = kmeans.fit_predict(data)

            inertia = kmeans.inertia_
            silhouette = silhouette_score(data, labels)
            db = davies_bouldin_score(data, labels)

            inertias.append(inertia)
            silhouette_scores.append(silhouette)
            db_scores.append(db)

        results[dataset_name][method_name] = {
            'inertia': inertias,
            'silhouette': silhouette_scores,
            'db': db_scores
        }

        # === Affichage console ===
        print(f"\n=== Résultats pour {dataset_name} - {method_name} ===")
        for i, k in enumerate(k_values):
            print(
                f"k = {k} | Inertie = {inertias[i]:.2f} | Silhouette = {silhouette_scores[i]:.3f} | DB = {db_scores[i]:.3f}")

# === 5. Tracer les résultats ===
metrics = ['inertia', 'silhouette', 'db']
titles = {
    'inertia': 'Erreur carrée (Inertie)',
    'silhouette': 'Silhouette Score',
    'db': 'Indice de Davies-Bouldin'
}

for metric in metrics:
    plt.figure(figsize=(10, 6))
    for dataset_name in datasets:
        for method_name in methods:
            scores = results[dataset_name][method_name][metric]
            label = f"{method_name} - {dataset_name}"
            plt.plot(k_values, scores, marker='o', label=label)
    plt.title(titles[metric])
    plt.xlabel("k")
    plt.ylabel(titles[metric])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/mall_{metric}.png")
    plt.show()
