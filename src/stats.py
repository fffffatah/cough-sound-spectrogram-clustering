import numpy as np
import torch
import matplotlib.pyplot as plt
import hdbscan
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

def extract_features(encoder, data, device):
    # Feature extraction
    encoder.eval()
    data = data.to(device)

    features_list = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            feat = encoder(batch)
            features_list.append(feat.cpu().numpy())

            batch_flipped = torch.flip(batch, dims=[3])  # Flip horizontally
            feat_flipped = encoder(batch_flipped)
            features_list.append(feat_flipped.cpu().numpy())

    features = np.concatenate(features_list)

    # Remove duplicates and normalize
    features = features[:len(data)]

    return features


def clustering_comparison(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=min(50, features.shape[1]))  # Reduce to 50 dims max
    features_pca = pca.fit_transform(features_scaled)

    print(f"Features reduced from {features.shape[1]} to {features_pca.shape[1]} dimensions")
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Define clustering algorithms to compare
    algorithms = {
        'K-Means': KMeans(n_clusters=8, random_state=42, n_init=50, max_iter=1000),
        'GMM': GaussianMixture(n_components=8, random_state=42, max_iter=200, covariance_type='diag'),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5, n_jobs=-1),
        'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.0),
        'Agglomerative': AgglomerativeClustering(n_clusters=8, linkage='ward'),
        'Spectral': SpectralClustering(n_clusters=8, random_state=42, affinity='nearest_neighbors', n_neighbors=15,
                                       assign_labels='kmeans'),
    }

    results = {}

    print("\nClustering comparison:")
    print("-" * 60)

    for name, algorithm in algorithms.items():
        try:
            if name == 'GMM':
                algorithm.fit(features_pca)
                labels = algorithm.predict(features_pca)
            else:
                labels = algorithm.fit_predict(features_pca)

            # Calculate metrics
            n_clusters = len(np.unique(labels[labels >= 0]))

            if n_clusters > 1:
                mask = labels >= 0
                if np.sum(mask) > 1:
                    silhouette = silhouette_score(features_pca[mask], labels[mask])
                    davies_bouldin = davies_bouldin_score(features_pca[mask], labels[mask])
                else:
                    silhouette = -1
                    davies_bouldin = 999
            else:
                silhouette = -1
                davies_bouldin = 999

            results[name] = {
                'labels': labels,
                'n_clusters': n_clusters,
                'silhouette_score': silhouette,
                'davies_bouldin': davies_bouldin
            }

            print(
                f"{name:12} | Clusters: {n_clusters:2} | Silhouette: {silhouette:7.3f} | DB Index: {davies_bouldin:7.3f}")

        except Exception as e:
            print(f"{name:12} | ERROR: {str(e)[:35]}...")
            results[name] = None

    print("-" * 60)

    # Find best
    best_algo = None
    best_score = -1

    for name, result in results.items():
        if result and result['silhouette_score'] > best_score:
            best_score = result['silhouette_score']
            best_algo = name

    if best_algo:
        print(f"Best algorithm: {best_algo} (Silhouette: {best_score:.3f})")
        best_labels = results[best_algo]['labels']
    else:
        print("Using K-Means as fallback")
        best_labels = results['K-Means']['labels'] if results['K-Means'] else None
        best_algo = 'K-Means'

    return results, best_labels, best_algo, features_pca


def visualize_comparison(features, results):
    pca_viz = PCA(n_components=2)
    features_2d = pca_viz.fit_transform(features)

    valid_results = {name: result for name, result in results.items() if result is not None}

    for name, result in valid_results.items():
        labels = result['labels']

        # Create individual plot
        plt.figure(figsize=(7, 6))
        plt.scatter(features_2d[:, 0], features_2d[:, 1],
                    c=labels, cmap='tab10', s=15, alpha=0.7,
                    edgecolors='k', linewidth=0.1)

        title = f'{name}\nClusters: {result["n_clusters"]}, Silhouette: {result["silhouette_score"]:.3f}'
        plt.title(title, fontsize=13, fontweight='bold')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True, alpha=0.3)

        # Save per algorithm
        filename = f"{name}_clustering.png"
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filename}")