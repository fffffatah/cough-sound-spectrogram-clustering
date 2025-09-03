import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as Fun
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

def setup():
    # Device configuration, use CPU if no CUDA enabled GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using: {device}")

    return device

def calculate_dataset_stats(files, data_dir):
    # Calculate mean and std for RGB channels across the dataset, we will need this during Normalization
    # Basic transform to tensor only

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Accumulate pixel values for each channel
    pixel_sum = torch.zeros(3)
    pixel_squared_sum = torch.zeros(3)
    num_pixels = 0

    print(f"Calculating stats from {len(files)} images...")

    for i, filename in enumerate(files):
        if i % 500 == 0:
            print(f"Processed {i}/{len(files)} images")

        try:
            img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
            img_tensor = transform(img)  # [3, 128, 128]

            # Sum across spatial dimensions (H, W)
            pixel_sum += img_tensor.sum(dim=[1, 2])
            pixel_squared_sum += (img_tensor ** 2).sum(dim=[1, 2])
            num_pixels += img_tensor.shape[1] * img_tensor.shape[2]  # H * W

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Calculate mean and std
    mean = pixel_sum / num_pixels
    std = torch.sqrt((pixel_squared_sum / num_pixels) - (mean ** 2))

    print(f"Dataset statistics:")
    print(f"Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    return mean.tolist(), std.tolist()

def load_images(data_dir, max_images=34394):
    files = [f for f in os.listdir(data_dir) if f.endswith('.png')][:max_images]
    # mean, std = calculate_dataset_stats(files, data_dir)

    # Using calculated mean and std for normalization
    mean = [0.3540, 0.1148, 0.2336]
    std = [0.3614, 0.1229, 0.2128]

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # 3 Channel Normalization
    ])
    
    images = []
    print(f"Loading {len(files)} images...")
    
    for i, filename in enumerate(files):
        if i % 500 == 0:
            print(f"Loaded {i}/{len(files)}")
        
        try:
            img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
            images.append(transform(img))

        except Exception as e:
            print(f"Error: {e}")
    
    return torch.stack(images)

def create_model():
    #Autoencoder

    # Custom Encoder
    encoder = nn.Sequential(
        # Input: 3x128x128 for RGB spectrogram
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 64x64x64

        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 128x32x32

        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 256x16x16

        nn.Conv2d(256, 512, 3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),  # 512x4x4

        nn.Flatten(),
        nn.Linear(512*4*4, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),  # Smaller latent space for better clustering
        nn.ReLU()
    )

    # Custom Decoder
    decoder = nn.Sequential(
        nn.Linear(512, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 512*4*4),
        nn.BatchNorm1d(512 * 4 * 4),
        nn.ReLU(),
        nn.Unflatten(1, (512, 4, 4)),

        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 256x8x8
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 128x16x16
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64x32x32
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 32x64x64
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # 3x128x128
        nn.Tanh()
    )
    
    return encoder, decoder

def contrastive_loss(features, temperature=0.5):
    #Simple contrastive loss

    batch_size = features.size(0)

    # Normalize features
    features_norm = Fun.normalize(features, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(features_norm, features_norm.t()) / temperature

    # Remove diagonal elements
    mask = torch.eye(batch_size, device=features.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

    # Encourage diversity (lower similarity between different samples)
    loss = torch.mean(torch.max(similarity_matrix, dim=1)[0])

    return loss

def train_model(encoder, decoder, data, device, epochs=10):
    # Training loop
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    data = data.to(device)

    # AdamW optimizer with different learning rates for encoder and decoder
    optimizer = optim.AdamW([
        {"params": encoder.parameters(), "lr": 2e-3, "weight_decay": 0.01},
        {"params": decoder.parameters(), "lr": 5e-4, "weight_decay": 0.0},
    ], betas=(0.9, 0.95))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss function
    mse_loss = nn.MSELoss()
    best_loss = float('inf')
    batch_size = 32  # Smaller batch size for better gradient estimates
    
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_contrast_loss = 0
        num_batches = 0
        
        # Shuffle data each epoch so that the model doesn't learn order of data
        indices = torch.randperm(len(data))
        data_shuffled = data[indices]
        
        for i in range(0, len(data_shuffled), batch_size):
            batch = data_shuffled[i:i+batch_size]
            optimizer.zero_grad()
            
            # Forward pass
            features = encoder(batch)
            reconstructed = decoder(features)
            
            # Combined loss
            recon_loss = mse_loss(reconstructed, batch)
            contrast_loss = contrastive_loss(features)
            
            # Weighted combination
            total_batch_loss = recon_loss + 0.1 * contrast_loss

            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_recon_loss += recon_loss.item()
            total_contrast_loss += contrast_loss.item()
            num_batches += 1
        
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        avg_recon = total_recon_loss / num_batches
        avg_contrast = total_contrast_loss / num_batches
        
        print(f'Epoch {epoch+1:2d}: Total Loss (avg)={avg_loss:.4f}, Reconstruction Loss (avg)={avg_recon:.4f}, Contrastive Loss (avg)={avg_contrast:.4f}, LR={scheduler.get_last_lr()[0]:.6f}')
        
        # Save if best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'loss': avg_loss,
                'epoch': epoch
            }, 'trained_model.pth')
            print(f'-> BEST MODEL SAVED! Loss: {avg_loss:.4f}')
    
    return encoder, decoder

def extract_features(encoder, data, device):
    # Feature extraction
    encoder.eval()
    data = data.to(device)

    features_list = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]

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
        'Agglomerative': AgglomerativeClustering(n_clusters=8, linkage='ward'),
        'Spectral': SpectralClustering(n_clusters=8, random_state=42,affinity='nearest_neighbors', n_neighbors=15, assign_labels='kmeans'),
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
            
            print(f"{name:12} | Clusters: {n_clusters:2} | Silhouette: {silhouette:7.3f} | DB Index: {davies_bouldin:7.3f}")
            
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

def main():
    # Setup
    device = setup()
    data_dir = "/mnt/d/Productivity/CSE715/cough-type-clustering/cough_dataset"
    
    # Load data
    print("\n1. Loading images...")
    images = load_images(data_dir, max_images=10000)
    print(f"Loaded {len(images)} images of shape {images[0].shape}")
    
    # Create model
    print("\n2. Creating model...")
    encoder, decoder = create_model()
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
    print(f"Model parameters: {total_params:,}")

    print("\n3. Training...")
    encoder, decoder = train_model(encoder, decoder, images, device, epochs=10)

    print("\n4. Feature extraction...")
    features = extract_features(encoder, images, device)
    print(f"Enhanced features shape: {features.shape}")

    print("\n5. Clustering comparison...")
    results, best_labels, best_algo, features_processed = clustering_comparison(features)

    print("\n6. Creating visualizations...")
    visualize_comparison(features_processed, results)
    
    print(f"\n✅ RESULTS!")
    print(f"✅ Best model saved as 'trained_model.pth'")
    print(f"✅ Best clustering algorithm: {best_algo}")

if __name__ == "__main__":
    main()
