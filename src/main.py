from model import *
from helpers import *
from stats import *

def setup():
    # Device configuration, use CPU if no CUDA enabled GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using: {device}")

    return device

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
