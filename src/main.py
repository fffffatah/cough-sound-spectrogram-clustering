import argparse
from model import *
from helpers import *
from stats import *
from conv_vae import *

def setup():
    # Device configuration, use CPU if no CUDA enabled GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using: {device}")

    return device

def main():
    # Args to choose encoder type
    parser = argparse.ArgumentParser(description='Cough Type Clustering')
    parser.add_argument('--model', choices=['custom', 'convvae'], default='custom',
                        help='Model type: custom (Contrastive Autoencoder) or convvae (Convolutional VAE)')
    args = parser.parse_args()

    # Setup
    device = setup()
    data_dir = os.path.dirname(__file__).replace('src','cough_dataset')
    
    # Load data
    print("\n1. Loading images...")
    images = load_images(data_dir, max_images=10000)
    print(f"Loaded {len(images)} images of shape {images[0].shape}")

    features = None

    # Create model
    if args.model == 'custom':
        print("\n2. Creating model...")
        encoder, decoder = create_model()

        total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
        print(f"Model parameters: {total_params:,}")

        print("\n3. Training...")
        encoder, decoder = train_model(encoder, decoder, images, device)

        print("\n4. Feature extraction...")
        features = extract_features(encoder, images, device)
    elif args.model == 'convvae':
        print("\n2. Creating ConvVAE model...")
        encoder, decoder = create_conv_vae()

        total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
        print(f"ConvVAE Model parameters: {total_params:,}")

        print("\n3. Training ConvVAE...")
        encoder, decoder = train_conv_vae(encoder, decoder, images, device)

        print("\n4. Feature extraction...")
        features = extract_features_conv_vae(encoder, images, device)

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
