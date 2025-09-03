import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as Fun

def create_model():
    # Autoencoder

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
        nn.Linear(512 * 4 * 4, 1024),
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
        nn.Linear(1024, 512 * 4 * 4),
        nn.BatchNorm1d(512 * 4 * 4),
        nn.ReLU(),
        nn.Unflatten(1, (512, 4, 4)),

        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 256x8x8
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 128x16x16
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64x32x32
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x64x64
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 3x128x128
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
            batch = data_shuffled[i:i + batch_size]
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

        print(
            f'Epoch {epoch + 1:2d}: Total Loss (avg)={avg_loss:.4f}, Reconstruction Loss (avg)={avg_recon:.4f}, Contrastive Loss (avg)={avg_contrast:.4f}, LR={scheduler.get_last_lr()[0]:.6f}')

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