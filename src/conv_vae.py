import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class ConvVAEEncoder(nn.Module):
    def __init__(self):
        super(ConvVAEEncoder, self).__init__()
        # Use the same architecture as your custom encoder
        self.conv_layers = nn.Sequential(
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
        )

        # Split into mu and logvar for VAE
        self.mu_layer = nn.Linear(1024, 512)
        self.logvar_layer = nn.Linear(1024, 512)

    def forward(self, x):
        h = self.conv_layers(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class ConvVAEDecoder(nn.Module):
    def __init__(self):
        super(ConvVAEDecoder, self).__init__()
        # Same decoder architecture as your custom model
        self.decoder = nn.Sequential(
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

    def forward(self, z):
        return self.decoder(z)


def reparameterize(mu, logvar):
    """Reparameterization trick for VAE"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def conv_vae_loss(reconstructed, original, mu, logvar, beta=1.0):
    """VAE loss function with KL divergence"""
    # Reconstruction loss
    recon_loss = F.mse_loss(reconstructed, original, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def create_conv_vae():
    """Create ConvVAE encoder and decoder"""
    encoder = ConvVAEEncoder()
    decoder = ConvVAEDecoder()
    return encoder, decoder


def train_conv_vae(encoder, decoder, data, device, epochs=10):
    """Training loop for Convolutional VAE"""
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    data = data.to(device)

    # AdamW optimizer with different learning rates for encoder and decoder
    optimizer = optim.AdamW([
        {"params": encoder.parameters(), "lr": 2e-3, "weight_decay": 0.01},
        {"params": decoder.parameters(), "lr": 5e-4, "weight_decay": 0.0},
    ], betas=(0.9, 0.95))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    batch_size = 32
    beta = 1.0  # KL weight

    print(f"Training ConvVAE for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0

        # Shuffle data each epoch
        indices = torch.randperm(len(data))
        data_shuffled = data[indices]

        for i in range(0, len(data_shuffled), batch_size):
            batch = data_shuffled[i:i + batch_size]
            optimizer.zero_grad()

            # Forward pass
            mu, logvar = encoder(batch)
            z = reparameterize(mu, logvar)
            reconstructed = decoder(z)

            # Calculate loss
            loss, recon_loss, kl_loss = conv_vae_loss(reconstructed, batch, mu, logvar, beta)

            # Normalize by batch size
            loss = loss / batch.size(0)
            recon_loss = recon_loss / batch.size(0)
            kl_loss = kl_loss / batch.size(0)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = total_loss / num_batches
        avg_recon = total_recon_loss / num_batches
        avg_kl = total_kl_loss / num_batches

        print(
            f'Epoch {epoch + 1:2d}: Total Loss (avg)={avg_loss:.4f}, Reconstruction Loss (avg)={avg_recon:.4f}, KL Loss (avg)={avg_kl:.4f}, LR={scheduler.get_last_lr()[0]:.6f}')

        # Save if best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'loss': avg_loss,
                'epoch': epoch
            }, 'trained_convvae_model.pth')
            print(f'-> BEST ConvVAE MODEL SAVED! Loss: {avg_loss:.4f}')

    return encoder, decoder


def extract_features_conv_vae(encoder, data, device):
    """Extract features using trained ConvVAE encoder"""
    encoder.eval()
    features_list = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)
            mu, logvar = encoder(batch)
            # Use mean (mu) as features for clustering
            features_list.append(mu.cpu())

    return torch.cat(features_list, dim=0)
