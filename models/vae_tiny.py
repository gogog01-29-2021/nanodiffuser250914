"""A minimal VAE for nano diffusion.

This is a toy VAE used to encode images into a low‑dimensional latent space
and decode back into RGB images. It is intentionally tiny for demonstration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAETiny(nn.Module):
    """Minimal autoencoder with gaussian latent variables."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 4):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Latent mean and logvar
        self.fc_mu = nn.Conv2d(32, latent_channels, kernel_size=3, padding=1)
        self.fc_logvar = nn.Conv2d(32, latent_channels, kernel_size=3, padding=1)
        # Decoder
        self.dec_conv1 = nn.Conv2d(latent_channels, 16, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(16, in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_conv1(z))
        recon = torch.sigmoid(self.dec_conv2(h))
        return recon

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
