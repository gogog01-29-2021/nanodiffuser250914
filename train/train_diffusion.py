"""
Training script for nano diffusion models.

This script demonstrates a simple training loop for a tiny latent diffusion model. It shows how to
initialize the UNet backbone, prepare a dummy dataset, and run optimization steps. For actual
projects you would replace the random data with a real dataset and expand the loss computation.

Usage:
    python train_diffusion.py --epochs 5 --batch_size 32

"""

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import tiny model components. These imports assume the repository structure:
# nano_diffusion/models/...
from models.unet_tiny import UNetTiny  # type: ignore
from models.vae_tiny import VAETiny  # type: ignore


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training hyperparameters."""
    parser = argparse.ArgumentParser(description="Train a mini diffusion model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()


def create_dummy_dataloader(batch_size: int) -> DataLoader:
    """
    Create a dummy DataLoader that yields random latent vectors.

    In practice you would load your real dataset here and encode it with the VAE.
    """
    # Generate random latent data (100 samples, 4 channels, 64x64 resolution)
    data = torch.randn(100, 4, 64, 64)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_one_epoch(model: torch.nn.Module, dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """
    Run a single training epoch over the DataLoader.
    """
    model.train()
    for batch, in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # Forward pass through the model
        output = model(batch, None, None)  # UNetTiny takes latent + timestep + cond; cond=None here
        # Compute a dummy loss as MSE between input and output
        loss = torch.nn.functional.mse_loss(output, batch)
        loss.backward()
        optimizer.step()



def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and optimizer
    model = UNetTiny().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Prepare DataLoader (replace with encoded real dataset)
    dataloader = create_dummy_dataloader(args.batch_size)

    for epoch in range(args.epochs):
        train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}/{args.epochs} completed")


if __name__ == "__main__":
    main()
