"""Train a diffusion model from scratch.

This script provides a skeleton for training a tiny diffusion model using the
modules in this repository. It should be adapted with dataset loading,
optimizer setup, and training loops.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional

try:
    # Import torchvision components lazily; only needed if using a real dataset
    from torchvision.datasets import ImageFolder  # type: ignore
    from torchvision import transforms  # type: ignore
except ImportError:
    ImageFolder = None  # type: ignore
    transforms = None  # type: ignore

from models.unet_tiny import UNetTiny
from models.vae_tiny import VAETiny
from core.schedules import edm_sigma_schedule
from core.losses import v_prediction_loss

def train(
    num_epochs: int = 1,
    batch_size: int = 16,
    dataset_dir: Optional[str] = None,
    checkpoint_interval: int = 100,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """Train the diffusion model.

    This function will attempt to load images from ``dataset_dir`` using
    ``torchvision.datasets.ImageFolder`` if provided. Images are resized to
    64×64 and converted to tensors. If ``dataset_dir`` is ``None`` or
    ``torchvision`` is not installed, a synthetic dataset of random tensors is
    used instead. The model weights are periodically saved to ``checkpoint_dir``
    every ``checkpoint_interval`` steps.

    Args:
        num_epochs: Number of full passes over the dataset.
        batch_size: Number of samples per batch.
        dataset_dir: Directory containing image data organized by class
            subdirectories. If ``None``, uses a synthetic dataset.
        checkpoint_interval: Number of training steps between checkpoint saves.
        checkpoint_dir: Directory to save checkpoint files into.
    """

    # Attempt to build a real dataset loader if dataset_dir is provided
    if dataset_dir and ImageFolder is not None and transforms is not None:
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )
        dataset = ImageFolder(dataset_dir, transform=transform)
    else:
        # Fallback synthetic dataset: random tensors shaped like images
        dataset = [torch.randn(3, 64, 64) for _ in range(100)]

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAETiny().to(device)
    unet = UNetTiny().to(device)
    optimizer = torch.optim.Adam(list(vae.parameters()) + list(unet.parameters()), lr=1e-3)
    sigmas = edm_sigma_schedule(10)

    global_step = 0
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch  # for ImageFolder
            x = x.to(device)
            optimizer.zero_grad()
            # Encode image to latent
            mu, logvar = vae.encode(x)
            z0 = vae.reparameterize(mu, logvar)
            # Sample noise level (use first sigma for simplicity)
            sigma = sigmas[0]
            noise = torch.randn_like(z0) * sigma
            zt = z0 + noise
            # Dummy time embedding
            t_emb = torch.zeros(z0.size(0), 1, 1, 1, device=device)
            pred = unet(zt, t_emb)
            loss = v_prediction_loss(pred, noise)
            loss.backward()
            optimizer.step()

            global_step += 1
            # Save checkpoint periodically
            if global_step % checkpoint_interval == 0:
                # Ensure checkpoint directory exists
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                ckpt_path = Path(checkpoint_dir) / f"checkpoint_epoch{epoch+1}_step{global_step}.pt"
                torch.save(
                    {
                        "unet": unet.state_dict(),
                        "vae": vae.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                    },
                    ckpt_path,
                )

        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

if __name__ == "__main__":
    train()
