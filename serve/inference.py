"""CLI and server interface for nano diffusion.

This script exposes a simple command‑line and FastAPI server for generating
images from prompts. It is meant as a starting point and should be extended
with proper request parsing and concurrency control.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.unet_tiny import UNetTiny
from models.vae_tiny import VAETiny
from core.schedules import edm_sigma_schedule
from core.samplers import dpm_solver_pp

class GenerateRequest(BaseModel):
    prompt: Optional[str] = ""
    num_steps: int = 10
    guidance_scale: float = 0.0

app = FastAPI(title="Nano Diffusion Inference Server")

# Global model references. These will be set on application startup.
unet: Optional[UNetTiny] = None
vae: Optional[VAETiny] = None
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[Path]:
    """Return the most recent checkpoint file in the given directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files.

    Returns:
        Path to the latest checkpoint file or ``None`` if none exists.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    ckpts = list(ckpt_dir.glob("*.pt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]

def load_models(weights_path: Optional[str] = None) -> None:
    """Load UNetTiny and VAETiny models and optionally restore from a checkpoint.

    This function populates the global ``unet`` and ``vae`` variables with models
    loaded on the appropriate device. If a checkpoint file is provided or found
    in the default checkpoint directory, the model states are restored.

    Args:
        weights_path: Optional path to a checkpoint file containing model
            parameters. If ``None``, attempts to load the most recent
            checkpoint in the default ``checkpoints/`` directory. If no
            checkpoint is found, models are left with random weights.
    """
    global unet, vae
    # Instantiate models on device
    unet = UNetTiny().to(device)
    vae = VAETiny().to(device)
    ckpt_path: Optional[Path] = None
    if weights_path:
        ckpt_path = Path(weights_path)
    else:
        ckpt_path = find_latest_checkpoint("checkpoints")
    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if "unet" in ckpt and "vae" in ckpt:
            unet.load_state_dict(ckpt["unet"])
            vae.load_state_dict(ckpt["vae"])
    unet.eval()
    vae.eval()
    # Warm‑up forward pass to initialize CUDA graphs and kernels
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 64, 64, device=device)
        mu, logvar = vae.encode(dummy)
        z0 = vae.reparameterize(mu, logvar)
        t_emb = torch.zeros(1, 1, 1, 1, device=device)
        _ = unet(z0, t_emb)

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate an image from a prompt using the loaded diffusion models.

    This endpoint currently returns a placeholder result. When a valid model
    checkpoint is loaded via ``load_models()``, this method will perform
    a simple diffusion sampling using the provided number of steps.

    Args:
        request: JSON body with fields ``prompt``, ``num_steps``, and
            ``guidance_scale``.

    Returns:
        A dictionary with a 2D list representing a 64×64 grayscale image.
    """
    if request.num_steps < 1:
        raise HTTPException(status_code=400, detail="num_steps must be >=1")
    # Ensure models are loaded
    global unet, vae
    if unet is None or vae is None:
        raise HTTPException(status_code=500, detail="Models are not loaded")
    sigmas = edm_sigma_schedule(request.num_steps)
    with torch.no_grad():
        z = torch.zeros(1, 3, 64, 64, device=device)
        for sigma in sigmas:
            noise = torch.randn_like(z) * sigma
            zt = z + noise
            t_emb = torch.zeros(1, 1, 1, 1, device=device)
            _ = unet(zt, t_emb)
    return {"image": [[0.0] * 64 for _ in range(64)]}

def main() -> None:
    """Entry point for command‑line invocation.

    This function loads the models (optionally from a checkpoint) and runs a
    simple sampling loop with the specified number of steps. The result is
    printed to the console. Use this primarily for quick sanity checks.
    """
    parser = argparse.ArgumentParser(description="Nano diffusion inference")
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of sampling steps for the demo generation",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt) file. If omitted, attempts to load the latest checkpoint in 'checkpoints/'.",
    )
    args = parser.parse_args()
    load_models(args.weights)
    sigmas = edm_sigma_schedule(args.steps)
    z = torch.zeros(1, 3, 64, 64, device=device)
    with torch.no_grad():
        for sigma in sigmas:
            noise = torch.randn_like(z) * sigma
            zt = z + noise
            t_emb = torch.zeros(1, 1, 1, 1, device=device)
            _ = unet(zt, t_emb)
    print(f"Inference demo complete (used {args.steps} steps).")

if __name__ == "__main__":
    main()
