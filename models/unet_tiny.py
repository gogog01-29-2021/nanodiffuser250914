"""A minimal U-Net architecture for tiny diffusion models.

This tiny U-Net down- and up-samples the input latent z_t. It uses
residual blocks with group normalization and SiLU activations. The model
expects z_t of shape [batch, in_ch, H, W] and returns a residual
prediction in the same shape.

Args:
    width (int): base channel count for all residual blocks.
    in_ch (int): number of input channels (latent dimensions).
    cond_ch (int): optional conditioning channels (unused).
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A simple residual block with GroupNorm and SiLU activation."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to the input."""
        return x + self.net(x)


class UNetTiny(nn.Module):
    """Tiny U-Net for latent diffusion.

    This architecture uses a single scale with four down blocks, a middle
    block, and four up blocks. Skip connections join each down and up block.

    Args:
        width (int): base channel count for the residual blocks.
        in_ch (int): number of channels in the input latent.
        cond_ch (int): number of conditioning channels (unused).
    """

    def __init__(self, width: int = 384, in_ch: int = 4, cond_ch: int = 0):
        super().__init__()
        self.inp = nn.Conv2d(
            in_ch, width, kernel_size=3, stride=1, padding=1
        )
        self.down = nn.ModuleList([ResidualBlock(width) for _ in range(4)])
        self.mid = ResidualBlock(width)
        self.up = nn.ModuleList([ResidualBlock(width) for _ in range(4)])
        self.outp = nn.Conv2d(
            width, in_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(
        self,
        zt: torch.Tensor,
        t_emb: torch.Tensor,
        cond: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply the tiny U-Net to a latent.

        Args:
            zt (Tensor): noised latent input at time t.
            t_emb (Tensor): time embedding (ignored here but kept for API consistency).
            cond (Tensor, optional): extra conditioning input (unused).

        Returns:
            Tensor: the predicted residual latent.
        """
        x = self.inp(zt)
        skips = []
        for blk in self.down:
            x = blk(x)
            skips.append(x)
        x = self.mid(x)
        for blk, s in zip(self.up, reversed(skips)):
            x = blk(x + s)
        return self.outp(x)
