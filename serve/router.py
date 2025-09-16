"""
Router utilities for NanoDiffusion inference.

This module encapsulates simple policy logic to select which model pipeline to use,
how many denoising steps to run, whether to apply weight quantization, and what
output resolution to produce. The router can be extended to implement more
sophisticated policies based on system load, user preferences, or model
capabilities.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class InferencePolicy:
    """Structure describing the inference policy to apply."""
    pipeline: str
    steps: int
    quantize: bool
    resolution: int

def choose_policy(num_steps: int = 8,
                  quantize: bool = False,
                  resolution: int = 512) -> InferencePolicy:
    """
    Determine an inference policy based on simple heuristics.

    Args:
        num_steps: Desired number of denoising steps. Fewer steps imply using a
            distilled or accelerated model if available. Defaults to 8.
        quantize: Whether to apply weight-only quantization to the model.
        resolution: Target image resolution (square images assumed).

    Returns:
        InferencePolicy: Selected policy parameters.
    """
    # Heuristic: use a distilled pipeline for 4 or fewer steps.
    if num_steps <= 4:
        pipeline = "distilled"
    else:
        pipeline = "base"

    return InferencePolicy(
        pipeline=pipeline,
        steps=num_steps,
        quantize=quantize,
        resolution=resolution,
    )
