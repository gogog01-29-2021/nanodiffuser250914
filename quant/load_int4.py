"""
Utilities for applying weight-only INT4 quantization to nano diffusion models.

This module provides placeholder functions for converting model weights to 4-bit
precision. To use actual quantization backends such as `bitsandbytes` or
`quanto`, integrate those libraries here and wrap supported layers accordingly.
"""

from typing import Any


def quantize_int4(model: Any) -> Any:
    """
    Apply weight-only INT4 quantization to a model.

    This is a stub implementation. In a real implementation, you would convert
    supported layers (such as linear and convolutional layers) to 4-bit
    precision using an appropriate quantization library.

    Args:
        model: A PyTorch ``nn.Module`` (or similar) to quantize.

    Returns:
        The quantized model (same object reference).
    """
    # TODO: integrate actual INT4 quantization backend, e.g., bitsandbytes or quanto.
    return model


def load_int4_weights(model: Any, checkpoint_path: str) -> Any:
    """
    Load INT4-quantized weights from a checkpoint file into the given model.

    The checkpoint should contain weights that have been quantized to INT4
    precision. This stub does not implement loading; it serves as a placeholder
    to illustrate where such logic would go.

    Args:
        model: The model instance to load weights into.
        checkpoint_path: Path to the checkpoint file containing quantized weights.

    Returns:
        The model with weights loaded.
    """
    # TODO: implement loading of INT4 weights from checkpoint.
    raise NotImplementedError(
        "Loading INT4 weights is not implemented. Provide a backend-specific implementation."
    )
