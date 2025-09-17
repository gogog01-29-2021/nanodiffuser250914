"""Quantization package initialization.

This file allows Python to treat the ``quant`` directory as a package.
Expose any quantization backends or utility functions as needed.
"""

from .load_int8 import quantize_int8, load_int8_weights  # noqa: F401
from .load_int4 import quantize_int4, load_int4_weights  # noqa: F401

__all__ = ["quantize_int8", "load_int8_weights", "quantize_int4", "load_int4_weights"]
