"""Loss functions for nano diffusion.

Defines the v‑prediction MSE loss for diffusion models, the consistency
distillation loss, and a placeholder for flow‑matching loss.
"""

import torch
import torch.nn.functional as F


def v_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error between prediction and target.

    Args:
        pred: Predicted velocities from the model.
        target: Ground truth velocities.

    Returns:
        Scalar tensor of mean squared error.
    """
    return F.mse_loss(pred, target)


def consistency_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Consistency loss between student and teacher outputs.

    Args:
        student: Student network output.
        teacher: Teacher network output (e.g., from a longer solver).

    Returns:
        Mean squared error between student and teacher outputs.
    """
    return F.mse_loss(student, teacher)


def flow_matching_loss(flow: torch.Tensor) -> torch.Tensor:
    """Placeholder for flow‑matching loss computation.

    Args:
        flow: Velocity field tensor.

    Returns:
        Zero tensor; real implementation should compute appropriate loss.
    """
    return torch.tensor(0.0)
