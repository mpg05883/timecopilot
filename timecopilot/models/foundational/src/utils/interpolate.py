import torch.nn.functional as F
from torch import Tensor


def interpolate(hidden_state: Tensor, prediction_length: int) -> Tensor:
    """
    Interpolates the hidden state from num_patches to prediction_length.

    Args:
        hidden_state (Tensor): Shape (batch_size, num_patches, emb_dim).

    Returns:
            Tensor: Shape (batch_size, prediction_length, emb_dim)
    """
    # Permute hidden state to (batch_size, emb_dim, num_patches)
    hidden_state = hidden_state.permute(0, 2, 1)

    # Interpolate to prediction length
    hidden_state = F.interpolate(
        hidden_state,
        size=prediction_length,
        mode="linear",
        align_corners=False,
    )

    # Permute back to (batch_size, prediction_length, emb_dim)
    hidden_state = hidden_state.permute(0, 2, 1)

    return hidden_state
