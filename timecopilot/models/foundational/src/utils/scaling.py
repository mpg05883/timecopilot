from torch import Tensor


def mean_abs_scaling(x: Tensor, min_scale: float = 1e-5) -> Tensor:
    """
    Computes a scaling factor based on the mean absolute value of the time
    series over the context window.

    Args:
        x (torch.Tensor): A 2D tensor of shape (batch_size,
        sequence_length) representing a batch of time series entries.
        min_scale (float): Minimum scale value to clamp to. Default is 1e-5.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 1) containing the scale
        for each time series.
    """
    # Compute the absolute value of each element in the context
    x_abs = x.abs()

    # Compute the mean along the time axis
    x_mean = x_abs.mean(1)

    # Ensure all the values are above min_scale
    clamped = x_mean.clamp(min=min_scale)

    # Add a singleton dimension so the output has shape (batch_size, 1)
    return clamped.unsqueeze(1)


def z_score_normalize(x: Tensor, dim=1, eps=1e-5):
    """
    Applies z-score normalization to a tensor along a specified dimension.

    Args:
        x (torch.Tensor): Input tensor to normalize.
        dim (int): Dimension along which to compute mean and standard
        deviation.
        eps (float): Small constant added to the denominator for numerical
        stability.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - The normalized tensor
            - The mean used for normalization
            - The standard deviation used for normalization
    """
    sample_mean = x.mean(dim, keepdim=True).detach()
    sample_std = x.std(dim, keepdim=True, unbiased=False).detach()
    return (x - sample_mean) / (sample_std + eps)
