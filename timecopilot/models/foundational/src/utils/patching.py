def compute_num_patches(context_length: int, patch_size: int, stride: int) -> int:
    """
    Computes the number of time series patches that'll be extracted from
    the input sequence.

    Returns:
        int:  The number of patches that'll be generated from the input
        sequence.
    """
    return ((context_length - patch_size) // stride) + 2
