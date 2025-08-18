from typing import Literal

from torch import Tensor
from transformers import GPT2Tokenizer
from transformers.tokenization_utils_base import BatchEncoding


def initialize_prompt_tokens(
    component: Literal["trend", "seasonal", "residual"],
    device: Literal["cpu", "cuda"],
) -> BatchEncoding:
    """
    Initializes GPT-2 prompt tokens for trend, seasonal, and residual components.

    Args:
        device (str): Device to move tokenized tensors to ("cpu" or "cuda").

    Returns:
        Dict[str, Dict[str, Tensor]]: Dictionary with keys "trend", "seasonal",
        and "residual", each containing a tokenized in
    """
    # Initialize a GPT=2 tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Tokenize the specified STL component's prompt
    prompt_tokens = gpt2_tokenizer(
        text=f"Predict the future time steps, given the {component}",
        return_tensors="pt",
    ).to(device)

    return prompt_tokens


def remove_prompt_tokens(
    hidden_past_target: Tensor,
    token_length: int,
    num_patches,
    use_token: bool,
    prompt: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Removes the prompt tokens from the given STL component hidden state.

    Args:
        hidden_past_target (Tensor): GPT-2 hidden state of the past target. Has
        shape (batch_size, sequence_length, emb_dim).
        token_length (int): Number of prompt tokens prepended per component.
        num_patches (_type_): Number of input patches per component.
        use_token (bool): Whether to use the representations of prompt tokens.
        prompt (bool): Whether prompt tokens were used at all.
    Returns:
        Tuple[Tensor, Tensor, Tensor]: Hidden states for trend, seasonal, and
        residual.
    """
    if not prompt:
        hidden_trend = hidden_past_target[:, :num_patches, :]
        hidden_seasonal = hidden_past_target[:, num_patches : 2 * num_patches, :]
        hidden_residual = hidden_past_target[:, 2 * num_patches :, :]

    if not use_token:
        pass

    # Compute each STL component's slice positions
    trend_end = token_length + num_patches

    seasonal_start = trend_end
    seasonal_end = seasonal_start + token_length + num_patches

    residual_start = seasonal_end

    # Remove prompt tokens
    hidden_trend = hidden_past_target[:, :trend_end, :]
    hidden_seasonal = hidden_past_target[:, seasonal_start:seasonal_end]
    hidden_residual = hidden_past_target[:, residual_start:, :]

    if not use_token:
        hidden_trend = hidden_trend[:, token_length:, :]
        hidden_seasonal = hidden_seasonal[:, token_length:, :]
        hidden_residual = hidden_residual[:, token_length:, :]

    return hidden_trend, hidden_seasonal, hidden_residual
