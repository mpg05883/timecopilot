from typing import Literal

from peft import LoraConfig, get_peft_model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


def freeze_gpt2_model(gpt2_model: GPT2Model) -> None:
    """
    Freezes all parameters in the GPT-2 model except for LayerNorms (ln) and
    positional embeddings (wpe).

    Args:
        gpt2_model (GPT2Model): The GPT-2 model to modify in-place.
    """
    for name, param in gpt2_model.named_parameters():
        if "ln" not in name and "wpe" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def initialize_gpt2_model(
    pretrained: bool,
    num_layers: int,
    freeze: bool,
    device: Literal["cpu", "cuda"],
) -> GPT2Model:
    """
    Initializes a GPT-2 model with optional pre-training, layer truncation,
    LoRA adaptation, and parameter freezing.

    Args:
        pretrained (bool): If True, loads the pretrained GPT-2 weights from
        Hugging Face. Else, initializes GPT-2 with random weights.
        num_layers (int): The number of transformer layers to keep in the GPT-2
        model. The model is truncated to this number of layers.
        freeze (bool): If True and `pretrained` is True, all parameters except
        layer norms and positional embeddings are frozen (non-trainable).
        device (str): The device to move the model to (e.g., "cpu" or "cuda").

    Returns:
        GPT2Model: A configured GPT-2 model.
    """
    if pretrained:
        # Load pre-trained GPT-2 model
        gpt2_model = GPT2Model.from_pretrained(
            "gpt2",
            output_attentions=True,
            output_hidden_states=True,
            attn_implementation="eager",
        )
    else:
        # Load randomly initialized GPT-2 model
        gpt2_config = GPT2Config()
        gpt2_model = GPT2Model(gpt2_config)

    # Move GPT-2 model to device
    gpt2_model = gpt2_model.to(device)

    # Set number of transformer blocks
    gpt2_model.h = gpt2_model.h[:num_layers]

    # Initialize LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="lora_only",
        fan_in_fan_out=True,
    )

    # Wrap GPT-2 model with LoRA modules
    gpt2_model = get_peft_model(gpt2_model, lora_config)

    # Freeze the GPT2 model except layer norms and position embeddings
    if freeze and pretrained:
        freeze_gpt2_model(gpt2_model)

    return gpt2_model
