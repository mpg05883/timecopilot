import os
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from gluonts.model.forecast_generator import (
    DistributionForecastGenerator,
    QuantileForecastGenerator,
)
from gluonts.torch.model.predictor import PyTorchPredictor
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from timecopilot.models.foundational.src.modules.gpt2 import initialize_gpt2_model
from timecopilot.models.foundational.src.modules.moving_average import MovingAverage
from timecopilot.models.foundational.src.modules.prompt_pool import PromptPool
from timecopilot.models.foundational.src.utils.decomposition import (
    compute_decomposition_loss,
    ensure_3d,
    have_all,
)
from timecopilot.models.foundational.src.utils.interpolate import interpolate
from timecopilot.models.foundational.src.utils.loss_utils import (
    get_criterion,
    get_distr_output,
    is_deterministic,
    is_quantile,
)
from timecopilot.models.foundational.src.utils.patching import compute_num_patches
from timecopilot.models.foundational.src.utils.prompting import (
    initialize_prompt_tokens,
    remove_prompt_tokens,
)
from timecopilot.models.foundational.src.utils.scaling import mean_abs_scaling
from torch import Tensor


class InstanceNorm(nn.Module):
    """
    See, also, RevIN. Apply standardization along the last dimension.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num(
                (x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0
            )
            scale = torch.where(scale == 0, torch.abs(loc) + self.eps, scale)
        else:
            loc, scale = loc_scale

        return (x - loc) / scale, (loc, scale)

    def inverse(
        self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        loc, scale = loc_scale
        return x * scale + loc


from transformers.models.gpt2.modeling_gpt2 import ACT2FN


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class TEMPO(nn.Module):
    """
    The time series forecasting model proposed in "TEMPO: Prompt-based
    Generative Pre-trained Transformer for Time Series Forecasting."
    """

    def __init__(self, config: DictConfig):
        """
        Initializes TEMPO's attributes and builds the model's subcomponents
        with the given configuration.

        Args:
            config (DictConfig): Configuration object containing model settings
            (e.g. loss function, prediction length, etc.).
        """
        super(TEMPO, self).__init__()

        # Time series attributes
        self.context_length = config.context_length
        self.prediction_length = config.prediction_length

        # Moving average and patch attributes
        self.kernel_size = config.kernel_size
        self.patch_size = config.patch_size
        self.stride = config.stride

        # GPT-2 attributes
        self.gpt = config.gpt
        self.num_gpt_layers = config.num_gpt_layers
        self.pretrained = config.pretrained
        self.freeze = config.freeze
        self.use_time_features = config.get("use_time_features", False)
        self.embedding_dim = config.embedding_dim

        # Prompt attributes
        self.prompt = config.prompt
        self.use_prompt = config.use_prompt
        self.prompt_pool = config.prompt_pool
        self.use_token = config.use_token

        # NOTE: The variable name "device" causes an error with Lightning
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Compute derived attributes
        self.num_patches = compute_num_patches(
            self.context_length,
            self.patch_size,
            self.stride,
        )

        # Loss and output settings
        self.loss_name = config.loss_name
        self.deterministic = is_deterministic(self.loss_name)
        self.is_quantile = is_quantile(self.loss_name)
        self.probabilistic = not self.deterministic and not self.is_quantile

        # Set decomposition loss function
        self.decomposition_criterion = nn.MSELoss()

        # Set loss function (deterministic) or distribution (probabilistic)
        if self.deterministic:
            self.criterion = get_criterion(self.loss_name)
        elif self.probabilistic and self.use_time_features:
            self.distr_output = get_distr_output(self.loss_name)
            self.args_proj = self.distr_output.get_args_proj(
                self.embedding_dim + 28
            ).to(self._device)
        elif self.probabilistic:
            self.distr_output = get_distr_output(self.loss_name)
            self.args_proj = self.distr_output.get_args_proj(self.embedding_dim).to(
                self._device
            )
        else:
            self.num_quantiles = len(config.quantiles)
            self.quantiles = torch.tensor(config.quantiles, dtype=self.dtype)

        self.norm_style = config.norm_style
        self.is_prob_quant = config.is_prob_quant
        self.is_dec_loss = config.is_dec_loss

        if self.norm_style == "instancenorm":
            self.norm = InstanceNorm()

        self.dropout_rate = config.dropout_rate
        self.dense_act_fn = config.dense_act_fn

        # Build model subcomponents and modules

        # if isinstance(module, ResidualBlock):
        #     module.hidden_layer.weight.data.normal_(
        #         mean=0.0,
        #         std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
        #     )
        #     if hasattr(module.hidden_layer, "bias") and module.hidden_layer.bias is not None:
        #         module.hidden_layer.bias.data.zero_()

        #     module.residual_layer.weight.data.normal_(
        #         mean=0.0,
        #         std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
        #     )
        #     if hasattr(module.residual_layer, "bias") and module.residual_layer.bias is not None:
        #         module.residual_layer.bias.data.zero_()

        #     module.output_layer.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
        #     if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
        #         module.output_layer.bias.data.zero_()

        self.build()

    # ========================= Public Methods =========================
    def build(self) -> None:
        """
        Builds the following model subcomponents and modules:
        - Scaling function
        - Moving average module
        - Patch padding layer
        - Local STL layers
        - Patch embedding layers
        - GPT-2 model and tokenizers
        - Prompt layers
        - Prompt pool
        - Decoder layers for deterministic forecasting
        - Expansion layers for probabilistic forecasting
        """
        self.scaling = mean_abs_scaling
        self.moving_average = MovingAverage(kernel_size=25, stride=1)
        self.patch_padding_layer = nn.ReplicationPad1d((0, self.stride))

        # Initialize layers to project from raw to normalized local STL
        # components.
        self.local_trend_layer = nn.Linear(
            self.context_length,
            self.context_length,
        )
        self.local_seasonal_layer = nn.Sequential(
            nn.Linear(self.context_length, 4 * self.context_length),
            nn.ReLU(),
            nn.Linear(4 * self.context_length, self.context_length),
        )
        self.local_residual_layer = nn.Linear(
            self.context_length,
            self.context_length,
        )

        if self.use_time_features:
            self.local_trend_with_time_feat = nn.Linear(29, 1)
            self.local_seasonal_with_time_feat = nn.Linear(29, 1)
            self.local_residual_with_time_feat = nn.Linear(29, 1)
        # Initialize layers to project time series patches into embedding
        # space.
        self.trend_patch_embedding_layer = nn.Linear(
            self.patch_size,
            self.embedding_dim,
        )
        self.seasonal_patch_embedding_layer = nn.Linear(
            self.patch_size,
            self.embedding_dim,
        )
        self.residual_patch_embedding_payer = nn.Linear(
            self.patch_size,
            self.embedding_dim,
        )

        if self.gpt:
            # Initialize GPT-2 model and each STL component's tokens
            self.gpt2_model = initialize_gpt2_model(
                self.pretrained,
                self.num_gpt_layers,
                self.freeze,
                self._device,
            )
            self.trend_prompt_tokens = initialize_prompt_tokens(
                "trend",
                self._device,
            )
            self.seasonal_prompt_tokens = initialize_prompt_tokens(
                "seasonal",
                self._device,
            )
            self.residual_prompt_tokens = initialize_prompt_tokens(
                "seasonal",
                self._device,
            )
            self.token_length = len(self.trend_prompt_tokens["input_ids"][0])

        if self.gpt and self.prompt:
            # Initialize STL component prompt layers
            self.trend_prompt_layer = nn.Linear(
                self.embedding_dim,
                self.embedding_dim,
            )
            self.seasonal_prompt_layer = nn.Linear(
                self.embedding_dim,
                self.embedding_dim,
            )
            self.residual_prompt_layer = nn.Linear(
                self.embedding_dim,
                self.embedding_dim,
            )

        if self.gpt and self.prompt and self.prompt_pool:
            # Initialize prompt pool
            self.prompt_pool = PromptPool(self.embedding_dim, self.num_patches)

        if self.probabilistic or self.is_quantile:
            # Initialize layers to expand interpolated STL component hidden
            # states
            self.trend_expansion_layer = nn.Linear(
                self.embedding_dim,
                self.embedding_dim,
            )
            self.seasonal_expansion_layer = nn.Linear(
                self.embedding_dim,
                self.embedding_dim,
            )
            self.residual_expansion_layer = nn.Linear(
                self.embedding_dim,
                self.embedding_dim,
            )
            if self.use_time_features:
                self.time_features_expansion_layer = nn.Linear(
                    self.embedding_dim + 28,
                    self.embedding_dim,
                )
            if self.is_quantile:
                self.output_patch_embedding = ResidualBlock(
                    in_dim=self.embedding_dim,
                    h_dim=self.embedding_dim,
                    out_dim=self.num_quantiles,
                    act_fn_name=self.dense_act_fn,
                    dropout_p=self.dropout_rate,
                )
            return

        # Initialize layers to decode GPT-2 hidden states from hidden space
        # back to time series space for deterministic forecasting.
        if self.prompt and self.use_prompt:
            in_features = self.embedding_dim * (self.num_patches + self.token_length)
        else:
            in_features = self.embedding_dim * self.num_patches

        self.trend_decoder_layer = nn.Linear(
            in_features,
            self.prediction_length,
        )
        self.seasonal_decoder_layer = nn.Linear(
            in_features,
            self.prediction_length,
        )
        self.residual_decoder_layer = nn.Linear(
            in_features,
            self.prediction_length,
        )

    @classmethod
    def load_pretrained_model(
        cls,
        _device,
        cfg=None,
        repo_id="Melady/TEMPO",
        filename="TEMPO-80M_v1.pth",
        cache_dir="./checkpoints/TEMPO_checkpoints",
    ):
        """
        Loads a pre-trained TEMPO model from Hugging Face.
        """
        # Download the model's checkpoint
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )

        # Download configs.json
        configs_path = hf_hub_download(
            repo_id=repo_id,
            filename="configs.json",
            cache_dir=cache_dir,
        )

        # Load configuration file
        if cfg is None:
            cfg = OmegaConf.load(configs_path)

        # Initialize model
        model = cls(cfg)

        # Construct full path to checkpoint
        model_path = os.path.join(cfg.checkpoints, cfg.model_id)
        best_model_path = model_path + "_checkpoint.pth"
        print(f"Loading model from: {best_model_path}")

        # Load state dict
        state_dict = torch.load(checkpoint_path, map_location=_device)
        return model.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        past_target: Tensor,
        past_trend: Tensor = None,
        past_seasonal: Tensor = None,
        past_residual: Tensor = None,
        past_time_feat: Tensor | None = None,
        future_time_feat: Tensor | None = None,
        test: bool = True,
    ):
        """
        Performs a forward pass of the TEMPO model.

        Args:
            past_target (Tensor): Ground truth values over the context window.
            past_trend (Tensor): Trend component over the context window.
            past_seasonal (Tensor): Seasonal component over the context window.
            past_residual (Tensor): Residual component over the context window.
            test (bool): If True, the decomposition loss is returned.

        Returns:
            If the model is deterministic:
                Tensor: Forecasted values.
            Else:
                Tuple: Distribution arguments at each timestep of the
                prediction length.
        """

        # Compute scaling factor
        # Normalize past target and components
        if self.norm_style == "instancenorm":
            past_target, (loc, scale) = self.norm(past_target)

        else:

            scale = self.scaling(past_target)
            loc = torch.zeros_like(scale)
            past_target = past_target / scale

        # past_target = past_target / scale

        # Ensure input has shape (batch_size, context_length, num_features)
        past_target = ensure_3d(past_target)
        past_trend = ensure_3d(past_trend)
        past_seasonal = ensure_3d(past_seasonal)
        past_residual = ensure_3d(past_residual)

        batch_size, _, num_features = past_target.shape

        # Decompose past target into local trend, seasonal, and residual
        local_trend, local_seasonal, local_residual = self._local_stl_decompose(
            past_target
        )

        # True if all past STL components are not None
        have_all_components = have_all(past_trend, past_seasonal, past_residual)
        # import pdb; pdb.set_trace()

        # Only compute decomposition loss during training and validation and if
        # all past STl components are given
        if have_all_components and not test:
            if self.is_dec_loss:
                if self.norm_style == "instancenorm":
                    past_trend, _ = self.norm(past_trend)
                    past_seasonal, _ = self.norm(past_seasonal)
                    past_residual, _ = self.norm(past_residual)
                else:
                    scale_trens = self.scaling(past_trend)
                    past_trend = past_trend / scale_trens
                    scale_seasonal = self.scaling(past_seasonal)
                    past_seasonal = past_seasonal / scale_seasonal
                    scale_residual = self.scaling(past_residual)
                    past_residual = past_residual / scale_residual

                true_stl_components = (past_target, past_seasonal, past_residual)
                local_stl_components = (local_trend, local_seasonal, local_residual)

                decomposition_loss = compute_decomposition_loss(
                    true_stl_components,
                    local_stl_components,
                    scale,
                )
            else:
                decomposition_loss = 0

        if self.use_time_features:
            # import pdb; pdb.set_trace()
            local_trend = self.local_trend_with_time_feat(
                torch.cat((local_trend, past_time_feat), dim=2)
            )
            local_seasonal = self.local_seasonal_with_time_feat(
                torch.cat((local_seasonal, past_time_feat), dim=2)
            )
            local_residual = self.local_residual_with_time_feat(
                torch.cat((local_residual, past_time_feat), dim=2)
            )
        # Patchify each STL component
        patched_trend = self._patchify(local_trend)
        patched_seasonal = self._patchify(local_seasonal)
        patched_residual = self._patchify(local_residual)

        # Project time series patches into embedding space
        trend_patch_embeds = self.trend_patch_embedding_layer(patched_trend)
        seasonal_patch_embeds = self.seasonal_patch_embedding_layer(patched_seasonal)
        residual_patch_embeds = self.residual_patch_embedding_payer(patched_residual)

        # Concatenate prompts to each STL component's embedding
        trend_patch_embeds = self._concat_prompts(trend_patch_embeds, "trend")
        seasonal_patch_embeds = self._concat_prompts(seasonal_patch_embeds, "seasonal")
        residual_patch_embeds = self._concat_prompts(residual_patch_embeds, "residual")

        # Concatenate all patch embeddings
        inputs_embeds = torch.cat(
            (trend_patch_embeds, seasonal_patch_embeds, residual_patch_embeds),
            dim=1,
        )

        # Perform GPT-2 forward pass to get past target hidden state
        hidden_past_target = self.gpt2_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
        ).last_hidden_state

        # Remove prompt tokens from each STL component's hidden state
        hidden_trend, hidden_seasonal, hidden_residual = remove_prompt_tokens(
            hidden_past_target,
            self.token_length,
            self.num_patches,
            self.use_token,
            self.prompt,
        )

        # If deterministic, project hidden states back to time series.
        if self.deterministic:
            # Reshape hidden STL components
            hidden_trend = hidden_trend.reshape(batch_size * num_features, -1)
            hidden_seasonal = hidden_seasonal.reshape(batch_size * num_features, -1)
            hidden_residual = hidden_residual.reshape(batch_size * num_features, -1)

            # Get forecasted STL components
            forecast_trend = self.trend_decoder_layer(hidden_trend)
            forecast_seasonal = self.seasonal_decoder_layer(hidden_seasonal)
            forecast_residual = self.residual_decoder_layer(hidden_residual)

            # Reshape forecasted STL components from (batch_size * num_features,
            # prediction_length) to (batch_size, prediction_length, num_features)
            forecast_trend = rearrange(
                forecast_trend,
                "(b m) l -> b l m",
                b=batch_size,
            )
            forecast_seasonal = rearrange(
                forecast_seasonal,
                "(b m) l -> b l m",
                b=batch_size,
            )
            forecast_residual = rearrange(
                forecast_residual,
                "(b m) l -> b l m",
                b=batch_size,
            )

            # Construct forecast
            forecasts = forecast_trend + forecast_residual + forecast_seasonal

            # Rescale output
            forecasts *= scale

            # Only return decomposition loss during training and validation
            outputs = forecasts if not test else (forecasts, decomposition_loss)

            return outputs

        elif self.probabilistic:
            # Interpolate hidden states to match prediction length
            interpolated_trend = interpolate(hidden_trend, self.prediction_length)
            interpolated_seasonal = interpolate(hidden_seasonal, self.prediction_length)
            interpolated_residual = interpolate(hidden_residual, self.prediction_length)

            # Apply linear projection to expand interpolated STL components
            expanded_trend = self.trend_expansion_layer(interpolated_trend)
            expanded_seasonal = self.seasonal_expansion_layer(interpolated_seasonal)
            expanded_residual = self.residual_expansion_layer(interpolated_residual)

            # Combine expanded STL components
            expanded = expanded_trend + expanded_seasonal + expanded_residual
            # Get distribution arguments at each future timestep
            if self.use_time_features:
                # # expanded = self.time_features_expansion_layer(torch.cat((expanded, future_time_feat), dim=2))
                # print("past_time_feat")
                # print(past_time_feat.shape)
                # print("future_time_feat")
                # print(future_time_feat.shape)
                # print("*"*100)
                # import pdb; pdb.set_trace()
                """
                if expanded.shape[1] != self.prediction_length:
                    interpolate_expanded = interpolate(expanded, self.prediction_length)
                else:
                    interpolate_expanded = expanded
                interpolate_expanded = interpolate(expanded, 12)
                future_time_feat = pd.read_csv("pde_time.csv").values[-12:, 1:].astype(np.float32)
                future_time_feat = torch.from_numpy(future_time_feat).float().to(self._device)
                future_time_feat = future_time_feat.repeat(batch_size, 1, 1)
                # import pdb; pdb.set_trace()
                expanded = interpolate_expanded
                """

                # import pdb; pdb.set_trace()
                expanded = torch.cat((expanded, future_time_feat), dim=2)
            distr_args = self.args_proj(expanded)
            if loc is not None:
                loc = torch.zeros_like(scale)

            # Only return decomposition loss during training and validation
            outputs = (
                (distr_args, loc, scale, decomposition_loss)
                if not test
                else (distr_args, loc, scale)
            )

            return outputs
        else:
            # Interpolate hidden states to match prediction length
            interpolated_trend = interpolate(hidden_trend, self.prediction_length)
            interpolated_seasonal = interpolate(hidden_seasonal, self.prediction_length)
            interpolated_residual = interpolate(hidden_residual, self.prediction_length)

            # Apply linear projection to expand interpolated STL components
            expanded_trend = self.trend_expansion_layer(interpolated_trend)
            expanded_seasonal = self.seasonal_expansion_layer(interpolated_seasonal)
            expanded_residual = self.residual_expansion_layer(interpolated_residual)

            # Combine expanded STL components
            expanded = expanded_trend + expanded_seasonal + expanded_residual

            # Get distribution arguments at each future timestep
            # import pdb; pdb.set_trace()
            quantile_preds_shape = (
                batch_size,
                self.num_quantiles,
                self.prediction_length,
            )
            # import pdb; pdb.set_trace()
            quantile_preds = self.output_patch_embedding(expanded).view(
                *quantile_preds_shape
            )
            if not test:
                outputs = ((quantile_preds,), loc, scale), decomposition_loss
            else:
                outputs = (
                    (
                        quantile_preds.reshape(
                            batch_size, self.prediction_length, self.num_quantiles
                        ),
                    ),
                    loc,
                    scale,
                )
            return outputs

    def get_predictor(
        self,
        input_transform,
        batch_size: int,
        prediction_length: int,
        verbose: bool = True,
    ) -> PyTorchPredictor:
        """
        Wraps the model in a GluonTS PyTorchPredictor for forecasting.

        Args:
            input_transform: GluonTS transform applied to input data.
            prediction_length (int): Number of future time steps to predict.
            batch_size (int): Number of samples per batch during inference.
            verbose (bool): Set to True to print which device the predictor is
            loaded on (Defaults to True).

        Returns:
            PyTorchPredictor: A predictor instance compatible with GluonTS.
        """
        self.prediction_length = prediction_length
        if verbose:
            print(f"Loading predictor on {self._device}...")
            print(f"Prediction length set to {self.prediction_length}")

        input_names = [
            "past_target",
            "past_trend",
            "past_seasonal",
            "past_residual",
            "past_time_feat",
            "future_time_feat",
        ]
        if self.probabilistic:
            return PyTorchPredictor(
                prediction_length=self.prediction_length,
                input_names=input_names,
                prediction_net=self,
                batch_size=batch_size,
                input_transform=input_transform,
                forecast_generator=DistributionForecastGenerator(self.distr_output),
            ).to(self._device)
        elif self.is_quantile:
            q_levels = [f"{q:.1f}" for q in np.linspace(0.1, 0.9, self.num_quantiles)]
            return PyTorchPredictor(
                prediction_length=self.prediction_length,
                input_names=input_names,
                prediction_net=self,
                batch_size=batch_size,
                input_transform=input_transform,
                forecast_generator=QuantileForecastGenerator(q_levels),
            ).to(self._device)

    # ========================= Private Methods =========================
    def _patchify(self, x: Tensor) -> Tensor:
        """
        Splits the input time series into patches.

        Args:
            x (Tensor): Input tensor of shape (batch_size, context_length).

        Returns:
            Tensor: Patches of shape (batch_size, num_patches, num_nodes,
            patch_size).
        """
        x = rearrange(x, "b l m -> b m l")
        x = self.patch_padding_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        return rearrange(x, "b m n p -> (b m) n p")

    def _concat_prompts(
        self,
        patch_embeds: Tensor,
        component_name: Literal["trend", "seasonal", "residual"],
    ) -> Tensor:
        """
        Concatenates the corresponding prompt to the given STL component's
        patch embeddings
        """
        if self.gpt and self.prompt:
            return self._get_embeds(
                patch_embeds,
                component_name,
                self.trend_prompt_tokens["input_ids"],
            )
        else:
            return self._get_embeds(patch_embeds)

    def _get_embeds(
        self,
        stl_component: Tensor,
        component_name: Literal["trend", "seasonal", "residual"],
        tokens=None,
    ):
        """_summary_

        Args:
            stl_component (Tensor): _description_
            component (Literal[&quot;&quot;]): _description_
            tokens (_type_, optional): _description_. Defaults to None.
        """
        if tokens is None:
            return self.gpt2_model(inputs_embeds=stl_component).last_hidden_state

        batch_size, _, _ = stl_component.shape

        if component_name == "trend":
            if self.prompt_pool:
                stl_component_prompt, reduce_sim, selected_trend_prompts = (
                    self.prompt_pool.select_top_k_prompts(
                        stl_component,
                        prompt_mask=None,
                    )
                )
                for selected_trend_prompt in selected_trend_prompts:
                    self.prompt_pool.trend_prompt_record[selected_trend_prompt] = (
                        self.prompt_pool.trend_prompt_record.get(
                            selected_trend_prompt, 0
                        )
                        + 1
                    )
                selected_prompts = selected_trend_prompts
            else:
                stl_component_prompt = self.gpt2_model.wte(tokens)
                stl_component_prompt = stl_component_prompt.repeat(batch_size, 1, 1)
                stl_component_prompt = self.trend_prompt_layer(stl_component_prompt)
            stl_component = torch.cat((stl_component_prompt, stl_component), dim=1)
        elif component_name == "seasonal":
            if self.prompt_pool:
                stl_component_prompt, reduce_sim, selected_seasonal_prompts = (
                    self.prompt_pool.select_top_k_prompts(
                        stl_component,
                        prompt_mask=None,
                    )
                )
                for selected_seasonal_prompt in selected_seasonal_prompts:
                    self.prompt_pool.seasonal_prompt_record[
                        selected_seasonal_prompt
                    ] = (
                        self.prompt_pool.seasonal_prompt_record.get(
                            selected_seasonal_prompt, 0
                        )
                        + 1
                    )
                selected_prompts = selected_seasonal_prompts
            else:
                stl_component_prompt = self.gpt2_model.wte(tokens)
                stl_component_prompt = stl_component_prompt.repeat(batch_size, 1, 1)
                stl_component_prompt = self.seasonal_prompt_layer(stl_component_prompt)
            stl_component = torch.cat((stl_component_prompt, stl_component), dim=1)
        else:
            if self.prompt_pool:
                stl_component_prompt, reduce_sim, selected_residual_prompts = (
                    self.prompt_pool.select_top_k_prompts(
                        stl_component,
                        prompt_mask=None,
                    )
                )
                for selected_residual_prompt in selected_residual_prompts:
                    self.prompt_pool.residual_prompt_record[
                        selected_residual_prompt
                    ] = (
                        self.prompt_pool.residual_prompt_record.get(
                            selected_residual_prompt, 0
                        )
                        + 1
                    )
                selected_prompts = selected_residual_prompts
            else:
                stl_component_prompt = self.gpt2_model.wte(tokens)
                stl_component_prompt = stl_component_prompt.repeat(batch_size, 1, 1)
                stl_component_prompt = self.residual_prompt_layer(stl_component_prompt)
            stl_component = torch.cat((stl_component_prompt, stl_component), dim=1)

        if self.prompt_pool:
            return stl_component, reduce_sim, selected_prompts
        else:
            return stl_component

    def _local_stl_decompose(
        self,
        past_target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Decomposes the past target into its trend, seasonal, and residual
        components.

        Args:
            past_target (Tensor): The input time series.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the estimated
            trend, seasonal, and residual components over the context window.
        """
        # Estimate trend over the context window
        local_trend = self.moving_average(past_target)

        # Perform linear transformation on local trend
        local_trend = local_trend.squeeze(2)
        local_trend = self.local_trend_layer(local_trend)
        local_trend = local_trend.unsqueeze(2)

        # Estimate seasonal over the context window
        local_seasonal = past_target - local_trend

        # Perform non-linear transformation on local seasonal
        local_seasonal = local_seasonal.squeeze(2)
        local_seasonal = self.local_seasonal_layer(local_seasonal)
        local_seasonal = local_seasonal.unsqueeze(2)

        # Estimate local residual over the context window
        local_residual = past_target - local_trend - local_seasonal

        return local_trend, local_seasonal, local_residual
