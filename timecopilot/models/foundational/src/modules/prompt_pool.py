import torch
import torch.nn as nn
from torch import Tensor


class PromptPool(nn.Module):
    """
    A pool of learnable prompts. Prompts are selected dynamically based on
    TEMPO's current input. Each prompt is represented by a key and value
    embedding:
    - The key is used to select prompts most similar to the current input.
    - The value is the prompt embedding itself to prepend to the input.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_patches: int,
        pool_size: int = 30,
        prompt_length: int = 3,
        top_k: int = 3,
        diversify: bool = True,
    ):
        """
        Initializes the prompt pool by generating a set of learnable keys and
        value embeddings, and sets up a summary map to extract input summaries.

        Args:
            embedding_dim (int): _description_
            num_patches (int): _description_
            pool_size (int): _description_
            prompt_length (int): _description_
            top_k (int): _description_
            diversify (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.top_k = top_k
        self.diversify = diversify

        # Dictionaries of learnable key and value prompt embeddings
        self.prompt_key_dict = nn.ParameterDict()
        self.prompt_value_dict = nn.ParameterDict()

        # Projects the input patch embeddings into a summary vector
        self.summary_map = nn.Linear(num_patches, 1)

        for i in range(pool_size):
            self.prompt_key_dict[f"key_{i}"] = nn.Parameter(torch.randn(embedding_dim))
            self.prompt_value_dict[f"value_{i}"] = nn.Parameter(
                torch.randn(prompt_length, embedding_dim)
            )

        # For logging and interpretability of selected prompts
        self.prompt_record_plot = {}  # stores raw inputs and selected prompts
        self.prompt_record_id = 0  # id counter for logging prompt usage
        self.prompt_record = {
            f"id_{i}": 0 for i in range(self.pool_size)
        }  # usage counter
        self.trend_prompt_record = {}  # specific tracking for trend prompts
        self.seasonal_prompt_record = {}  # specific tracking for seasonal prompts
        self.residual_prompt_record = {}  # specific tracking for residual prompts
        self.diversify = True  # option to diversify prompt selection

    def select_top_k_prompts(
        self,
        summary: Tensor,
        prompt_mask: Tensor = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Selects the top-k most relevant prompts for a given input summary
        vector based on its cosine similarity with the prompt keys.

        Args:
            patch_embeddings (torch.Tensor): The patch embeddings for each
            time series. Shape: (batch_size, num_patches, embedding_dim)
            prompt_mask (Optional[torch.Tensor]): Optional mask specifying
            which prompts to use (useful for debugging or manual control).
            Shape: (batch_size, top_k)
        """
        prompt_key_matrix = torch.stack(
            tuple([self.prompt_key_dict[i] for i in self.prompt_key_dict.keys()])
        )
        prompt_norm = self.l2_normalize(prompt_key_matrix, dim=1)  # Pool_size, C
        summary_reshaped = summary.view(-1, self.patch_num)
        summary_mapped = self.summary_map(summary_reshaped)
        summary = summary_mapped.view(-1, 768)
        summary_embed_norm = self.l2_normalize(summary, dim=1)
        similarity = torch.matmul(summary_embed_norm, prompt_norm.t())
        if prompt_mask is not None:
            idx = prompt_mask
        else:
            _, idx = torch.topk(similarity, k=self.top_k, dim=1)
        if prompt_mask is None:
            count_of_keys = torch.bincount(torch.flatten(idx), minlength=15)
            for i in range(len(count_of_keys)):
                self.prompt_record[f"id_{i}"] += count_of_keys[i].item()

        prompt_value_matrix = torch.stack(
            tuple([self.prompt_value_dict[i] for i in self.prompt_value_dict.keys()])
        )
        batched_prompt_raw = prompt_value_matrix[idx].squeeze(1)
        batch_size, top_k, length, c = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)

        batched_key_norm = prompt_norm[idx]
        summary_embed_norm = summary_embed_norm.unsqueeze(1)
        sim = batched_key_norm * summary_embed_norm
        reduce_sim = torch.sum(sim) / summary.shape[0]

        # Return the sorted tuple of selected prompts along with batched_prompt and reduce_sim
        selected_prompts = [tuple(sorted(row)) for row in idx.tolist()]

        return batched_prompt, reduce_sim, selected_prompts
