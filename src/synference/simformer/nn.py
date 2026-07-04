"""Neural network components for the PyTorch Simformer.

Faithful port of the score-transformer architecture from Gloeckler et al. (2024),
"All-in-one simulation-based inference" (original JAX/haiku implementation in
``probjax``/``scoresbibm``). Each scalar variable is one token; the network predicts
the (scaled) score of the noised joint distribution at diffusion time ``t``.
"""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
"""Correction factor so a +/-2 sigma truncated normal has the requested stddev."""


def variance_scaling_init_(tensor: torch.Tensor, scale: float, fan_in: int) -> torch.Tensor:
    """Initialize ``tensor`` in-place with haiku-style fan-in variance scaling.

    Draws from a truncated normal (+/- 2 stddev) with stddev ``sqrt(scale / fan_in)``,
    matching ``hk.initializers.VarianceScaling(scale)`` defaults.

    Args:
        tensor: Tensor to initialize in-place.
        scale: Variance scale factor.
        fan_in: Number of input units.

    Returns:
        The initialized tensor.
    """
    stddev = math.sqrt(scale / max(1.0, fan_in)) / TRUNCATED_NORMAL_STDDEV_FACTOR
    return nn.init.trunc_normal_(tensor, mean=0.0, std=stddev, a=-2 * stddev, b=2 * stddev)


class GaussianFourierEmbedding(nn.Module):
    """Gaussian Fourier feature embedding, mostly used to embed diffusion time.

    The random projection matrix ``B`` is drawn once at initialization and frozen
    (registered as a buffer so it is saved/restored with the state dict), matching
    the stop-gradient behaviour of the original implementation.
    """

    def __init__(self, output_dim: int = 128, input_dim: int = 1):
        """Initialize the embedding.

        Args:
            output_dim: Output embedding dimension.
            input_dim: Dimension of the input (1 for scalar time).
        """
        super().__init__()
        self.output_dim = output_dim
        half_dim = output_dim // 2 + 1
        self.register_buffer("B", torch.randn(half_dim, input_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Embed ``inputs`` of shape ``(..., input_dim)`` to ``(..., output_dim)``.

        Args:
            inputs: Input tensor with the projected dimension last.

        Returns:
            Fourier features ``[cos(2 pi x B^T), sin(2 pi x B^T)]`` truncated to
            ``output_dim``.
        """
        proj = 2 * math.pi * inputs @ self.B.T
        out = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return out[..., : self.output_dim]


class ScalarTokenizer(nn.Module):
    """Tokenize scalar variables into embedding vectors.

    Each token is the concatenation of a (frozen) learned node-id embedding and the
    scalar value tiled to the remaining width. The node-id embedding is initialized
    orthogonally and never trained, replicating the ``stop_gradient`` applied in the
    original ``probjax`` tokenizer.
    """

    def __init__(self, output_dim: int, num_nodes: int):
        """Initialize the tokenizer.

        Args:
            output_dim: Total token dimension (split between id and value parts).
            num_nodes: Maximum number of distinct node ids.
        """
        super().__init__()
        self.output_dim = output_dim
        self.id_dim = output_dim // 2
        self.value_dim = output_dim - self.id_dim
        self.node_embedding = nn.Embedding(num_nodes, self.id_dim)
        nn.init.orthogonal_(self.node_embedding.weight, gain=0.5)
        # The original implementation stop-gradients the node embedding, so it stays
        # at its random orthogonal initialization.
        self.node_embedding.weight.requires_grad_(False)

    def forward(self, node_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Tokenize values.

        Args:
            node_ids: Integer node ids of shape ``(num_nodes,)`` or ``(B, num_nodes)``.
            values: Scalar values of shape ``(B, num_nodes, 1)``.

        Returns:
            Tokens of shape ``(B, num_nodes, output_dim)``.
        """
        batch, num_nodes, _ = values.shape
        node_ids = node_ids.reshape(-1, num_nodes).long()
        id_embedding = self.node_embedding(node_ids)  # (1 or B, num_nodes, id_dim)
        id_embedding = id_embedding.expand(batch, num_nodes, self.id_dim)
        value_embedding = values.expand(batch, num_nodes, self.value_dim)
        return torch.cat([id_embedding, value_embedding], dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with an optional boolean attention mask.

    ``mask[i, j] = True`` means token ``i`` may attend to token ``j``. Masked logits
    are set to ``-1e30`` before the softmax, as in the original implementation.
    """

    def __init__(self, model_size: int, num_heads: int, key_size: int, init_scale: float):
        """Initialize projections.

        Args:
            model_size: Token embedding width.
            num_heads: Number of attention heads.
            key_size: Per-head key/query/value size.
            init_scale: Variance-scaling factor for weight init.
        """
        super().__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        inner = num_heads * key_size
        self.query_proj = nn.Linear(model_size, inner)
        self.key_proj = nn.Linear(model_size, inner)
        self.value_proj = nn.Linear(model_size, inner)
        self.out_proj = nn.Linear(inner, model_size)
        for layer in (self.query_proj, self.key_proj, self.value_proj):
            variance_scaling_init_(layer.weight, init_scale, model_size)
            nn.init.zeros_(layer.bias)
        variance_scaling_init_(self.out_proj.weight, init_scale, inner)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply masked self-attention.

        Args:
            x: Tokens of shape ``(B, T, model_size)``.
            mask: Optional boolean mask of shape ``(T, T)`` or ``(B, T, T)``.

        Returns:
            Attended tokens of shape ``(B, T, model_size)``.
        """
        batch, num_tokens, _ = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(batch, num_tokens, self.num_heads, self.key_size).transpose(1, 2)

        q = split_heads(self.query_proj(x))  # (B, H, T, K)
        k = split_heads(self.key_proj(x))
        v = split_heads(self.value_proj(x))

        logits = q @ k.transpose(-2, -1) / math.sqrt(self.key_size)  # (B, H, T, T)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None, :, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            else:
                raise ValueError(f"Mask must have ndim 2 or 3, got {mask.ndim}.")
            logits = logits.masked_fill(~mask, -1e30)
        weights = F.softmax(logits, dim=-1)
        attn = weights @ v  # (B, H, T, K)
        attn = attn.transpose(1, 2).reshape(batch, num_tokens, -1)
        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block with time context injected in the MLP.

    Structure (matching ``probjax.nn.transformers.Transformer``)::

        h = LN(h)
        h = h + MHA(h, mask)
        h = LN(h)
        h = h + (
            MLP(h)
            + gelu(
                Linear(time)
            )
        )
    """

    def __init__(
        self,
        model_size: int,
        num_heads: int,
        key_size: int,
        widening_factor: int,
        num_hidden_layers: int,
        time_dim: int,
        init_scale: float,
    ):
        """Initialize the block.

        Args:
            model_size: Token embedding width.
            num_heads: Number of attention heads.
            key_size: Per-head attention size.
            widening_factor: MLP hidden width multiplier.
            num_hidden_layers: Number of hidden layers in the MLP.
            time_dim: Dimension of the time-context embedding.
            init_scale: Variance-scaling factor for weight init.
        """
        super().__init__()
        self.ln_attn = nn.LayerNorm(model_size)
        self.attention = MultiHeadAttention(model_size, num_heads, key_size, init_scale)
        self.ln_mlp = nn.LayerNorm(model_size)

        hidden = widening_factor * model_size
        mlp_layers: list[nn.Module] = []
        in_dim = model_size
        for _ in range(num_hidden_layers):
            layer = nn.Linear(in_dim, hidden)
            variance_scaling_init_(layer.weight, init_scale, in_dim)
            nn.init.zeros_(layer.bias)
            mlp_layers += [layer, nn.GELU()]
            in_dim = hidden
        out_layer = nn.Linear(in_dim, model_size)
        variance_scaling_init_(out_layer.weight, init_scale, in_dim)
        nn.init.zeros_(out_layer.bias)
        mlp_layers.append(out_layer)
        self.mlp = nn.Sequential(*mlp_layers)

        self.time_proj = nn.Linear(time_dim, model_size)
        variance_scaling_init_(self.time_proj.weight, init_scale, time_dim)
        nn.init.zeros_(self.time_proj.bias)

    def forward(
        self,
        h: torch.Tensor,
        time_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the block.

        Args:
            h: Tokens of shape ``(B, T, model_size)``.
            time_embedding: Time context of shape ``(B, time_dim)``.
            mask: Optional boolean attention mask, ``(T, T)`` or ``(B, T, T)``.

        Returns:
            Updated tokens of shape ``(B, T, model_size)``.
        """
        h = h + self.attention(self.ln_attn(h), mask=mask)
        normed = self.ln_mlp(h)
        dense = self.mlp(normed)
        context = F.gelu(self.time_proj(time_embedding))
        dense = dense + context[:, None, :]
        return h + dense


class ScoreTransformer(nn.Module):
    """Transformer score network over per-variable tokens.

    The forward pass returns the *unscaled* network output of shape
    ``(B, num_nodes, 1)``; dividing by the SDE marginal standard deviation (done in
    :class:`synference.simformer.SimformerModel`) turns it into a score estimate.

    Defaults match the paper configuration (``score_transformer_small``).
    """

    def __init__(
        self,
        num_nodes: int,
        token_dim: int = 40,
        condition_token_dim: int = 10,
        condition_token_init_scale: float = 0.1,
        time_embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 6,
        attn_size: int = 10,
        widening_factor: int = 3,
        num_hidden_layers: int = 1,
    ):
        """Initialize the network.

        Args:
            num_nodes: Total number of variables (theta dims + x dims).
            token_dim: Tokenizer output width (id + value parts).
            condition_token_dim: Width of the learned condition token.
            condition_token_init_scale: Init stddev of the condition token.
            time_embedding_dim: Gaussian Fourier time-embedding width.
            num_heads: Attention heads per layer.
            num_layers: Number of transformer blocks.
            attn_size: Per-head key/query/value size.
            widening_factor: MLP hidden width multiplier.
            num_hidden_layers: Hidden layers per MLP block.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.config = {
            "num_nodes": num_nodes,
            "token_dim": token_dim,
            "condition_token_dim": condition_token_dim,
            "condition_token_init_scale": condition_token_init_scale,
            "time_embedding_dim": time_embedding_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "attn_size": attn_size,
            "widening_factor": widening_factor,
            "num_hidden_layers": num_hidden_layers,
        }

        self.tokenizer = ScalarTokenizer(token_dim, num_nodes)
        self.time_embedding = GaussianFourierEmbedding(time_embedding_dim)
        self.condition_token = nn.Parameter(
            torch.randn(1, 1, condition_token_dim) * condition_token_init_scale
        )

        model_size = token_dim + condition_token_dim
        init_scale = 2.0 / num_layers
        self.blocks = nn.ModuleList(
            TransformerBlock(
                model_size=model_size,
                num_heads=num_heads,
                key_size=attn_size,
                widening_factor=widening_factor,
                num_hidden_layers=num_hidden_layers,
                time_dim=time_embedding_dim,
                init_scale=init_scale,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(model_size)
        self.head = nn.Linear(model_size, 1)
        variance_scaling_init_(self.head.weight, 1.0, model_size)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        node_ids: torch.Tensor,
        condition_mask: torch.Tensor,
        edge_mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """Evaluate the network.

        Args:
            t: Diffusion times of shape ``(B,)`` (or scalar, broadcast to the batch).
            x: Variable values of shape ``(B, num_nodes, 1)``.
            node_ids: Integer node ids of shape ``(num_nodes,)``.
            condition_mask: Boolean mask, ``(num_nodes,)`` or ``(B, num_nodes)``;
                True marks observed (conditioned) variables.
            edge_mask: Optional boolean attention mask, ``(T, T)`` or ``(B, T, T)``.
                None means dense attention.

        Returns:
            Unscaled network output of shape ``(B, num_nodes, 1)``.
        """
        batch, num_nodes, _ = x.shape
        t = torch.atleast_1d(t).to(x)
        if t.shape[0] == 1 and batch > 1:
            t = t.expand(batch)

        tokens = self.tokenizer(node_ids, x)
        condition_mask = condition_mask.reshape(-1, num_nodes, 1).to(x)
        condition_token = condition_mask * self.condition_token
        condition_token = condition_token.expand(batch, num_nodes, -1)
        h = torch.cat([tokens, condition_token], dim=-1)

        time_embedding = self.time_embedding(t[..., None])

        for block in self.blocks:
            h = block(h, time_embedding, mask=edge_mask)
        h = self.final_norm(h)
        return self.head(h)
