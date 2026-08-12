"""Condition-mask samplers and attention (edge) mask builders.

Two distinct mask concepts are used by the Simformer — do not confuse them:

- **Condition mask**: boolean vector over nodes ``[theta..., x...]`` where True marks
  an *observed* (conditioned) variable. Sampled per training example; supplied by the
  user at inference time.
- **Edge / base (attention) mask**: boolean ``(T, T)`` matrix where ``mask[i, j] = True``
  allows token ``i`` to attend to token ``j``. Encodes dependency structure; ``None``
  means dense attention (the paper's training default).
"""

from typing import Callable, Optional, Union

import numpy as np
import torch
from scipy.stats import beta as _beta_dist


def _fix_all_true_rows(condition_mask: torch.Tensor) -> torch.Tensor:
    """Force rows where every node is conditioned back to fully latent (all False)."""
    all_true = condition_mask.all(dim=-1, keepdim=True)
    return condition_mask & ~all_true


def joint_condition_mask(
    num_samples: int,
    theta_dim: int,
    x_dim: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """All-False masks: model the full joint distribution."""
    return torch.zeros(num_samples, theta_dim + x_dim, dtype=torch.bool)


def posterior_condition_mask(
    num_samples: int,
    theta_dim: int,
    x_dim: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Condition on all data nodes: standard posterior masks."""
    row = torch.tensor([False] * theta_dim + [True] * x_dim)
    return row.expand(num_samples, -1).clone()


def likelihood_condition_mask(
    num_samples: int,
    theta_dim: int,
    x_dim: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Condition on all parameter nodes: likelihood masks."""
    row = torch.tensor([True] * theta_dim + [False] * x_dim)
    return row.expand(num_samples, -1).clone()


def random_condition_mask(
    num_samples: int,
    theta_dim: int,
    x_dim: int,
    generator: Optional[torch.Generator] = None,
    alpha: float = 1.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Bernoulli masks with per-batch probability drawn from ``Beta(alpha, beta)``."""
    # torch.distributions.Beta.sample() cannot accept a generator, so the
    # Beta draw is done via inverse-CDF from a generator-seeded uniform
    # instead — otherwise a seeded `generator` wouldn't make this reproducible.
    u = torch.rand((), generator=generator).item()
    prob = float(_beta_dist.ppf(u, alpha, beta))
    mask = torch.rand(num_samples, theta_dim + x_dim, generator=generator) < prob
    return _fix_all_true_rows(mask)


def structured_random_condition_mask(
    num_samples: int,
    theta_dim: int,
    x_dim: int,
    generator: Optional[torch.Generator] = None,
    p_joint: float = 0.2,
    p_posterior: float = 0.2,
    p_likelihood: float = 0.2,
    p_rnd1: float = 0.2,
    p_rnd2: float = 0.2,
    rnd1_prob: float = 0.3,
    rnd2_prob: float = 0.7,
) -> torch.Tensor:
    """Sample per-row masks from {joint, posterior, likelihood, two random masks}.

    Matches ``scoresbibm.utils.condition_masks.sample_strutured_conditional_mask``:
    the two Bernoulli candidate masks are drawn once per call and shared across the
    rows that select them; rows that end up all-True are forced back to all-False.

    Args:
        num_samples: Number of mask rows to draw.
        theta_dim: Number of parameter nodes.
        x_dim: Number of data nodes.
        generator: Optional torch random generator.
        p_joint: Probability of an all-latent row.
        p_posterior: Probability of a posterior row.
        p_likelihood: Probability of a likelihood row.
        p_rnd1: Probability of the first random candidate row.
        p_rnd2: Probability of the second random candidate row.
        rnd1_prob: Bernoulli probability of the first random candidate.
        rnd2_prob: Bernoulli probability of the second random candidate.

    Returns:
        Boolean condition masks of shape ``(num_samples, theta_dim + x_dim)``.
    """
    total = theta_dim + x_dim
    candidates = torch.stack(
        [
            torch.zeros(total, dtype=torch.bool),
            torch.tensor([False] * theta_dim + [True] * x_dim),
            torch.tensor([True] * theta_dim + [False] * x_dim),
            torch.rand(total, generator=generator) < rnd1_prob,
            torch.rand(total, generator=generator) < rnd2_prob,
        ]
    )
    probs = torch.tensor([p_joint, p_posterior, p_likelihood, p_rnd1, p_rnd2])
    choice = torch.multinomial(probs, num_samples, replacement=True, generator=generator)
    return _fix_all_true_rows(candidates[choice])


CONDITION_MASK_FNS = {
    "joint": joint_condition_mask,
    "posterior": posterior_condition_mask,
    "likelihood": likelihood_condition_mask,
    "random": random_condition_mask,
    "structured_random": structured_random_condition_mask,
}


def get_condition_mask_fn(name: str, **kwargs) -> Callable:
    """Look up a condition-mask sampler by name.

    Args:
        name: One of ``joint``, ``posterior``, ``likelihood``, ``random``,
            ``structured_random``.
        **kwargs: Fixed keyword arguments bound to the sampler (e.g. ``p_joint``).

    Returns:
        A callable ``fn(num_samples, theta_dim, x_dim, generator=None)``.

    Raises:
        ValueError: If ``name`` is not a known sampler.
    """
    key = name.lower()
    if key not in CONDITION_MASK_FNS:
        raise ValueError(
            f"Unknown condition mask fn '{name}'. Choose from {sorted(CONDITION_MASK_FNS)}."
        )
    if kwargs:
        from functools import partial

        return partial(CONDITION_MASK_FNS[key], **kwargs)
    return CONDITION_MASK_FNS[key]


def build_base_mask(
    kind: Union[str, np.ndarray, torch.Tensor, None],
    theta_dim: int,
    x_dim: int,
) -> Optional[torch.Tensor]:
    """Build the base (attention) mask over nodes ordered ``[theta..., x...]``.

    Args:
        kind: ``"full"``/``None`` for dense attention (returns None); ``"directed"``
            (alias ``"causal"``) for the structured mask where parameters attend only
            to themselves, data attend to all parameters and causally within data, and
            parameters do not attend to data; or an explicit boolean ``(T, T)`` array.
        theta_dim: Number of parameter nodes.
        x_dim: Number of data nodes.

    Returns:
        A boolean ``(T, T)`` tensor, or None for dense attention.

    Raises:
        ValueError: If ``kind`` is an unknown string or a wrongly shaped array.
    """
    total = theta_dim + x_dim
    if kind is None or (isinstance(kind, str) and kind.lower() == "full"):
        return None
    if isinstance(kind, str):
        if kind.lower() in ("directed", "causal"):
            mask = torch.zeros(total, total, dtype=torch.bool)
            mask[:theta_dim, :theta_dim] = torch.eye(theta_dim, dtype=torch.bool)
            mask[theta_dim:, :theta_dim] = True
            mask[theta_dim:, theta_dim:] = torch.tril(torch.ones(x_dim, x_dim, dtype=torch.bool))
            return mask
        raise ValueError(
            f"Unknown base mask kind '{kind}'. Use 'full', 'directed'/'causal', or an array."
        )
    mask = torch.as_tensor(np.asarray(kind), dtype=torch.bool)
    if mask.shape != (total, total):
        raise ValueError(f"Custom base mask must have shape ({total}, {total}), got {mask.shape}.")
    return mask
