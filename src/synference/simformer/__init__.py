"""Native PyTorch implementation of the Simformer (Gloeckler et al. 2024).

A transformer + score-diffusion all-in-one SBI model: a single network trained on the
joint ``[theta, x]`` with per-example condition masks provides posterior, likelihood,
and arbitrary conditionals, with optional interval-constrained sampling via guidance.
"""

from .masks import (
    build_base_mask,
    get_condition_mask_fn,
    joint_condition_mask,
    likelihood_condition_mask,
    posterior_condition_mask,
    structured_random_condition_mask,
)
from .model import SimformerModel, UniformBoxPrior, posterior_mask
from .nn import GaussianFourierEmbedding, ScalarTokenizer, ScoreTransformer, TransformerBlock
from .sampling import (
    euler_maruyama_reverse,
    guided_euler_maruyama,
    heun_probability_flow,
    interval_constraint_score,
    probability_flow_log_prob,
)
from .sde import SDE_CLASSES, VESDE, VPSDE, BaseSDE, build_sde
from .train import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_SDE_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    adaptive_grad_clip_,
    denoising_score_matching_loss,
    mean_std_per_node,
    merge_config,
    train_simformer,
)

__all__ = [
    "SimformerModel",
    "UniformBoxPrior",
    "posterior_mask",
    "ScoreTransformer",
    "TransformerBlock",
    "ScalarTokenizer",
    "GaussianFourierEmbedding",
    "BaseSDE",
    "VESDE",
    "VPSDE",
    "SDE_CLASSES",
    "build_sde",
    "build_base_mask",
    "get_condition_mask_fn",
    "joint_condition_mask",
    "posterior_condition_mask",
    "likelihood_condition_mask",
    "structured_random_condition_mask",
    "train_simformer",
    "denoising_score_matching_loss",
    "adaptive_grad_clip_",
    "mean_std_per_node",
    "merge_config",
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_SDE_CONFIG",
    "DEFAULT_TRAIN_CONFIG",
    "euler_maruyama_reverse",
    "heun_probability_flow",
    "guided_euler_maruyama",
    "interval_constraint_score",
    "probability_flow_log_prob",
]
