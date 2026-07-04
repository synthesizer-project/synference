"""Training loop for the PyTorch Simformer.

Ports ``scoresbibm.methods.score_transformer.train_transformer_model``: denoising
score matching over the joint ``[theta, x]`` with per-example condition masks,
adaptive gradient clipping + Adam with a linear-decay schedule, a Monte-Carlo
validation split with early stopping, and optional per-node z-scoring of the data.
"""

import copy
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from .. import logger
from .masks import build_base_mask, get_condition_mask_fn
from .model import SimformerModel
from .nn import ScoreTransformer
from .sde import BaseSDE, build_sde

DEFAULT_MODEL_CONFIG: Dict = {
    "token_dim": 40,
    "condition_token_dim": 10,
    "condition_token_init_scale": 0.1,
    "time_embedding_dim": 128,
    "num_heads": 4,
    "num_layers": 6,
    "attn_size": 10,
    "widening_factor": 3,
    "num_hidden_layers": 1,
}

DEFAULT_SDE_CONFIG: Dict = {
    "name": "vesde",
    "sigma_min": 1e-4,
    "sigma_max": 15.0,
    "T_min": 1e-5,
    "T_max": 1.0,
    "scale_min": 1e-3,
}

DEFAULT_TRAIN_CONFIG: Dict = {
    "learning_rate": 1e-3,
    "min_learning_rate": 1e-6,
    "z_score_data": True,
    "total_number_steps_scaling": 3,
    "max_number_steps": 100_000,
    "min_number_steps": 5_000,
    "training_batch_size": 1000,
    "val_every": 50,
    "clip_max_norm": 10.0,
    "condition_mask_fn": "structured_random",
    "validation_fraction": 0.05,
    "val_repeat": 5,
    "val_error_ratio": 1.1,
    "stop_early_count": 5,
    "rebalance_loss": False,
    "print_every_fraction": 0.1,
}


def merge_config(defaults: Dict, overrides: Optional[Dict], config_name: str) -> Dict:
    """Merge override values into a default config, rejecting unknown keys.

    Args:
        defaults: The default configuration dictionary.
        overrides: User overrides (or None).
        config_name: Name used in the error message.

    Returns:
        A new merged dictionary.

    Raises:
        ValueError: If an override key is not present in the defaults.
    """
    config = dict(defaults)
    if overrides:
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise ValueError(
                f"Unknown {config_name} keys: {sorted(unknown)}. Valid keys: {sorted(defaults)}."
            )
        config.update(overrides)
    return config


def mean_std_per_node(data: np.ndarray, node_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-node-id mean and clipped standard deviation of the training data.

    Args:
        data: Training data of shape ``(N, T)``.
        node_ids: Integer node ids of shape ``(T,)`` (repeated ids share statistics).

    Returns:
        Tuple of per-node mean and stddev arrays, each of shape ``(T,)``
        (stddev clipped below at 1e-2).
    """
    node_ids = np.asarray(node_ids).reshape(-1)
    mean_per_id = {}
    std_per_id = {}
    for node in np.unique(node_ids):
        values = data[:, node_ids == node]
        mean_per_id[node] = float(np.mean(values))
        std_per_id[node] = max(float(np.std(values)), 1e-2)
    mean = np.array([mean_per_id[i] for i in node_ids], dtype=np.float32)
    std = np.array([std_per_id[i] for i in node_ids], dtype=np.float32)
    return mean, std


def denoising_score_matching_loss(
    net: ScoreTransformer,
    sde: BaseSDE,
    data: torch.Tensor,
    node_ids: torch.Tensor,
    condition_mask: torch.Tensor,
    edge_mask: Optional[torch.Tensor] = None,
    rebalance_loss: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Masked denoising score-matching loss.

    Times are drawn uniformly in ``[T_min, T_max]``; the state is noised with the
    transition kernel except at conditioned nodes, which keep their clean values and
    contribute zero loss. The target score is ``-eps / std(t)`` and the summed
    per-node error is weighted by ``sde.weight(t)``.

    Args:
        net: The score network.
        sde: The diffusion SDE.
        data: Clean training batch of shape ``(B, T)`` (z-scored space).
        node_ids: Integer node ids of shape ``(T,)``.
        condition_mask: Boolean masks of shape ``(B, T)``.
        edge_mask: Optional attention mask (``(T, T)`` or ``(B, T, T)``).
        rebalance_loss: If True, divide each example's loss by its number of
            unconditioned nodes.
        generator: Optional torch random generator.

    Returns:
        Scalar loss tensor.
    """
    batch = data.shape[0]
    device = data.device
    t = torch.rand(batch, device=device, generator=generator) * (sde.T_max - sde.T_min) + sde.T_min
    eps = torch.randn(data.shape, device=device, generator=generator)
    mean_t = sde.mean_scale(t)[:, None] * data
    std_t = sde.transition_std(t)[:, None].expand_as(data)
    x_t = mean_t + std_t * eps
    x_t = torch.where(condition_mask, data, x_t)

    raw = net(t, x_t[..., None], node_ids, condition_mask, edge_mask=edge_mask)[..., 0]
    score_pred = raw / sde.output_scale(t)[:, None]
    score_target = -eps / std_t

    sq_err = (score_pred - score_target) ** 2
    sq_err = torch.where(condition_mask, torch.zeros_like(sq_err), sq_err)
    per_example = sde.weight(t) * sq_err.sum(dim=-1)
    if rebalance_loss:
        num_latent = (~condition_mask).sum(dim=-1)
        per_example = torch.where(
            num_latent > 0, per_example / num_latent.clamp(min=1), torch.zeros_like(per_example)
        )
    return per_example.mean()


def adaptive_grad_clip_(
    parameters,
    clipping: float = 10.0,
    eps: float = 1e-3,
) -> None:
    """In-place adaptive gradient clipping (port of ``optax.adaptive_grad_clip``).

    Each parameter's gradient is rescaled so its unit-wise norm does not exceed
    ``clipping`` times the unit-wise parameter norm (floored at ``eps``).

    Args:
        parameters: Iterable of parameters with gradients.
        clipping: Maximum gradient-to-parameter norm ratio.
        eps: Floor on the parameter norm.
    """
    for param in parameters:
        if param.grad is None:
            continue
        if param.ndim <= 1:
            p_norm = param.norm().clamp(min=eps)
            g_norm = param.grad.norm().clamp(min=1e-6)
            scale = torch.clamp(clipping * p_norm / g_norm, max=1.0)
            param.grad.mul_(scale)
        else:
            dims = tuple(range(1, param.ndim))
            p_norm = param.norm(dim=dims, keepdim=True).clamp(min=eps)
            g_norm = param.grad.norm(dim=dims, keepdim=True).clamp(min=1e-6)
            scale = torch.clamp(clipping * p_norm / g_norm, max=1.0)
            param.grad.mul_(scale)


def train_simformer(
    theta: np.ndarray,
    x: np.ndarray,
    model_config: Optional[Dict] = None,
    sde_config: Optional[Dict] = None,
    train_config: Optional[Dict] = None,
    base_mask=None,
    meta: Optional[Dict] = None,
    device: str = "cpu",
    seed: Optional[int] = None,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None,
) -> Tuple[SimformerModel, Dict]:
    """Train a Simformer on parameter/observation pairs.

    Args:
        theta: Parameters of shape ``(N, theta_dim)``.
        x: Observations of shape ``(N, x_dim)``.
        model_config: Overrides for :data:`DEFAULT_MODEL_CONFIG`.
        sde_config: Overrides for :data:`DEFAULT_SDE_CONFIG` (must include ``name``
            only to switch SDE type; unknown keys are rejected per type).
        train_config: Overrides for :data:`DEFAULT_TRAIN_CONFIG`.
        base_mask: Base attention mask — ``"full"``/None for dense, ``"directed"``,
            or an explicit boolean ``(T, T)`` array (see
            :func:`synference.simformer.masks.build_base_mask`).
        meta: Metadata stored on the returned model (parameter/feature names, prior
            ranges, etc.).
        device: Torch device for training.
        seed: Random seed (None for nondeterministic).
        verbose: Log training progress.
        progress_callback: Optional callable ``(step, train_loss, val_loss)``.

    Returns:
        Tuple of the trained :class:`SimformerModel` (on CPU, eval mode) and a stats
        dict (loss traces, steps run, early-stopping info, wall time).
    """
    model_config = merge_config(DEFAULT_MODEL_CONFIG, model_config, "model_config")
    train_config = merge_config(DEFAULT_TRAIN_CONFIG, train_config, "train_config")
    sde_config_full = dict(DEFAULT_SDE_CONFIG)
    if sde_config:
        sde_name = str(sde_config.get("name", sde_config_full["name"])).lower()
        if sde_name != sde_config_full["name"]:
            # Switching SDE type: start from that type's own defaults.
            sde_config_full = {"name": sde_name}
        sde_config_full.update(sde_config)
        sde_config_full["name"] = sde_name

    theta = np.asarray(theta, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if theta.ndim != 2 or x.ndim != 2 or theta.shape[0] != x.shape[0]:
        raise ValueError(
            f"theta and x must be 2D with matching first dimension, got {theta.shape}, {x.shape}."
        )
    theta_dim, x_dim = theta.shape[1], x.shape[1]
    num_nodes = theta_dim + x_dim
    data = np.hstack([theta, x])
    node_ids = np.arange(num_nodes)

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
        torch.manual_seed(int(seed))

    # Per-node z-scoring.
    if train_config["z_score_data"]:
        z_mean, z_std = mean_std_per_node(data, node_ids)
        data = (data - z_mean) / z_std
    else:
        z_mean = z_std = None

    x0_mean = data.mean(axis=0)
    x0_var = data.var(axis=0)
    sde_kwargs = {k: v for k, v in sde_config_full.items() if k != "name"}
    sde = build_sde(sde_config_full["name"], x0_mean, x0_var, **sde_kwargs)

    net = ScoreTransformer(num_nodes=num_nodes, **model_config)
    device = torch.device(device)
    net.to(device)
    sde.to(device)

    # Validation split (first fraction of rows, as in the original implementation).
    n_val = max(int(train_config["validation_fraction"] * data.shape[0]), 0)
    data_t = torch.as_tensor(data, device=device)
    data_val = data_t[:n_val].repeat(train_config["val_repeat"], 1) if n_val > 0 else None
    data_train = data_t[n_val:]
    node_ids_t = torch.as_tensor(node_ids, device=device)

    total_steps = int(
        np.clip(
            data.shape[0] * train_config["total_number_steps_scaling"],
            train_config["min_number_steps"],
            train_config["max_number_steps"],
        )
    )
    batch_size = int(train_config["training_batch_size"])
    val_every = max(total_steps // int(train_config["val_every"]), 1)
    print_every = max(int(total_steps * train_config["print_every_fraction"]), 1)

    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=train_config["learning_rate"]
    )
    half = total_steps // 2
    lr_ratio = train_config["min_learning_rate"] / train_config["learning_rate"]

    def lr_lambda(step: int) -> float:
        if step < half or half == 0:
            return 1.0
        frac = min((step - half) / max(total_steps - half, 1), 1.0)
        return 1.0 + frac * (lr_ratio - 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    condition_mask_fn = get_condition_mask_fn(train_config["condition_mask_fn"])
    edge_mask = build_base_mask(base_mask, theta_dim, x_dim)
    edge_mask_t = edge_mask.to(device) if edge_mask is not None else None

    def compute_loss(batch_data: torch.Tensor) -> torch.Tensor:
        condition_mask = condition_mask_fn(
            batch_data.shape[0], theta_dim, x_dim, generator=generator
        ).to(device)
        return denoising_score_matching_loss(
            net,
            sde,
            batch_data,
            node_ids_t,
            condition_mask,
            edge_mask=edge_mask_t,
            rebalance_loss=train_config["rebalance_loss"],
        )

    stats: Dict = {
        "train_loss": [],
        "val_loss": [],
        "val_steps": [],
        "total_steps_planned": total_steps,
        "early_stopped": False,
    }
    train_loss_ema = None
    min_val_loss = np.inf
    best_state = None
    early_stopping_counter = 0
    start_time = time.time()

    net.train()
    for step in range(total_steps):
        idx = torch.randint(
            0, data_train.shape[0], (batch_size,), device="cpu", generator=generator
        )
        batch = data_train[idx.to(device)]

        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(batch)
        loss.backward()
        adaptive_grad_clip_(
            [p for p in net.parameters() if p.requires_grad], train_config["clip_max_norm"]
        )
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.detach())
        train_loss_ema = (
            loss_value if train_loss_ema is None else 0.9 * train_loss_ema + 0.1 * loss_value
        )
        stats["train_loss"].append(loss_value)

        if data_val is not None and step > 50 and (step % val_every) == 0:
            net.eval()
            with torch.no_grad():
                val_loss = float(compute_loss(data_val).detach())
            net.train()
            stats["val_loss"].append(val_loss)
            stats["val_steps"].append(step)

            if val_loss / train_loss_ema > train_config["val_error_ratio"]:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_state = copy.deepcopy({k: v.cpu() for k, v in net.state_dict().items()})
            if early_stopping_counter > train_config["stop_early_count"]:
                stats["early_stopped"] = True
                if verbose:
                    logger.info(f"Early stopping at step {step} (val loss {val_loss:.4f}).")
                break

        if verbose and (step % print_every) == 0:
            message = f"Step {step}/{total_steps}: train loss {train_loss_ema:.4f}"
            if stats["val_loss"]:
                message += f", val loss {stats['val_loss'][-1]:.4f}"
            logger.info(message)
        if progress_callback is not None:
            progress_callback(
                step, train_loss_ema, stats["val_loss"][-1] if stats["val_loss"] else None
            )

    if stats["early_stopped"] and best_state is not None:
        net.load_state_dict(best_state)

    stats["steps_run"] = len(stats["train_loss"])
    stats["best_val_loss"] = None if np.isinf(min_val_loss) else float(min_val_loss)
    stats["final_train_loss_ema"] = train_loss_ema
    stats["training_time_s"] = time.time() - start_time

    net.eval()
    net.to("cpu")
    sde.to("cpu")
    model = SimformerModel(
        net=net,
        sde=sde,
        theta_dim=theta_dim,
        x_dim=x_dim,
        node_ids=node_ids,
        base_mask=None if edge_mask is None else edge_mask.cpu().numpy(),
        z_score_mean=z_mean,
        z_score_std=z_std,
        meta=meta or {},
        sampling_defaults={"num_steps": 500, "method": "sde"},
    )
    return model, stats
