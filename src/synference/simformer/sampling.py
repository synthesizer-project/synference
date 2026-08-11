"""Fixed-step SDE/ODE integrators and guidance for the PyTorch Simformer.

All integrators operate in the model's (z-scored) space on batched states of shape
``(B, T)`` with a boolean condition mask of shape ``(T,)`` (True = observed). Observed
entries are clamped: both drift and diffusion are multiplied by the latent indicator,
matching ``scoresbibm.methods.models.AllConditionalScoreModel``.

``score_fn(t, x)`` takes a scalar time tensor and the ``(B, T)`` state and returns the
``(B, T)`` score estimate.
"""

from typing import Callable, Optional, Tuple

import torch

from .sde import BaseSDE


def euler_maruyama_reverse(
    score_fn: Callable,
    sde: BaseSDE,
    x_T: torch.Tensor,
    condition_mask: torch.Tensor,
    num_steps: int = 500,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Integrate the reverse SDE from ``T_max`` to ``T_min`` with Euler-Maruyama.

    Args:
        score_fn: Score function ``(t, x) -> score`` on ``(B, T)`` states.
        sde: The diffusion SDE.
        x_T: Initial state at ``T_max``, shape ``(B, T)``, observed entries already
            set to their conditioning values.
        condition_mask: Boolean mask of shape ``(T,)``; True marks observed nodes.
        num_steps: Number of integration steps.
        generator: Optional torch random generator (on the same device as ``x_T``).

    Returns:
        The state at ``T_min``, shape ``(B, T)`` (observed entries unchanged).
    """
    latent = (~condition_mask).to(x_T)
    ts = torch.linspace(0.0, sde.T_max - sde.T_min, num_steps, device=x_T.device)
    x = x_T
    for n in range(num_steps - 1):
        dt = ts[n + 1] - ts[n]
        t = sde.T_max - ts[n]
        score = score_fn(t, x)
        g = sde.diffusion(t)
        drift = -(sde.drift(t, x) - g**2 * score) * latent
        noise = torch.randn(x.shape, device=x.device, generator=generator)
        x = x + drift * dt + g * torch.sqrt(dt) * noise * latent
    return x


def heun_probability_flow(
    score_fn: Callable,
    sde: BaseSDE,
    x_T: torch.Tensor,
    condition_mask: torch.Tensor,
    num_steps: int = 500,
) -> torch.Tensor:
    """Integrate the reverse probability-flow ODE with Heun's method.

    Args:
        score_fn: Score function ``(t, x) -> score`` on ``(B, T)`` states.
        sde: The diffusion SDE.
        x_T: Initial state at ``T_max``, shape ``(B, T)``.
        condition_mask: Boolean mask of shape ``(T,)``; True marks observed nodes.
        num_steps: Number of integration steps.

    Returns:
        The state at ``T_min``, shape ``(B, T)``.
    """
    latent = (~condition_mask).to(x_T)

    def drift_backward(s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = sde.T_max - s
        score = score_fn(t, x)
        dx = sde.drift(t, x) - 0.5 * sde.diffusion(t) ** 2 * score
        return -dx * latent

    ts = torch.linspace(0.0, sde.T_max - sde.T_min, num_steps, device=x_T.device)
    x = x_T
    for n in range(num_steps - 1):
        dt = ts[n + 1] - ts[n]
        k1 = drift_backward(ts[n], x)
        k2 = drift_backward(ts[n + 1], x + dt * k1)
        x = x + 0.5 * dt * (k1 + k2)
    return x


def interval_constraint_score(
    x0_hat: torch.Tensor,
    scale: torch.Tensor,
    constraint_mask: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gradient of the smoothed box-constraint log-probability at ``x0_hat``.

    Implements the analytic gradient of ``scoresbibm.methods.guidance.log_step_fn``:
    ``sum log sigmoid(scale (x - a)) + log sigmoid(scale (b - x))`` over constrained
    nodes, differentiated with respect to ``x0_hat``.

    Args:
        x0_hat: Tweedie estimate of the clean state, shape ``(B, T)``.
        scale: Guidance scale ``s(t)`` (scalar tensor).
        constraint_mask: Boolean mask of shape ``(T,)`` marking constrained nodes.
        a: Lower bounds, shape ``(T,)``; non-finite entries mean unbounded below.
        b: Upper bounds, shape ``(T,)``; non-finite entries mean unbounded above.

    Returns:
        Constraint score of shape ``(B, T)`` (zero outside ``constraint_mask``).
    """
    grad = torch.zeros_like(x0_hat)
    mask = constraint_mask.to(x0_hat)
    if a is not None:
        finite_a = torch.isfinite(a)
        a_safe = torch.where(finite_a, a, torch.zeros_like(a))
        # d/dx log sigmoid(s (x - a)) = s sigmoid(-s (x - a))
        term = scale * torch.sigmoid(-scale * (x0_hat - a_safe))
        grad = grad + term * finite_a.to(x0_hat) * mask
    if b is not None:
        finite_b = torch.isfinite(b)
        b_safe = torch.where(finite_b, b, torch.zeros_like(b))
        # d/dx log sigmoid(s (b - x)) = -s sigmoid(-s (b - x))
        term = -scale * torch.sigmoid(-scale * (b_safe - x0_hat))
        grad = grad + term * finite_b.to(x0_hat) * mask
    return grad


def guided_euler_maruyama(
    score_fn: Callable,
    sde: BaseSDE,
    x_T: torch.Tensor,
    condition_mask: torch.Tensor,
    constraint_score_fn: Callable,
    num_steps: int = 500,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Reverse Euler-Maruyama with generalized (Tweedie) guidance.

    At each step the clean state is estimated via Tweedie's formula,
    ``x0_hat = (x + std(t)^2 score) / mean_scale(t)``, and
    ``constraint_score_fn(x0_hat, t)`` is added to the model score, following
    ``scoresbibm.methods.guidance.generalized_guidance``.

    Args:
        score_fn: Score function ``(t, x) -> score`` on ``(B, T)`` states.
        sde: The diffusion SDE.
        x_T: Initial state at ``T_max``, shape ``(B, T)``.
        condition_mask: Boolean mask of shape ``(T,)``; True marks observed nodes
            (hard-conditioned; constrained nodes must not be in this mask).
        constraint_score_fn: Callable ``(x0_hat, t) -> (B, T)`` guidance score.
        num_steps: Number of integration steps.
        generator: Optional torch random generator.

    Returns:
        The state at ``T_min``, shape ``(B, T)``.
    """
    latent = (~condition_mask).to(x_T)
    ts = torch.linspace(sde.T_min, sde.T_max, num_steps, device=x_T.device)
    x = x_T
    t1 = ts[-1]
    for n in range(num_steps - 2, -1, -1):
        t0 = ts[n]
        dt = t0 - t1  # negative
        score = score_fn(t1, x)
        x0_hat = (x + sde.transition_std(t1) ** 2 * score) / sde.mean_scale(t1)
        score = score + constraint_score_fn(x0_hat, t1)
        g = sde.diffusion(t1)
        drift = (sde.drift(t1, x) - g**2 * score) * latent
        noise = torch.randn(x.shape, device=x.device, generator=generator)
        x = x + drift * dt + g * torch.sqrt(torch.abs(dt)) * noise * latent
        t1 = t0
    return x


def _divergence_exact(
    vector_field: Callable,
    t: torch.Tensor,
    x: torch.Tensor,
    latent_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vector field and its exact divergence over the latent dimensions.

    Args:
        vector_field: Callable ``(t, x) -> (B, T)``.
        t: Scalar time tensor.
        x: State of shape ``(B, T)``.
        latent_idx: Long tensor of latent dimension indices.

    Returns:
        Tuple of the vector field value ``(B, T)`` and divergence ``(B,)``.
    """
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        v = vector_field(t, x)
        div = torch.zeros(x.shape[0], device=x.device)
        for dim in latent_idx.tolist():
            grad = torch.autograd.grad(v[:, dim].sum(), x, create_graph=False, retain_graph=True)[0]
            div = div + grad[:, dim]
    return v.detach(), div.detach()


def _divergence_hutchinson(
    vector_field: Callable,
    t: torch.Tensor,
    x: torch.Tensor,
    latent_idx: torch.Tensor,
    num_probes: int = 8,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vector field and Hutchinson (Rademacher) divergence estimate.

    Args:
        vector_field: Callable ``(t, x) -> (B, T)``.
        t: Scalar time tensor.
        x: State of shape ``(B, T)``.
        latent_idx: Long tensor of latent dimension indices.
        num_probes: Number of Rademacher probe vectors.
        generator: Optional torch random generator.

    Returns:
        Tuple of the vector field value ``(B, T)`` and divergence estimate ``(B,)``.
    """
    latent = torch.zeros(x.shape[-1], device=x.device)
    latent[latent_idx] = 1.0
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        v = vector_field(t, x)
        div = torch.zeros(x.shape[0], device=x.device)
        for _ in range(num_probes):
            probe = torch.randint(
                0, 2, x.shape, device=x.device, generator=generator, dtype=x.dtype
            )
            probe = (2 * probe - 1) * latent
            vjp = torch.autograd.grad((v * probe).sum(), x, retain_graph=True)[0]
            div = div + (vjp * probe).sum(dim=-1)
        div = div / num_probes
    return v.detach(), div.detach()


def probability_flow_log_prob(
    score_fn: Callable,
    sde: BaseSDE,
    x0: torch.Tensor,
    condition_mask: torch.Tensor,
    num_steps: int = 250,
    divergence: str = "exact",
    hutchinson_probes: int = 8,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Log probability of the latent entries of ``x0`` via the probability-flow ODE.

    Integrates the forward (noising) probability-flow ODE from ``T_min`` to ``T_max``
    with Euler steps, accumulating the divergence (instantaneous change of variables),
    and evaluates the terminal Gaussian at ``T_max``. All in z-scored model space.

    Args:
        score_fn: Score function ``(t, x) -> score`` on ``(B, T)`` states.
        sde: The diffusion SDE.
        x0: Full state at ``T_min``, shape ``(B, T)``, observed entries set to the
            conditioning values, latent entries set to the values being evaluated.
        condition_mask: Boolean mask of shape ``(T,)``; True marks observed nodes.
        num_steps: Number of integration steps.
        divergence: ``"exact"`` (autograd trace over latent dims) or ``"hutchinson"``.
        hutchinson_probes: Number of probes for the Hutchinson estimator.
        generator: Optional torch random generator (Hutchinson only).

    Returns:
        Log probabilities of shape ``(B,)``.

    Raises:
        ValueError: If ``divergence`` is not a known estimator.
    """
    latent_mask = ~condition_mask
    latent_idx = torch.nonzero(latent_mask, as_tuple=False).flatten()
    latent = latent_mask.to(x0)

    def vector_field(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        score = score_fn(t, x)
        return (sde.drift(t, x) - 0.5 * sde.diffusion(t) ** 2 * score) * latent

    if divergence == "exact":
        div_fn = lambda t, x: _divergence_exact(vector_field, t, x, latent_idx)  # noqa: E731
    elif divergence == "hutchinson":
        div_fn = lambda t, x: _divergence_hutchinson(  # noqa: E731
            vector_field, t, x, latent_idx, num_probes=hutchinson_probes, generator=generator
        )
    else:
        raise ValueError(f"Unknown divergence estimator '{divergence}'.")

    ts = torch.linspace(sde.T_min, sde.T_max, num_steps, device=x0.device)
    x = x0
    log_det = torch.zeros(x0.shape[0], device=x0.device)
    for n in range(num_steps - 1):
        dt = ts[n + 1] - ts[n]
        v, div = div_fn(ts[n], x)
        x = x + v * dt
        log_det = log_det + div * dt

    mean_end = sde.marginal_mean_end()[latent_mask]
    std_end = sde.marginal_std_end()[latent_mask]
    y = x[:, latent_mask]
    log_q = (
        -0.5 * ((y - mean_end) / std_end) ** 2
        - torch.log(std_end)
        - 0.5 * torch.log(torch.tensor(2 * torch.pi, device=x0.device))
    ).sum(dim=-1)
    return log_q + log_det


__all__ = [
    "euler_maruyama_reverse",
    "heun_probability_flow",
    "guided_euler_maruyama",
    "interval_constraint_score",
    "probability_flow_log_prob",
]
