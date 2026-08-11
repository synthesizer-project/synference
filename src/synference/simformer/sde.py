"""Diffusion SDEs for the PyTorch Simformer.

Ports the variance-exploding (VE) and variance-preserving (VP) SDEs from
``probjax.distributions.sde`` together with the loss weighting and score output
scaling from ``scoresbibm.methods.sde``.

Conventions (forward SDE ``dx = f(t, x) dt + g(t) dW``, ``t in [T_min, T_max]``):

- ``mean_scale(t)``: transition-kernel mean scale, ``E[x_t | x_0] = mean_scale(t) x_0``.
- ``transition_var(t)``: transition-kernel variance ``Var[x_t | x_0]``.
- ``weight(t)``: denoising score-matching loss weight.
- ``output_scale(t)``: the raw network output is divided by
  ``clamp(sqrt(transition_var(t)), scale_min)`` to produce a score estimate.
- ``marginal_mean_end`` / ``marginal_std_end``: per-node data-marginal statistics at
  ``T_max`` (using stored per-node data mean/variance), used to draw ``x_T``.
"""

import math
from typing import Dict

import numpy as np
import torch


class BaseSDE(torch.nn.Module):
    """Base class holding shared time constants and per-node data statistics."""

    def __init__(
        self,
        x0_mean: np.ndarray,
        x0_var: np.ndarray,
        T_min: float = 1e-5,
        T_max: float = 1.0,
        scale_min: float = 1e-3,
    ):
        """Initialize the SDE.

        Args:
            x0_mean: Per-node mean of the (z-scored) training data, shape ``(T,)``.
            x0_var: Per-node variance of the training data, shape ``(T,)``.
            T_min: Minimum diffusion time (never evaluate below this).
            T_max: Maximum diffusion time.
            scale_min: Lower clamp on the marginal stddev in ``output_scale``.
        """
        super().__init__()
        self.T_min = float(T_min)
        self.T_max = float(T_max)
        self.scale_min = float(scale_min)
        self.register_buffer("x0_mean", torch.as_tensor(x0_mean, dtype=torch.float32))
        self.register_buffer("x0_var", torch.as_tensor(x0_var, dtype=torch.float32))

    @property
    def config(self) -> Dict[str, float]:
        """Serializable constructor arguments (excluding data statistics)."""
        return {"T_min": self.T_min, "T_max": self.T_max, "scale_min": self.scale_min}

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward drift ``f(t, x)``."""
        raise NotImplementedError

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion ``g(t)`` (scalar function of time)."""
        raise NotImplementedError

    def mean_scale(self, t: torch.Tensor) -> torch.Tensor:
        """Transition-kernel mean scale ``E[x_t | x_0] / x_0``."""
        raise NotImplementedError

    def transition_var(self, t: torch.Tensor) -> torch.Tensor:
        """Transition-kernel variance ``Var[x_t | x_0]``."""
        raise NotImplementedError

    def transition_std(self, t: torch.Tensor) -> torch.Tensor:
        """Transition-kernel standard deviation."""
        return torch.sqrt(self.transition_var(t))

    def weight(self, t: torch.Tensor) -> torch.Tensor:
        """Loss weight for denoising score matching."""
        raise NotImplementedError

    def output_scale(self, t: torch.Tensor) -> torch.Tensor:
        """Factor the raw network output is divided by to obtain the score."""
        return torch.clamp(self.transition_std(t), min=self.scale_min)

    def marginal_mean_end(self) -> torch.Tensor:
        """Per-node data-marginal mean at ``T_max``, shape ``(T,)``."""
        t = torch.tensor(self.T_max, device=self.x0_mean.device)
        return self.mean_scale(t) * self.x0_mean

    def marginal_var_end(self) -> torch.Tensor:
        """Per-node data-marginal variance at ``T_max``, shape ``(T,)``."""
        raise NotImplementedError

    def marginal_std_end(self) -> torch.Tensor:
        """Per-node data-marginal standard deviation at ``T_max``."""
        return torch.sqrt(self.marginal_var_end())


class VESDE(BaseSDE):
    """Variance-exploding SDE (paper default: ``sigma_min=1e-4``, ``sigma_max=15``)."""

    name = "vesde"

    def __init__(
        self,
        x0_mean: np.ndarray,
        x0_var: np.ndarray,
        sigma_min: float = 1e-4,
        sigma_max: float = 15.0,
        T_min: float = 1e-5,
        T_max: float = 1.0,
        scale_min: float = 1e-3,
    ):
        """Initialize the VE SDE.

        Args:
            x0_mean: Per-node data mean, shape ``(T,)``.
            x0_var: Per-node data variance, shape ``(T,)``.
            sigma_min: Noise scale at ``t=0``.
            sigma_max: Noise scale at ``t=1``.
            T_min: Minimum diffusion time.
            T_max: Maximum diffusion time.
            scale_min: Lower clamp on the output-scale stddev.
        """
        super().__init__(x0_mean, x0_var, T_min=T_min, T_max=T_max, scale_min=scale_min)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self._log_ratio = math.log(self.sigma_max / self.sigma_min)

    @property
    def config(self) -> Dict[str, float]:
        """Serializable constructor arguments (excluding data statistics)."""
        return {
            "name": self.name,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            **super().config,
        }

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward drift is zero for the VE SDE."""
        return torch.zeros_like(x)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """``g(t) = sigma_min (sigma_max/sigma_min)^t sqrt(2 log(sigma_max/sigma_min))``."""
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return sigma_t * math.sqrt(2 * self._log_ratio)

    def mean_scale(self, t: torch.Tensor) -> torch.Tensor:
        """Mean scale is one for the VE SDE."""
        return torch.ones_like(torch.as_tensor(t, dtype=torch.float32))

    def transition_var(self, t: torch.Tensor) -> torch.Tensor:
        """``sigma_min^2 (sigma_max/sigma_min)^(2t)``."""
        return self.sigma_min**2 * (self.sigma_max / self.sigma_min) ** (2 * t)

    def weight(self, t: torch.Tensor) -> torch.Tensor:
        """Standard SDE weighting ``g(t)^2``."""
        return self.diffusion(t) ** 2

    def marginal_var_end(self) -> torch.Tensor:
        """Data variance plus transition variance at ``T_max``."""
        t = torch.tensor(self.T_max, device=self.x0_var.device)
        return self.x0_var + self.transition_var(t)


class VPSDE(BaseSDE):
    """Variance-preserving SDE (``beta_min=0.01``, ``beta_max=10``)."""

    name = "vpsde"

    def __init__(
        self,
        x0_mean: np.ndarray,
        x0_var: np.ndarray,
        beta_min: float = 0.01,
        beta_max: float = 10.0,
        T_min: float = 1e-5,
        T_max: float = 1.0,
        scale_min: float = 0.0,
    ):
        """Initialize the VP SDE.

        Args:
            x0_mean: Per-node data mean, shape ``(T,)``.
            x0_var: Per-node data variance, shape ``(T,)``.
            beta_min: Noise schedule value at ``t=0``.
            beta_max: Noise schedule value at ``t=1``.
            T_min: Minimum diffusion time.
            T_max: Maximum diffusion time.
            scale_min: Lower clamp on the output-scale stddev.
        """
        super().__init__(x0_mean, x0_var, T_min=T_min, T_max=T_max, scale_min=scale_min)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    @property
    def config(self) -> Dict[str, float]:
        """Serializable constructor arguments (excluding data statistics)."""
        return {
            "name": self.name,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            **super().config,
        }

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``f(t, x) = -0.5 beta(t) x``."""
        return -0.5 * self._beta(t) * x

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """``g(t) = sqrt(beta(t))``."""
        return torch.sqrt(self._beta(t))

    def mean_scale(self, t: torch.Tensor) -> torch.Tensor:
        """``exp(-0.25 t^2 (beta_max - beta_min) - 0.5 t beta_min)``."""
        t = torch.as_tensor(t, dtype=torch.float32)
        return torch.exp(-0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min)

    def transition_var(self, t: torch.Tensor) -> torch.Tensor:
        """``1 - mean_scale(t)^2``."""
        return 1.0 - self.mean_scale(t) ** 2

    def weight(self, t: torch.Tensor) -> torch.Tensor:
        """Likelihood weighting ``clamp(1 - mean_scale(t)^2, min=1e-4)``."""
        return torch.clamp(self.transition_var(t), min=1e-4)

    def marginal_var_end(self) -> torch.Tensor:
        """``1 + mean_scale(T)^2 (var_0 - 1)`` per node."""
        t = torch.tensor(self.T_max, device=self.x0_var.device)
        return 1.0 + self.mean_scale(t) ** 2 * (self.x0_var - 1.0)


SDE_CLASSES = {"vesde": VESDE, "vpsde": VPSDE}


def build_sde(name: str, x0_mean: np.ndarray, x0_var: np.ndarray, **kwargs) -> BaseSDE:
    """Build an SDE by name.

    Args:
        name: Either ``"vesde"`` or ``"vpsde"`` (case-insensitive).
        x0_mean: Per-node data mean, shape ``(T,)``.
        x0_var: Per-node data variance, shape ``(T,)``.
        **kwargs: Extra keyword arguments passed to the SDE constructor
            (e.g. ``sigma_min``, ``beta_max``, ``T_min``).

    Returns:
        The constructed SDE.

    Raises:
        ValueError: If ``name`` is not a known SDE.
    """
    key = name.lower()
    if key not in SDE_CLASSES:
        raise ValueError(f"Unknown SDE '{name}'. Choose from {sorted(SDE_CLASSES)}.")
    return SDE_CLASSES[key](x0_mean, x0_var, **kwargs)
