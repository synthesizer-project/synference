"""Trained Simformer model wrapper and box prior.

:class:`SimformerModel` bundles the trained score network, the SDE, per-node z-score
statistics, and metadata into a single object with a numpy-friendly inference API
(sampling with arbitrary condition masks, interval-constrained sampling via guidance,
and probability-flow log probabilities). It serializes to a plain ``torch.save`` dict
and is reloadable in a fresh process.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .nn import ScoreTransformer
from .sampling import (
    euler_maruyama_reverse,
    guided_euler_maruyama,
    heun_probability_flow,
    interval_constraint_score,
    probability_flow_log_prob,
)
from .sde import BaseSDE, build_sde

SAVE_FORMAT_VERSION = 1


class UniformBoxPrior:
    """Independent uniform prior over a named box, matching the old ``GalaxyPrior`` API.

    Attributes:
        prior_ranges: Mapping of parameter name to ``(low, high)``.
        param_order: Parameter names in sampling order.
        theta_dim: Number of parameters.
    """

    def __init__(self, prior_ranges: Dict[str, Tuple[float, float]], param_order: Sequence[str]):
        """Initialize the prior.

        Args:
            prior_ranges: Mapping of parameter name to ``(low, high)`` bounds.
            param_order: Parameter names defining the dimension order.
        """
        self.prior_ranges = {name: tuple(prior_ranges[name]) for name in param_order}
        self.param_order = list(param_order)
        self.theta_dim = len(self.param_order)
        lows = torch.tensor([self.prior_ranges[p][0] for p in self.param_order])
        highs = torch.tensor([self.prior_ranges[p][1] for p in self.param_order])
        self.low = lows.float()
        self.high = highs.float()

    def sample(
        self,
        sample_shape: Union[int, Tuple[int, ...]] = (1,),
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw uniform samples from the box.

        Args:
            sample_shape: Number of samples or a shape tuple ``(n,)``.
            generator: Optional torch random generator.

        Returns:
            Samples of shape ``(n, theta_dim)``.
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        n = int(np.prod(sample_shape))
        u = torch.rand(n, self.theta_dim, generator=generator)
        return self.low + u * (self.high - self.low)

    def sample_n(self, num_samples: int) -> torch.Tensor:
        """Alias of :meth:`sample` taking a plain integer (SBI_Fitter compatibility)."""
        return self.sample((num_samples,))

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Log density of ``theta`` under the box prior.

        Args:
            theta: Parameter values of shape ``(..., theta_dim)``.

        Returns:
            Log probabilities of shape ``(...,)`` (``-inf`` outside the box).
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)
        inside = ((theta >= self.low) & (theta <= self.high)).all(dim=-1)
        log_volume = torch.log(self.high - self.low).sum()
        out = torch.full(theta.shape[:-1], -torch.inf)
        out[inside] = -log_volume
        return out


class SimformerModel:
    """A trained all-conditional Simformer score model.

    The node order is ``[theta..., x...]``; condition masks are boolean vectors over
    the nodes with True marking observed variables. All public methods accept and
    return values in original (un-z-scored) units.
    """

    def __init__(
        self,
        net: ScoreTransformer,
        sde: BaseSDE,
        theta_dim: int,
        x_dim: int,
        node_ids: Optional[np.ndarray] = None,
        base_mask: Optional[np.ndarray] = None,
        z_score_mean: Optional[np.ndarray] = None,
        z_score_std: Optional[np.ndarray] = None,
        meta: Optional[dict] = None,
        sampling_defaults: Optional[dict] = None,
    ):
        """Initialize the wrapper.

        Args:
            net: Trained score network.
            sde: The diffusion SDE (holding per-node data statistics).
            theta_dim: Number of parameter nodes.
            x_dim: Number of data nodes.
            node_ids: Integer node ids, shape ``(theta_dim + x_dim,)``. Defaults to
                ``arange``.
            base_mask: Optional boolean base attention mask, shape ``(T, T)``.
            z_score_mean: Per-node z-score means, shape ``(T,)``, or None.
            z_score_std: Per-node z-score stddevs, shape ``(T,)``, or None.
            meta: Metadata dict (parameter/feature names, prior ranges, configs).
            sampling_defaults: Default sampling settings (``num_steps``, ``method``).
        """
        self.net = net
        self.sde = sde
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.num_nodes = self.theta_dim + self.x_dim
        if node_ids is None:
            node_ids = np.arange(self.num_nodes)
        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.base_mask = None if base_mask is None else np.asarray(base_mask, dtype=bool)
        self.z_score_mean = None if z_score_mean is None else np.asarray(z_score_mean, np.float32)
        self.z_score_std = None if z_score_std is None else np.asarray(z_score_std, np.float32)
        self.meta = meta or {}
        self.sampling_defaults = sampling_defaults or {"num_steps": 500, "method": "sde"}
        self.device = torch.device("cpu")

    def to(self, device: Union[str, torch.device]) -> "SimformerModel":
        """Move the model to ``device`` and return self."""
        self.device = torch.device(device)
        self.net.to(self.device)
        self.sde.to(self.device)
        return self

    # ------------------------------------------------------------------ helpers
    def _as_condition_mask(self, condition_mask) -> torch.Tensor:
        mask = torch.as_tensor(np.asarray(condition_mask), dtype=torch.bool)
        # make sure mask on the correct device for the score network
        mask = mask.to(self.device)
        if mask.shape != (self.num_nodes,):
            raise ValueError(
                f"condition_mask must have shape ({self.num_nodes},), got {tuple(mask.shape)}."
            )
        if mask.all():
            raise ValueError("condition_mask cannot condition on every node.")
        return mask

    def _z_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.z_score_mean is None:
            mean = torch.zeros(self.num_nodes, device=self.device)
            std = torch.ones(self.num_nodes, device=self.device)
        else:
            mean = torch.as_tensor(self.z_score_mean, device=self.device)
            std = torch.as_tensor(self.z_score_std, device=self.device)
        return mean, std

    def _resolve_edge_mask(self, edge_mask) -> Optional[torch.Tensor]:
        if edge_mask is None:
            if self.base_mask is None:
                return None
            edge_mask = self.base_mask
        mask = torch.as_tensor(np.asarray(edge_mask), dtype=torch.bool, device=self.device)
        return mask

    def score(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition_mask: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate the score estimate in z-scored space.

        Args:
            t: Scalar diffusion time (tensor).
            x: State of shape ``(B, T)``.
            condition_mask: Boolean mask of shape ``(T,)``.
            edge_mask: Optional attention mask; defaults to the stored base mask.

        Returns:
            Score of shape ``(B, T)``.
        """
        node_ids = torch.as_tensor(self.node_ids, device=x.device)
        raw = self.net(
            torch.atleast_1d(t),
            x[..., None],
            node_ids,
            condition_mask.to(x.device),
            edge_mask=self._resolve_edge_mask(edge_mask),
        )[..., 0]
        return raw / self.sde.output_scale(t)

    def _score_fn(self, condition_mask: torch.Tensor, edge_mask=None):
        def score_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.score(t, x, condition_mask, edge_mask=edge_mask)

        return score_fn

    def _init_x_T(
        self,
        num_samples: int,
        x_o_z: torch.Tensor,
        condition_mask: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        mean_end = self.sde.marginal_mean_end().to(self.device)
        std_end = self.sde.marginal_std_end().to(self.device)
        noise = torch.randn(num_samples, self.num_nodes, device=self.device, generator=generator)
        x_T = mean_end + std_end * noise
        x_T[:, condition_mask] = x_o_z
        return x_T

    def _prepare_x_o(self, x_o, condition_mask: torch.Tensor) -> torch.Tensor:
        """Z-score observed values ordered by the conditioned node indices."""
        x_o = torch.as_tensor(np.asarray(x_o, dtype=np.float32), device=self.device).reshape(-1)
        n_cond = int(condition_mask.sum())
        if x_o.numel() != n_cond:
            raise ValueError(f"x_o has {x_o.numel()} values but the mask conditions on {n_cond}.")
        mean, std = self._z_stats()
        return (x_o - mean[condition_mask]) / std[condition_mask]

    # ----------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        x_o,
        condition_mask,
        num_steps: Optional[int] = None,
        method: Optional[str] = None,
        edge_mask=None,
        generator: Optional[torch.Generator] = None,
    ) -> np.ndarray:
        """Sample the latent (unconditioned) nodes given observed values.

        Args:
            num_samples: Number of posterior/conditional samples to draw.
            x_o: Observed values (original units) for the conditioned nodes, in node
                order; shape ``(n_conditioned,)``.
            condition_mask: Boolean mask of shape ``(T,)``; True marks observed nodes.
            num_steps: Number of integration steps (default from
                ``sampling_defaults``).
            method: ``"sde"`` (Euler-Maruyama) or ``"ode"`` (Heun probability flow).
            edge_mask: Optional attention mask override.
            generator: Optional torch random generator on the model device.

        Returns:
            Samples of shape ``(num_samples, n_latent)`` in original units.

        Raises:
            ValueError: If ``method`` is unknown.
        """
        condition_mask = self._as_condition_mask(condition_mask)
        num_steps = num_steps or self.sampling_defaults.get("num_steps", 500)
        method = method or self.sampling_defaults.get("method", "sde")

        x_o_z = self._prepare_x_o(x_o, condition_mask)
        x_T = self._init_x_T(num_samples, x_o_z, condition_mask, generator)
        score_fn = self._score_fn(condition_mask, edge_mask=edge_mask)

        if method == "sde":
            x_final = euler_maruyama_reverse(
                score_fn, self.sde, x_T, condition_mask, num_steps=num_steps, generator=generator
            )
        elif method == "ode":
            x_final = heun_probability_flow(
                score_fn, self.sde, x_T, condition_mask, num_steps=num_steps
            )
        else:
            raise ValueError(f"Unknown sampling method '{method}'. Use 'sde' or 'ode'.")

        return self._extract_latents(x_final, condition_mask)

    def _extract_latents(self, x_final: torch.Tensor, condition_mask: torch.Tensor) -> np.ndarray:
        mean, std = self._z_stats()
        latents = x_final[:, ~condition_mask]
        latents = latents * std[~condition_mask] + mean[~condition_mask]
        return latents.cpu().numpy()

    @torch.no_grad()
    def sample_batched(
        self,
        num_samples: int,
        x_o_batch,
        condition_mask,
        batch_size: int = 100,
        **kwargs,
    ) -> np.ndarray:
        """Sample conditionals for a batch of observations.

        Args:
            num_samples: Samples per observation.
            x_o_batch: Observations of shape ``(n_obs, n_conditioned)``.
            condition_mask: Boolean mask of shape ``(T,)`` shared by all observations.
            batch_size: Number of observations integrated simultaneously (the
                effective state batch is ``batch_size * num_samples``).
            **kwargs: Forwarded to :meth:`sample` (``num_steps``, ``method``,
                ``edge_mask``, ``generator``).

        Returns:
            Samples of shape ``(n_obs, num_samples, n_latent)`` in original units.
        """
        condition_mask = self._as_condition_mask(condition_mask)
        x_o_batch = np.atleast_2d(np.asarray(x_o_batch, dtype=np.float32))
        n_obs = x_o_batch.shape[0]
        num_steps = kwargs.pop("num_steps", None) or self.sampling_defaults.get("num_steps", 500)
        method = kwargs.pop("method", None) or self.sampling_defaults.get("method", "sde")
        edge_mask = kwargs.pop("edge_mask", None)
        generator = kwargs.pop("generator", None)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        score_fn = self._score_fn(condition_mask, edge_mask=edge_mask)
        results = []
        for start in range(0, n_obs, batch_size):
            chunk = x_o_batch[start : start + batch_size]
            n_chunk = chunk.shape[0]
            mean, std = self._z_stats()
            x_o_z = (torch.as_tensor(chunk, device=self.device) - mean[condition_mask]) / std[
                condition_mask
            ]
            x_o_z = x_o_z.repeat_interleave(num_samples, dim=0)
            x_T = self._init_x_T(n_chunk * num_samples, x_o_z, condition_mask, generator)
            if method == "sde":
                x_final = euler_maruyama_reverse(
                    score_fn,
                    self.sde,
                    x_T,
                    condition_mask,
                    num_steps=num_steps,
                    generator=generator,
                )
            elif method == "ode":
                x_final = heun_probability_flow(
                    score_fn, self.sde, x_T, condition_mask, num_steps=num_steps
                )
            else:
                raise ValueError(f"Unknown sampling method '{method}'. Use 'sde' or 'ode'.")
            latents = self._extract_latents(x_final, condition_mask)
            results.append(latents.reshape(n_chunk, num_samples, -1))
        return np.concatenate(results, axis=0)

    @torch.no_grad()
    def sample_intervals(
        self,
        num_samples: int,
        x_o,
        condition_mask,
        constraint_mask,
        a=None,
        b=None,
        scale_bias: float = 0.0,
        num_steps: Optional[int] = None,
        edge_mask=None,
        generator: Optional[torch.Generator] = None,
    ) -> np.ndarray:
        """Sample with interval constraints on a subset of nodes via guidance.

        Constrained nodes are removed from the hard condition mask (a node cannot be
        both clamped and constrained) and guided towards the box ``[a, b]`` using the
        Tweedie generalized-guidance scheme.

        Args:
            num_samples: Number of samples to draw.
            x_o: Observed values (original units) for nodes that remain in the
                effective condition mask (``condition_mask & ~constraint_mask``),
                in node order.
            condition_mask: Boolean mask of shape ``(T,)`` of observed nodes.
            constraint_mask: Boolean mask of shape ``(T,)`` of interval-constrained
                nodes.
            a: Lower bounds in original units — scalar, array of shape
                ``(n_constrained,)``, or None for unbounded below.
            b: Upper bounds in original units, like ``a``.
            scale_bias: Additive bias in the guidance scale
                ``1 / (marginal_var(t) + bias)``; the faithful default is 0, but
                values of order 1e-2 improve stability for the VE SDE near ``t=0``.
            num_steps: Number of integration steps.
            edge_mask: Optional attention mask override.
            generator: Optional torch random generator.

        Returns:
            Samples of shape ``(num_samples, n_latent)`` in original units, where
            latent nodes are all nodes not in the *effective* condition mask
            (i.e. constrained nodes are included in the output).
        """
        condition_mask = self._as_condition_mask(condition_mask)
        constraint_mask = torch.as_tensor(
            np.asarray(constraint_mask), dtype=torch.bool, device=self.device
        )
        if constraint_mask.shape != (self.num_nodes,):
            raise ValueError(f"constraint_mask must have shape ({self.num_nodes},).")
        effective_condition = condition_mask & ~constraint_mask
        num_steps = num_steps or self.sampling_defaults.get("num_steps", 500)

        mean, std = self._z_stats()
        a_full = self._expand_bounds(a, constraint_mask, mean, std)
        b_full = self._expand_bounds(b, constraint_mask, mean, std)

        x_o_z = self._prepare_x_o(x_o, effective_condition)
        x_T = self._init_x_T(num_samples, x_o_z, effective_condition, generator)
        score_fn = self._score_fn(effective_condition, edge_mask=edge_mask)

        def constraint_score_fn(x0_hat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            scale = 1.0 / (self.sde.transition_var(t) + scale_bias)
            return interval_constraint_score(x0_hat, scale, constraint_mask, a=a_full, b=b_full)

        x_final = guided_euler_maruyama(
            score_fn,
            self.sde,
            x_T,
            effective_condition,
            constraint_score_fn,
            num_steps=num_steps,
            generator=generator,
        )
        return self._extract_latents(x_final, effective_condition)

    def _expand_bounds(self, bound, constraint_mask, mean, std) -> Optional[torch.Tensor]:
        """Expand user bounds to a z-scored full-node vector (non-finite = unbounded)."""
        if bound is None:
            return None
        full = torch.full((self.num_nodes,), torch.nan, device=self.device)
        values = torch.as_tensor(
            np.broadcast_to(
                np.asarray(bound, dtype=np.float32), (int(constraint_mask.sum()),)
            ).copy(),
            device=self.device,
        )
        full[constraint_mask] = values
        return (full - mean) / std

    # ---------------------------------------------------------------- log prob
    def log_prob(
        self,
        theta,
        x_o,
        condition_mask,
        num_steps: int = 250,
        divergence: str = "exact",
        hutchinson_probes: int = 8,
        edge_mask=None,
        generator: Optional[torch.Generator] = None,
    ) -> np.ndarray:
        """Log probability of latent values via the probability-flow ODE.

        Args:
            theta: Latent values (original units), shape ``(n, n_latent)`` or
                ``(n_latent,)``, in node order of the unconditioned nodes.
            x_o: Observed values (original units) for the conditioned nodes — either a
                single vector shared by all rows of ``theta``, or one row per ``theta``
                row (shape ``(n, n_conditioned)``).
            condition_mask: Boolean mask of shape ``(T,)``; True marks observed nodes.
            num_steps: Number of ODE steps.
            divergence: ``"exact"`` or ``"hutchinson"``.
            hutchinson_probes: Probes for the Hutchinson estimator.
            edge_mask: Optional attention mask override.
            generator: Optional torch random generator.

        Returns:
            Log probabilities of shape ``(n,)`` (scalar array for 1D input),
            in original units (z-score Jacobian included).
        """
        condition_mask = self._as_condition_mask(condition_mask)
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float32))
        n_latent = int((~condition_mask).sum())
        if theta.shape[1] != n_latent:
            raise ValueError(f"theta must have {n_latent} columns, got {theta.shape[1]}.")

        mean, std = self._z_stats()
        n_cond = int(condition_mask.sum())
        x_o_arr = np.atleast_2d(np.asarray(x_o, dtype=np.float32)).reshape(-1, n_cond)
        if x_o_arr.shape[0] not in (1, theta.shape[0]):
            raise ValueError(
                f"x_o must be a single vector or one row per theta row, got {x_o_arr.shape}."
            )
        x_o_t = torch.as_tensor(x_o_arr, device=self.device)
        x_o_z = (x_o_t - mean[condition_mask]) / std[condition_mask]
        theta_t = torch.as_tensor(theta, device=self.device)
        theta_z = (theta_t - mean[~condition_mask]) / std[~condition_mask]

        x0 = torch.zeros(theta.shape[0], self.num_nodes, device=self.device)
        x0[:, condition_mask] = x_o_z
        x0[:, ~condition_mask] = theta_z

        score_fn = self._score_fn(condition_mask, edge_mask=edge_mask)
        log_p_z = probability_flow_log_prob(
            score_fn,
            self.sde,
            x0,
            condition_mask,
            num_steps=num_steps,
            divergence=divergence,
            hutchinson_probes=hutchinson_probes,
            generator=generator,
        )
        jacobian = torch.log(std[~condition_mask]).sum()
        out = (log_p_z - jacobian).cpu().numpy()
        return out if out.size > 1 else out.reshape(())

    # ------------------------------------------------------------ serialization
    def save(self, path: str) -> None:
        """Serialize the model to ``path`` as a plain ``torch.save`` dict.

        Args:
            path: Destination file path.
        """
        payload = {
            "format_version": SAVE_FORMAT_VERSION,
            "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
            "model_config": self.net.config,
            "sde_config": self.sde.config,
            "sde_x0_mean": self.sde.x0_mean.cpu().numpy(),
            "sde_x0_var": self.sde.x0_var.cpu().numpy(),
            "z_score_mean": self.z_score_mean,
            "z_score_std": self.z_score_std,
            "theta_dim": self.theta_dim,
            "x_dim": self.x_dim,
            "node_ids": self.node_ids,
            "base_mask": self.base_mask,
            "meta": self.meta,
            "sampling_defaults": self.sampling_defaults,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: Union[str, torch.device] = "cpu") -> "SimformerModel":
        """Load a model saved with :meth:`save`.

        Args:
            path: Path to the saved file.
            map_location: Device to load onto.

        Returns:
            The reconstructed model.

        Raises:
            ValueError: If the file is not a payload produced by :meth:`save`, or
                has an unknown format version.

        Warning:
            This loads with ``torch.load(..., weights_only=False)``, which can
            execute arbitrary code when deserializing a malicious file. Only
            call this on ``path`` values you trust (e.g. files produced by
            :meth:`save` on this machine or received from a trusted source) —
            never on checkpoints from an untrusted or unauthenticated origin.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported Simformer save format version: {payload!r}.")
        version = payload.get("format_version")
        if version != SAVE_FORMAT_VERSION:
            raise ValueError(f"Unsupported Simformer save format version: {version}.")
        net = ScoreTransformer(**payload["model_config"])
        net.load_state_dict(payload["state_dict"])
        net.eval()
        sde_config = dict(payload["sde_config"])
        sde = build_sde(
            sde_config.pop("name"),
            payload["sde_x0_mean"],
            payload["sde_x0_var"],
            **sde_config,
        )
        model = cls(
            net=net,
            sde=sde,
            theta_dim=payload["theta_dim"],
            x_dim=payload["x_dim"],
            node_ids=payload["node_ids"],
            base_mask=payload["base_mask"],
            z_score_mean=payload["z_score_mean"],
            z_score_std=payload["z_score_std"],
            meta=payload["meta"],
            sampling_defaults=payload["sampling_defaults"],
        )
        return model.to(map_location)


def posterior_mask(theta_dim: int, x_dim: int) -> np.ndarray:
    """Standard posterior condition mask ``[False]*theta_dim + [True]*x_dim``.

    Args:
        theta_dim: Number of parameter nodes.
        x_dim: Number of data nodes.

    Returns:
        Boolean array of shape ``(theta_dim + x_dim,)``.
    """
    return np.array([False] * theta_dim + [True] * x_dim)


__all__ = ["SimformerModel", "UniformBoxPrior", "posterior_mask"]
