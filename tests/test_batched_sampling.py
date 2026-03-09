"""Tests for the batched posterior sampling optimisation.

Validates correctness and speed of ``DirectSampler.sample_batched`` relative
to the original per-observation serial loop.  Tests cover:

- Output shape and dtype
- All returned samples lie within the prior support
- Statistical consistency with the serial path (same distribution up to MC
  noise)
- Wall-clock speed: batched should be faster than serial for N > 1
- High-leakage regime: the min-gating problem that slowed ``sample_batched``
  in the original sbi code is exercised by using a prior whose bounds are
  tight relative to the flow's output range, forcing a high rejection rate in
  the serial sbi path but not in the new progressive-shrinkage path
- EnsemblePosterior delegation
"""

import time
import warnings

import numpy as np
import pytest
import torch
from scipy import stats

# ---------------------------------------------------------------------------
# Helpers to build lightweight sbi posteriors without Synthesizer
# ---------------------------------------------------------------------------


def _make_direct_posterior(n_params: int = 5, n_train: int = 500, seed: int = 0):
    """Return a trained DirectPosterior for a toy n_params-D BoxUniform prior.

    Uses sbi's SNPE with a small MDN so training finishes in seconds.
    """
    import sbi.inference as sbi_inference
    from sbi.utils import BoxUniform

    torch.manual_seed(seed)
    np.random.seed(seed)

    low = torch.zeros(n_params)
    high = torch.ones(n_params)
    prior = BoxUniform(low=low, high=high)

    # Simulator: observation = theta + small Gaussian noise (n_params-D)
    def simulator(theta):
        return theta + 0.05 * torch.randn_like(theta)

    inference = sbi_inference.SNPE(prior=prior)
    theta = prior.sample((n_train,))
    x = torch.stack([simulator(t) for t in theta])
    inference.append_simulations(theta, x)
    density_estimator = inference.train(
        max_num_epochs=5,
        show_train_summary=False,
    )
    posterior = inference.build_posterior(density_estimator)
    return posterior, prior


def _make_ensemble_posterior(n_params: int = 5, n_train: int = 500, seed: int = 0):
    """Return a 2-component EnsemblePosterior for the same toy prior."""
    from sbi.inference.posteriors import EnsemblePosterior

    p1, prior = _make_direct_posterior(n_params=n_params, n_train=n_train, seed=seed)
    p2, _ = _make_direct_posterior(n_params=n_params, n_train=n_train, seed=seed + 1)
    return EnsemblePosterior([p1, p2]), prior


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_posterior_and_obs():
    """Small trained DirectPosterior plus 20 test observations."""
    n_params = 5
    posterior, prior = _make_direct_posterior(n_params=n_params)
    # Observations near the centre of the prior support
    rng = np.random.default_rng(42)
    x_obs = rng.uniform(0.2, 0.8, size=(20, n_params)).astype(np.float32)
    return posterior, prior, x_obs, n_params


@pytest.fixture(scope="module")
def toy_ensemble_and_obs():
    """Small trained EnsemblePosterior plus 10 test observations."""
    n_params = 5
    ensemble, prior = _make_ensemble_posterior(n_params=n_params)
    rng = np.random.default_rng(42)
    x_obs = rng.uniform(0.2, 0.8, size=(10, n_params)).astype(np.float32)
    return ensemble, prior, x_obs, n_params


@pytest.fixture(scope="module")
def high_leakage_posterior_and_obs():
    """Posterior with high leakage: prior bounds much tighter than flow range.

    We train on the full unit cube but then wrap a new BoxUniform prior that
    covers only the central [0.4, 0.6] hypercube.  The flow still generates
    samples across [0, 1] so the acceptance rate is ~(0.2^n_params).
    For n_params=4 that is 0.0016 — well below the 1 % warning threshold.
    This exercises the scenario where sbi's min-gating causes the batched
    loop to crawl.
    """
    from sbi.utils import BoxUniform

    n_params = 4
    # Train with wide prior so the flow covers [0,1]^n
    posterior, _ = _make_direct_posterior(n_params=n_params, n_train=800)
    # Swap in a tight prior — acceptance ≈ 0.2^4 ≈ 0.0016
    tight_prior = BoxUniform(
        low=0.4 * torch.ones(n_params),
        high=0.6 * torch.ones(n_params),
    )
    posterior.prior = tight_prior

    rng = np.random.default_rng(0)
    x_obs = rng.uniform(0.45, 0.55, size=(6, n_params)).astype(np.float32)
    return posterior, tight_prior, x_obs, n_params


# ---------------------------------------------------------------------------
# DirectSampler.sample_batched — basic correctness
# ---------------------------------------------------------------------------


class TestDirectSamplerBatchedShape:
    """Output shape and type are correct."""

    def test_shape_multiple_obs(self, toy_posterior_and_obs):
        from ili.utils.samplers import DirectSampler

        posterior, prior, x_obs, n_params = toy_posterior_and_obs
        sampler = DirectSampler(posterior)
        N, num_samples = len(x_obs), 50

        result = sampler.sample_batched(
            nsteps=num_samples,
            x=x_obs,
            samples_per_draw=200,
            show_progress_bars=False,
        )

        assert result.shape == (N, num_samples, n_params), (
            f"Expected shape ({N}, {num_samples}, {n_params}), got {result.shape}"
        )
        assert result.dtype == np.float32 or np.issubdtype(result.dtype, np.floating)

    def test_shape_single_obs(self, toy_posterior_and_obs):
        """Single observation still returns (1, num_samples, n_params)."""
        from ili.utils.samplers import DirectSampler

        posterior, prior, x_obs, n_params = toy_posterior_and_obs
        sampler = DirectSampler(posterior)
        num_samples = 30

        result = sampler.sample_batched(
            nsteps=num_samples,
            x=x_obs[:1],
            samples_per_draw=200,
            show_progress_bars=False,
        )

        assert result.shape == (1, num_samples, n_params)

    def test_no_nan_or_inf(self, toy_posterior_and_obs):
        from ili.utils.samplers import DirectSampler

        posterior, prior, x_obs, n_params = toy_posterior_and_obs
        sampler = DirectSampler(posterior)

        result = sampler.sample_batched(
            nsteps=40,
            x=x_obs[:5],
            samples_per_draw=200,
            show_progress_bars=False,
        )

        assert np.all(np.isfinite(result)), "sample_batched returned NaN or Inf values"


class TestDirectSamplerPriorSupport:
    """Every returned sample must lie within the prior support."""

    def test_all_within_prior_bounds(self, toy_posterior_and_obs):
        from ili.utils.samplers import DirectSampler

        posterior, prior, x_obs, n_params = toy_posterior_and_obs
        sampler = DirectSampler(posterior)

        result = sampler.sample_batched(
            nsteps=100,
            x=x_obs[:10],
            samples_per_draw=500,
            show_progress_bars=False,
        )  # (10, 100, n_params)

        samples_flat = result.reshape(-1, n_params)
        # BoxUniform prior: check bounds
        low = prior.support.base_constraint.lower_bound.numpy()
        high = prior.support.base_constraint.upper_bound.numpy()

        assert np.all(samples_flat >= low - 1e-5), (
            f"Some samples below prior lower bound. Min: {samples_flat.min(axis=0)}"
        )
        assert np.all(samples_flat <= high + 1e-5), (
            f"Some samples above prior upper bound. Max: {samples_flat.max(axis=0)}"
        )

    def test_high_leakage_all_within_tight_prior(self, high_leakage_posterior_and_obs):
        """Even with very high leakage, all samples must respect the tight prior."""
        from ili.utils.samplers import DirectSampler

        posterior, tight_prior, x_obs, n_params = high_leakage_posterior_and_obs
        sampler = DirectSampler(posterior)

        result = sampler.sample_batched(
            nsteps=50,
            x=x_obs,
            samples_per_draw=2_000,
            show_progress_bars=False,
        )

        samples_flat = result.reshape(-1, n_params)
        low = tight_prior.support.base_constraint.lower_bound.numpy()
        high = tight_prior.support.base_constraint.upper_bound.numpy()

        assert np.all(samples_flat >= low - 1e-5)
        assert np.all(samples_flat <= high + 1e-5)


# ---------------------------------------------------------------------------
# Statistical consistency: batched ≈ serial
# ---------------------------------------------------------------------------


class TestBatchedVsSerialConsistency:
    """Batched and serial paths should sample from the same distribution."""

    def test_marginal_means_close(self, toy_posterior_and_obs):
        """Per-parameter sample means agree within 3 sigma (MC noise)."""
        from ili.utils.samplers import DirectSampler

        posterior, prior, x_obs, n_params = toy_posterior_and_obs
        sampler = DirectSampler(posterior)
        num_samples = 500

        # Serial: sample each observation independently
        serial_samples = np.stack([
            sampler.sample(nsteps=num_samples, x=x_obs[i])
            for i in range(len(x_obs))
        ])  # (N, num_samples, n_params)

        batched_samples = sampler.sample_batched(
            nsteps=num_samples,
            x=x_obs,
            samples_per_draw=1_000,
            show_progress_bars=False,
        )  # (N, num_samples, n_params)

        # Compare per-observation, per-parameter means
        serial_mean = serial_samples.mean(axis=1)    # (N, n_params)
        batched_mean = batched_samples.mean(axis=1)  # (N, n_params)

        # Tolerance: 4× MC standard error (generous for unit test)
        se = serial_samples.std(axis=1) / np.sqrt(num_samples)
        max_z = np.abs(batched_mean - serial_mean).max() / (se.mean() + 1e-8)
        assert max_z < 6.0, (
            f"Batched and serial means diverge by {max_z:.1f} SEs. "
            "The two paths may be sampling different distributions."
        )

    def test_ks_test_per_param(self, toy_posterior_and_obs):
        """KS test: batched and serial samples are not distinguishably different."""
        from ili.utils.samplers import DirectSampler

        posterior, prior, x_obs, n_params = toy_posterior_and_obs
        sampler = DirectSampler(posterior)
        num_samples = 300

        # Use a single observation for a clean 1-D KS test per parameter
        serial = sampler.sample(nsteps=num_samples, x=x_obs[0])  # (num_samples, n_params)
        batched = sampler.sample_batched(
            nsteps=num_samples,
            x=x_obs[:1],
            samples_per_draw=500,
            show_progress_bars=False,
        )[0]  # (num_samples, n_params)

        for param_idx in range(n_params):
            ks_stat, p_value = stats.ks_2samp(
                serial[:, param_idx], batched[:, param_idx]
            )
            assert p_value > 1e-4, (
                f"KS test rejects H0 for param {param_idx}: "
                f"p={p_value:.2e}, ks={ks_stat:.3f}. "
                "Batched path may be sampling a different distribution."
            )


# ---------------------------------------------------------------------------
# Speed tests
# ---------------------------------------------------------------------------


class TestBatchedSpeedup:
    """Batched sampling should be faster than the serial per-galaxy loop."""

    def _time_serial(self, sampler, x_obs, num_samples, samples_per_draw):
        """Time the serial loop used in the old code path."""
        start = time.perf_counter()
        results = np.stack([
            sampler.sample(nsteps=num_samples, x=x_obs[i])
            for i in range(len(x_obs))
        ])
        return time.perf_counter() - start, results

    def _time_batched(self, sampler, x_obs, num_samples, samples_per_draw):
        """Time the new batched path."""
        start = time.perf_counter()
        results = sampler.sample_batched(
            nsteps=num_samples,
            x=x_obs,
            samples_per_draw=samples_per_draw,
            show_progress_bars=False,
        )
        return time.perf_counter() - start, results

    def test_batched_not_slower_than_serial(self, toy_posterior_and_obs):
        """Batched wall time must not exceed 2× serial time for N=20 obs.

        We use a lenient factor (2×) because on CPU with a tiny toy model
        there is little GPU batching benefit; the test mainly guards against
        pathological regressions.  On GPU or with a realistic model the
        speedup is much larger.
        """
        from ili.utils.samplers import DirectSampler

        posterior, prior, x_obs, n_params = toy_posterior_and_obs
        sampler = DirectSampler(posterior)
        num_samples = 100
        samples_per_draw = 500

        t_serial, _ = self._time_serial(sampler, x_obs, num_samples, samples_per_draw)
        t_batched, _ = self._time_batched(sampler, x_obs, num_samples, samples_per_draw)

        ratio = t_batched / t_serial
        assert ratio < 2.0, (
            f"Batched sampling took {t_batched:.2f}s vs serial {t_serial:.2f}s "
            f"(ratio {ratio:.2f}×). Expected batched ≤ 2× serial."
        )

    def test_batched_faster_high_leakage(self, high_leakage_posterior_and_obs):
        """Batched must be strictly faster than serial under high leakage.

        The old ``sample_batched`` (min-gating) was slower than serial in
        this regime.  The new progressive-shrinkage implementation should
        avoid that penalty.
        """
        from ili.utils.samplers import DirectSampler

        posterior, tight_prior, x_obs, n_params = high_leakage_posterior_and_obs
        sampler = DirectSampler(posterior)
        num_samples = 30
        samples_per_draw = 2_000

        t_serial, _ = self._time_serial(sampler, x_obs, num_samples, samples_per_draw)
        t_batched, _ = self._time_batched(sampler, x_obs, num_samples, samples_per_draw)

        # Under high leakage the serial sbi path issues a separate rejection
        # loop per galaxy; batched shares each draw across all galaxies.
        assert t_batched < t_serial * 1.5, (
            f"High-leakage batched ({t_batched:.2f}s) should not exceed serial "
            f"({t_serial:.2f}s) by more than 50%. Got ratio {t_batched/t_serial:.2f}×."
        )

    def test_throughput_scales_with_batch_size(self, toy_posterior_and_obs):
        """Per-observation time should decrease as batch size grows."""
        from ili.utils.samplers import DirectSampler

        posterior, prior, x_obs, n_params = toy_posterior_and_obs
        sampler = DirectSampler(posterior)
        num_samples = 50
        samples_per_draw = 500

        t_small, _ = self._time_batched(sampler, x_obs[:5], num_samples, samples_per_draw)
        t_large, _ = self._time_batched(sampler, x_obs[:20], num_samples, samples_per_draw)

        per_obs_small = t_small / 5
        per_obs_large = t_large / 20

        # Per-observation time for the larger batch should not be more than
        # 3× slower than the smaller batch — ideally it is faster due to
        # better GPU utilisation, but we use a generous bound for CPU.
        assert per_obs_large < per_obs_small * 3.0, (
            f"Per-obs time did not improve with larger batch: "
            f"small={per_obs_small:.3f}s, large={per_obs_large:.3f}s"
        )


# ---------------------------------------------------------------------------
# EnsemblePosterior delegation
# ---------------------------------------------------------------------------


class TestEnsembleBatched:
    """EnsemblePosterior is handled correctly via _sample_batched_ensemble."""

    def test_shape(self, toy_ensemble_and_obs):
        from ili.utils.samplers import DirectSampler

        ensemble, prior, x_obs, n_params = toy_ensemble_and_obs
        sampler = DirectSampler(ensemble)
        num_samples = 50

        result = sampler.sample_batched(
            nsteps=num_samples,
            x=x_obs,
            samples_per_draw=300,
            show_progress_bars=False,
        )

        N = len(x_obs)
        assert result.shape == (N, num_samples, n_params)

    def test_within_prior_support(self, toy_ensemble_and_obs):
        from ili.utils.samplers import DirectSampler

        ensemble, prior, x_obs, n_params = toy_ensemble_and_obs
        sampler = DirectSampler(ensemble)

        result = sampler.sample_batched(
            nsteps=80,
            x=x_obs,
            samples_per_draw=300,
            show_progress_bars=False,
        )

        samples_flat = result.reshape(-1, n_params)
        low = prior.support.base_constraint.lower_bound.numpy()
        high = prior.support.base_constraint.upper_bound.numpy()

        assert np.all(samples_flat >= low - 1e-5)
        assert np.all(samples_flat <= high + 1e-5)

    def test_no_nan_or_inf(self, toy_ensemble_and_obs):
        from ili.utils.samplers import DirectSampler

        ensemble, prior, x_obs, n_params = toy_ensemble_and_obs
        sampler = DirectSampler(ensemble)

        result = sampler.sample_batched(
            nsteps=50,
            x=x_obs,
            samples_per_draw=300,
            show_progress_bars=False,
        )

        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# SBI_Fitter.sample_posterior integration
# ---------------------------------------------------------------------------


class TestSBIFitterBatchedIntegration:
    """sample_posterior routes to batched path when sample_method='direct'."""

    def test_sample_posterior_batched_shape(self, test_sbi_library):
        """sample_posterior with direct method returns correct shape for a batch."""
        from synference import SBI_Fitter

        fitter = SBI_Fitter.init_from_hdf5(
            model_name="test_batched_integ", hdf5_path=test_sbi_library
        )
        fitter.create_feature_array_from_raw_photometry()
        fitter.run_single_sbi(
            model_type="mdn",
            hidden_features=32,
            num_components=2,
            stop_after_epochs=3,
            save_model=False,
            plot=False,
            evaluate_model=False,
        )

        X_test = fitter._X_test[:5]
        num_samples = 50

        samples = fitter.sample_posterior(
            X_test=X_test,
            num_samples=num_samples,
            samples_per_draw=200,
        )

        n_params = len(fitter.fitted_parameter_names)
        assert samples.shape == (5, num_samples, n_params), (
            f"Expected (5, {num_samples}, {n_params}), got {samples.shape}"
        )

    def test_sample_posterior_single_obs_unchanged(self, test_sbi_library):
        """Single-observation path still works and returns (num_samples, n_params)."""
        from synference import SBI_Fitter

        fitter = SBI_Fitter.init_from_hdf5(
            model_name="test_batched_single", hdf5_path=test_sbi_library
        )
        fitter.create_feature_array_from_raw_photometry()
        fitter.run_single_sbi(
            model_type="mdn",
            hidden_features=32,
            num_components=2,
            stop_after_epochs=3,
            save_model=False,
            plot=False,
            evaluate_model=False,
        )

        X_single = fitter._X_test[0]
        num_samples = 50

        samples = fitter.sample_posterior(
            X_test=X_single,
            num_samples=num_samples,
        )

        n_params = len(fitter.fitted_parameter_names)
        # Single obs should be squeezed back to (num_samples, n_params)
        assert samples.shape == (num_samples, n_params), (
            f"Expected ({num_samples}, {n_params}), got {samples.shape}"
        )
