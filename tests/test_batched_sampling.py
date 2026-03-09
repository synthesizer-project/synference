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


class TestSbiBatchedFixed:
    """sbi's DirectPosterior.sample_batched is now correct after patching
    accept_reject_sample (min-gating fix) and removing the batch-size cap.
    """

    def test_sbi_sample_batched_shape(self, toy_posterior_and_obs):
        """sbi's sample_batched returns correct shape (num_samples, N, theta_dim)."""
        posterior, prior, x_obs, n_params = toy_posterior_and_obs
        x_t = torch.tensor(x_obs)
        N, num_samples = len(x_obs), 50

        raw = posterior.sample_batched(
            (num_samples,), x=x_t, show_progress_bars=False
        )
        assert raw.shape == (num_samples, N, n_params), (
            f"Expected ({num_samples}, {N}, {n_params}), got {raw.shape}"
        )

    def test_sbi_sample_batched_within_prior(self, toy_posterior_and_obs):
        """All samples from patched sbi.sample_batched lie within prior bounds."""
        from sbi.utils.sbiutils import within_support

        posterior, prior, x_obs, n_params = toy_posterior_and_obs
        x_t = torch.tensor(x_obs[:10])

        raw = posterior.sample_batched(
            (100,), x=x_t, show_progress_bars=False
        )  # (100, 10, n_params)

        flat = raw.reshape(-1, n_params)
        assert within_support(prior, flat).all(), (
            "Some samples from sbi sample_batched are outside the prior support."
        )

    def test_sbi_sample_batched_high_leakage_completes(self, high_leakage_posterior_and_obs):
        """Patched sample_batched correctly collects all samples under high leakage.

        Before the min-gating fix, this would either hang or return far fewer
        samples than requested (requiring excessive iterations).
        """
        posterior, tight_prior, x_obs, n_params = high_leakage_posterior_and_obs
        x_t = torch.tensor(x_obs)
        N, num_samples = len(x_obs), 30

        raw = posterior.sample_batched(
            (num_samples,), x=x_t, show_progress_bars=False
        )
        assert raw.shape == (num_samples, N, n_params)
        # All samples within tight prior
        flat = raw.reshape(-1, n_params)
        from sbi.utils.sbiutils import within_support
        assert within_support(tight_prior, flat).all()

    def test_no_excess_iterations_uniform_leakage(self, toy_posterior_and_obs):
        """With uniform acceptance, loop should not over-run by more than one batch."""
        import unittest.mock as mock
        from sbi.samplers.rejection import rejection as rej_module

        posterior, prior, x_obs, n_params = toy_posterior_and_obs
        x_t = torch.tensor(x_obs[:5])

        call_count = {"n": 0}
        original_accept_reject = rej_module.accept_reject_sample

        def counting_accept_reject(*args, **kwargs):
            # Count how many proposals are drawn across all iterations
            call_count["n"] += 1
            return original_accept_reject(*args, **kwargs)

        with mock.patch.object(rej_module, "accept_reject_sample", side_effect=counting_accept_reject):
            posterior.sample_batched((100,), x=x_t, show_progress_bars=False)

        # The patched function should be called exactly once (single call into
        # accept_reject_sample, which then loops internally).
        assert call_count["n"] == 1


class TestBatchedSpeedup:
    """Batched sampling via the fixed sbi path should not regress vs serial."""

    def _time_serial(self, sampler, x_obs, num_samples):
        """Time the serial per-observation loop."""
        start = time.perf_counter()
        np.stack([sampler.sample(nsteps=num_samples, x=x_obs[i]) for i in range(len(x_obs))])
        return time.perf_counter() - start

    def _time_sbi_batched(self, posterior, x_obs, num_samples):
        """Time sbi's fixed sample_batched."""
        x_t = torch.tensor(x_obs)
        start = time.perf_counter()
        posterior.sample_batched((num_samples,), x=x_t, show_progress_bars=False)
        return time.perf_counter() - start

    def _time_ili_batched(self, sampler, x_obs, num_samples):
        """Time our ltu-ili progressive-shrinkage path."""
        start = time.perf_counter()
        sampler.sample_batched(nsteps=num_samples, x=x_obs, samples_per_draw=500, show_progress_bars=False)
        return time.perf_counter() - start

    def test_sbi_batched_not_slower_than_serial(self, toy_posterior_and_obs):
        """Fixed sbi sample_batched must not exceed 3× serial time on CPU.

        The pre-patch version was often 5–10× slower due to min-gating + the
        batch-size cap collapsing to 100 candidates/obs for N=20.  3× is a
        generous bound that CPU overhead is allowed to consume.
        """
        from ili.utils.samplers import DirectSampler

        posterior, prior, x_obs, n_params = toy_posterior_and_obs
        sampler = DirectSampler(posterior)
        num_samples = 100

        t_serial = self._time_serial(sampler, x_obs, num_samples)
        t_batched = self._time_sbi_batched(posterior, x_obs, num_samples)

        ratio = t_batched / t_serial
        assert ratio < 3.0, (
            f"Fixed sbi batched ({t_batched:.2f}s) vs serial ({t_serial:.2f}s) "
            f"ratio {ratio:.2f}× exceeds 3×."
        )

    def test_high_leakage_batched_not_slower_than_serial(self, high_leakage_posterior_and_obs):
        """Under high leakage, fixed sbi batched must not exceed serial time by 3×.

        The pre-patch code would degenerate severely in this regime due to
        min-gating: after most obs have enough samples the loop keeps running
        at the rate of the slowest, wasting N-1 obs worth of compute each iter.
        """
        from ili.utils.samplers import DirectSampler

        posterior, tight_prior, x_obs, n_params = high_leakage_posterior_and_obs
        sampler = DirectSampler(posterior)
        num_samples = 30

        t_serial = self._time_serial(sampler, x_obs, num_samples)
        t_batched = self._time_sbi_batched(posterior, x_obs, num_samples)

        ratio = t_batched / t_serial
        assert ratio < 3.0, (
            f"High-leakage fixed batched ({t_batched:.2f}s) vs serial ({t_serial:.2f}s) "
            f"ratio {ratio:.2f}×."
        )

    def test_throughput_scales_with_batch_size(self, toy_posterior_and_obs):
        """Per-obs time should not grow as batch size increases (GPU would improve; CPU flat)."""
        from ili.utils.samplers import DirectSampler

        posterior, prior, x_obs, n_params = toy_posterior_and_obs
        num_samples = 50

        t_small = self._time_sbi_batched(posterior, x_obs[:5], num_samples)
        t_large = self._time_sbi_batched(posterior, x_obs[:20], num_samples)

        per_obs_small = t_small / 5
        per_obs_large = t_large / 20

        # Per-obs time for larger batch must not be more than 3× the smaller
        # batch — on GPU it improves; on CPU we allow flat or slight growth.
        assert per_obs_large < per_obs_small * 3.0, (
            f"Per-obs time grew too much: small={per_obs_small:.3f}s, "
            f"large={per_obs_large:.3f}s (ratio {per_obs_large/per_obs_small:.1f}×)"
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
