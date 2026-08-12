"""Tests for the native PyTorch Simformer implementation."""

import numpy as np
import pytest
import torch

from synference.simformer import (
    DEFAULT_MODEL_CONFIG,
    VESDE,
    VPSDE,
    ScoreTransformer,
    SimformerModel,
    UniformBoxPrior,
    build_base_mask,
    denoising_score_matching_loss,
    euler_maruyama_reverse,
    interval_constraint_score,
    merge_config,
    structured_random_condition_mask,
    train_simformer,
)

THETA_DIM = 2
X_DIM = 3
NUM_NODES = THETA_DIM + X_DIM


@pytest.fixture
def small_net():
    """A small randomly initialized score network."""
    torch.manual_seed(0)
    return ScoreTransformer(num_nodes=NUM_NODES, num_layers=2, token_dim=16, condition_token_dim=4)


@pytest.fixture
def random_model(small_net):
    """An untrained SimformerModel wrapping the small network."""
    sde = VESDE(np.zeros(NUM_NODES), np.ones(NUM_NODES))
    return SimformerModel(
        net=small_net,
        sde=sde,
        theta_dim=THETA_DIM,
        x_dim=X_DIM,
        z_score_mean=np.arange(NUM_NODES, dtype=np.float32),
        z_score_std=np.linspace(1.0, 2.0, NUM_NODES).astype(np.float32),
        meta={"param_names": ["a", "b"], "feature_names": ["f1", "f2", "f3"]},
    )


class TestScoreTransformer:
    """Architecture-level tests."""

    def test_forward_shapes(self, small_net):
        """Output has one scalar per token for various mask shapes."""
        batch = 4
        t = torch.rand(batch)
        x = torch.randn(batch, NUM_NODES, 1)
        node_ids = torch.arange(NUM_NODES)
        cond_1d = torch.zeros(NUM_NODES, dtype=torch.bool)
        cond_2d = torch.zeros(batch, NUM_NODES, dtype=torch.bool)
        edge_2d = torch.ones(NUM_NODES, NUM_NODES, dtype=torch.bool)
        edge_3d = torch.ones(batch, NUM_NODES, NUM_NODES, dtype=torch.bool)

        for cond in (cond_1d, cond_2d):
            for edge in (None, edge_2d, edge_3d):
                out = small_net(t, x, node_ids, cond, edge_mask=edge)
                assert out.shape == (batch, NUM_NODES, 1)

    def test_edge_mask_isolation(self, small_net):
        """With an identity attention mask, node outputs only depend on own input."""
        t = torch.full((1,), 0.5)
        node_ids = torch.arange(NUM_NODES)
        cond = torch.zeros(NUM_NODES, dtype=torch.bool)
        eye_mask = torch.eye(NUM_NODES, dtype=torch.bool)

        x = torch.randn(1, NUM_NODES, 1)
        x_perturbed = x.clone()
        x_perturbed[0, 1, 0] += 10.0

        out = small_net(t, x, node_ids, cond, edge_mask=eye_mask)
        out_perturbed = small_net(t, x_perturbed, node_ids, cond, edge_mask=eye_mask)
        # Node 1 changes, all others are unaffected.
        assert not torch.allclose(out[0, 1], out_perturbed[0, 1])
        unchanged = [i for i in range(NUM_NODES) if i != 1]
        assert torch.allclose(out[0, unchanged], out_perturbed[0, unchanged], atol=1e-6)

        # Dense attention propagates the perturbation everywhere.
        out_dense = small_net(t, x, node_ids, cond)
        out_dense_pert = small_net(t, x_perturbed, node_ids, cond)
        assert not torch.allclose(out_dense[0, 0], out_dense_pert[0, 0])

    def test_frozen_embeddings(self, small_net):
        """Node-id embedding and Fourier projection stay fixed; condition token trains."""
        node_embedding_before = small_net.tokenizer.node_embedding.weight.clone()
        fourier_before = small_net.time_embedding.B.clone()
        condition_before = small_net.condition_token.clone()

        optimizer = torch.optim.Adam(
            [p for p in small_net.parameters() if p.requires_grad], lr=1e-2
        )
        t = torch.rand(8)
        x = torch.randn(8, NUM_NODES, 1)
        # Condition on some nodes so the (mask-gated) condition token receives gradient.
        cond = torch.tensor([False, False, True, True, True])
        loss = small_net(t, x, torch.arange(NUM_NODES), cond).pow(2).mean() + 1.0
        loss.backward()
        optimizer.step()

        assert torch.equal(small_net.tokenizer.node_embedding.weight, node_embedding_before)
        assert torch.equal(small_net.time_embedding.B, fourier_before)
        assert not torch.equal(small_net.condition_token, condition_before)


class TestSDE:
    """Closed-form checks of the SDE quantities."""

    def test_vesde_closed_forms(self):
        """VE transition variance, diffusion, and weight match the formulas."""
        sde = VESDE(np.zeros(2), np.ones(2), sigma_min=1e-4, sigma_max=15.0)
        t = torch.tensor([1e-5, 0.5, 1.0])
        expected_var = 1e-8 * (15.0 / 1e-4) ** (2 * t.numpy())
        assert np.allclose(sde.transition_var(t).numpy(), expected_var, rtol=1e-5)
        expected_g = 1e-4 * (15.0 / 1e-4) ** t.numpy() * np.sqrt(2 * np.log(15.0 / 1e-4))
        assert np.allclose(sde.diffusion(t).numpy(), expected_g, rtol=1e-5)
        assert np.allclose(sde.weight(t).numpy(), expected_g**2, rtol=1e-5)
        assert np.allclose(sde.mean_scale(t).numpy(), 1.0)
        # Data-marginal std at T_max: sqrt(var_0 + sigma_max^2).
        assert np.allclose(sde.marginal_std_end().numpy(), np.sqrt(1 + 225.0), rtol=1e-5)

    def test_vpsde_closed_forms(self):
        """VP mean scale, variance, drift, and weight match the formulas."""
        sde = VPSDE(np.zeros(2), np.ones(2), beta_min=0.01, beta_max=10.0)
        t = torch.tensor([1e-5, 0.5, 1.0])
        phi = np.exp(-0.25 * t.numpy() ** 2 * (10.0 - 0.01) - 0.5 * t.numpy() * 0.01)
        assert np.allclose(sde.mean_scale(t).numpy(), phi, rtol=1e-5)
        assert np.allclose(sde.transition_var(t).numpy(), 1 - phi**2, rtol=1e-4)
        assert np.allclose(sde.weight(t).numpy(), np.clip(1 - phi**2, 1e-4, None), rtol=1e-4)
        x = torch.ones(2)
        beta_half = 0.01 + 0.5 * (10.0 - 0.01)
        assert np.allclose(sde.drift(torch.tensor(0.5), x).numpy(), -0.5 * beta_half, rtol=1e-5)
        # Unit data variance keeps the marginal variance at one.
        assert np.allclose(sde.marginal_var_end().numpy(), 1.0, rtol=1e-5)


class TestMasks:
    """Condition- and base-mask behaviour."""

    def test_structured_random_masks(self):
        """Structured masks have the right shape, never all-True, and hit all types."""
        gen = torch.Generator().manual_seed(0)
        masks = structured_random_condition_mask(2000, THETA_DIM, X_DIM, generator=gen)
        assert masks.shape == (2000, NUM_NODES)
        assert not masks.all(dim=-1).any()

        posterior_row = torch.tensor([False] * THETA_DIM + [True] * X_DIM)
        likelihood_row = torch.tensor([True] * THETA_DIM + [False] * X_DIM)
        n_joint = (~masks.any(dim=-1)).sum()
        n_posterior = (masks == posterior_row).all(dim=-1).sum()
        n_likelihood = (masks == likelihood_row).all(dim=-1).sum()
        assert n_joint > 100
        assert n_posterior > 100
        assert n_likelihood > 100

    def test_build_base_mask(self):
        """'full' is dense (None), 'directed' has the expected block structure."""
        assert build_base_mask("full", THETA_DIM, X_DIM) is None
        assert build_base_mask(None, THETA_DIM, X_DIM) is None

        mask = build_base_mask("directed", THETA_DIM, X_DIM)
        assert mask.shape == (NUM_NODES, NUM_NODES)
        # Parameters attend only to themselves.
        assert torch.equal(mask[:THETA_DIM, :THETA_DIM], torch.eye(THETA_DIM, dtype=torch.bool))
        # Parameters do not attend to data.
        assert not mask[:THETA_DIM, THETA_DIM:].any()
        # Data attend to all parameters and causally within data.
        assert mask[THETA_DIM:, :THETA_DIM].all()
        assert torch.equal(
            mask[THETA_DIM:, THETA_DIM:],
            torch.tril(torch.ones(X_DIM, X_DIM, dtype=torch.bool)),
        )

        custom = np.eye(NUM_NODES, dtype=bool)
        assert torch.equal(
            build_base_mask(custom, THETA_DIM, X_DIM), torch.eye(NUM_NODES, dtype=torch.bool)
        )
        with pytest.raises(ValueError):
            build_base_mask(np.eye(3, dtype=bool), THETA_DIM, X_DIM)
        with pytest.raises(ValueError):
            build_base_mask("banana", THETA_DIM, X_DIM)

    def test_merge_config_rejects_unknown_keys(self):
        """Config overrides with typos raise instead of being silently ignored."""
        with pytest.raises(ValueError, match="d_model"):
            merge_config(DEFAULT_MODEL_CONFIG, {"d_model": 128}, "model_config")


class TestLoss:
    """Denoising score-matching loss semantics."""

    def test_conditioned_nodes_contribute_zero_loss(self, small_net):
        """Corrupting the prediction at conditioned nodes leaves the loss unchanged."""
        sde = VESDE(np.zeros(NUM_NODES), np.ones(NUM_NODES))
        data = torch.randn(16, NUM_NODES)
        node_ids = torch.arange(NUM_NODES)
        cond = torch.zeros(16, NUM_NODES, dtype=torch.bool)
        cond[:, THETA_DIM:] = True  # condition on all x nodes

        class CorruptedNet(torch.nn.Module):
            """Adds a huge offset to the output at conditioned nodes."""

            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, t, x, node_ids, condition_mask, edge_mask=None):
                out = self.base(t, x, node_ids, condition_mask, edge_mask=edge_mask)
                return out + 1e6 * condition_mask.reshape(out.shape[0], -1, 1).float()

        loss_clean = denoising_score_matching_loss(
            small_net, sde, data, node_ids, cond, generator=torch.Generator().manual_seed(3)
        )
        loss_corrupted = denoising_score_matching_loss(
            CorruptedNet(small_net),
            sde,
            data,
            node_ids,
            cond,
            generator=torch.Generator().manual_seed(3),
        )
        assert torch.allclose(loss_clean, loss_corrupted)


class TestSampling:
    """Sampling semantics on an untrained model."""

    def test_conditioned_values_clamped_throughout(self, random_model):
        """The integrator never moves observed entries away from x_o."""
        cond = torch.tensor([False, False, True, True, True])
        x_o_z = torch.tensor([0.3, -0.2, 1.5])
        x_T = torch.randn(8, NUM_NODES)
        x_T[:, cond] = x_o_z

        seen_states = []

        def recording_score_fn(t, x):
            seen_states.append(x.clone())
            return random_model.score(t, x, cond)

        final = euler_maruyama_reverse(
            recording_score_fn,
            random_model.sde,
            x_T,
            cond,
            num_steps=10,
            generator=torch.Generator().manual_seed(0),
        )
        for state in seen_states + [final]:
            assert torch.allclose(state[:, cond], x_o_z.expand(8, -1))

    def test_sample_shapes_and_units(self, random_model):
        """sample/sample_batched return original-unit latents of the right shape."""
        cond = np.array([False, False, True, True, True])
        gen = torch.Generator().manual_seed(0)
        samples = random_model.sample(
            16, x_o=[1.0, 2.0, 3.0], condition_mask=cond, num_steps=8, generator=gen
        )
        assert samples.shape == (16, 2)
        assert np.isfinite(samples).all()

        batch = random_model.sample_batched(
            8, np.random.randn(5, 3), condition_mask=cond, num_steps=8, batch_size=2
        )
        assert batch.shape == (5, 8, 2)
        assert np.isfinite(batch).all()

    def test_all_conditioned_mask_rejected(self, random_model):
        """A mask conditioning on every node is invalid."""
        with pytest.raises(ValueError):
            random_model.sample(
                4, x_o=np.zeros(NUM_NODES), condition_mask=np.ones(NUM_NODES, dtype=bool)
            )


class TestGuidance:
    """Interval-guidance components."""

    def test_interval_constraint_score_matches_autograd(self):
        """The analytic box-constraint gradient matches autograd of log_step_fn."""
        torch.manual_seed(0)
        x = torch.randn(6, NUM_NODES, requires_grad=True)
        constraint_mask = torch.tensor([True, False, True, False, False])
        a = torch.tensor([-0.5, torch.nan, 0.1, torch.nan, torch.nan])
        b = torch.tensor([0.8, torch.nan, torch.inf, torch.nan, torch.nan])
        scale = torch.tensor(3.0)

        mask_f = constraint_mask.float()
        finite_a, finite_b = torch.isfinite(a), torch.isfinite(b)
        log_p = (
            torch.nn.functional.logsigmoid(scale * (x - torch.nan_to_num(a))) * (mask_f * finite_a)
        ).sum() + (
            torch.nn.functional.logsigmoid(scale * (torch.nan_to_num(b) - x)) * (mask_f * finite_b)
        ).sum()
        (expected,) = torch.autograd.grad(log_p, x)

        actual = interval_constraint_score(x.detach(), scale, constraint_mask, a=a, b=b)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_sample_intervals_shapes(self, random_model):
        """Constrained nodes are returned among the latents."""
        cond = np.array([False, False, True, True, True])
        constraint = np.array([True, False, False, False, False])
        samples = random_model.sample_intervals(
            8,
            x_o=[1.0, 2.0, 3.0],
            condition_mask=cond,
            constraint_mask=constraint,
            a=[-1.0],
            b=[1.0],
            num_steps=8,
            scale_bias=1e-2,
            generator=torch.Generator().manual_seed(0),
        )
        assert samples.shape == (8, 2)
        assert np.isfinite(samples).all()


class TestLogProb:
    """Probability-flow log-probability."""

    def test_exact_and_hutchinson_agree(self, random_model):
        """The two divergence estimators agree within Monte-Carlo tolerance."""
        cond = np.array([False, False, True, True, True])
        theta = np.array([[0.1, 0.2], [0.5, -0.3]], dtype=np.float32)
        x_o = [1.0, 2.0, 3.0]
        lp_exact = random_model.log_prob(theta, x_o, cond, num_steps=25)
        lp_hutch = random_model.log_prob(
            theta,
            x_o,
            cond,
            num_steps=25,
            divergence="hutchinson",
            hutchinson_probes=64,
            generator=torch.Generator().manual_seed(0),
        )
        assert np.all(np.isfinite(lp_exact))
        assert np.allclose(lp_exact, lp_hutch, rtol=0.15, atol=0.3)


class TestSerialization:
    """Save/load round-trips."""

    def test_roundtrip_scores_and_samples(self, random_model, tmp_path):
        """A reloaded model reproduces scores exactly and samples with a shared seed."""
        path = str(tmp_path / "model.pt")
        random_model.save(path)
        reloaded = SimformerModel.load(path)

        cond = torch.tensor([False, False, True, True, True])
        t = torch.tensor(0.5)
        x = torch.randn(4, NUM_NODES)
        with torch.no_grad():
            s1 = random_model.score(t, x, cond)
            s2 = reloaded.score(t, x, cond)
        assert torch.equal(s1, s2)

        samples1 = random_model.sample(
            8,
            x_o=[1.0, 2.0, 3.0],
            condition_mask=cond.numpy(),
            num_steps=8,
            generator=torch.Generator().manual_seed(7),
        )
        samples2 = reloaded.sample(
            8,
            x_o=[1.0, 2.0, 3.0],
            condition_mask=cond.numpy(),
            num_steps=8,
            generator=torch.Generator().manual_seed(7),
        )
        assert np.array_equal(samples1, samples2)
        assert reloaded.meta["param_names"] == ["a", "b"]
        assert np.array_equal(reloaded.z_score_mean, random_model.z_score_mean)


def two_moons_simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Two-moons simulator (Lueckmann et al. 2021 benchmark parametrization).

    Args:
        theta: Parameters of shape ``(N, 2)`` in ``[-1, 1]^2``.
        rng: Numpy random generator.

    Returns:
        Observations of shape ``(N, 2)``.
    """
    n = theta.shape[0]
    alpha = rng.uniform(-np.pi / 2, np.pi / 2, n)
    r = rng.normal(0.1, 0.01, n)
    p = np.stack([r * np.cos(alpha) + 0.25, r * np.sin(alpha)], axis=1)
    shift = np.stack(
        [-np.abs(theta[:, 0] + theta[:, 1]), (-theta[:, 0] + theta[:, 1])], axis=1
    ) / np.sqrt(2)
    return p + shift


@pytest.fixture(scope="module")
def two_moons_model():
    """Train a small Simformer on the two-moons task (shared across tests)."""
    rng = np.random.default_rng(42)
    n_train = 10_000
    theta = rng.uniform(-1, 1, (n_train, 2))
    x = two_moons_simulator(theta, rng)

    model, stats = train_simformer(
        theta,
        x,
        model_config={"num_layers": 3},
        train_config={
            "min_number_steps": 3000,
            "max_number_steps": 3000,
            "training_batch_size": 512,
        },
        seed=0,
        verbose=False,
    )
    test_theta = rng.uniform(-1, 1, (100, 2))
    test_x = two_moons_simulator(test_theta, rng)
    return model, stats, (theta, x), (test_theta, test_x)


class TestTwoMoons:
    """End-to-end validation on the two-moons benchmark."""

    POSTERIOR_MASK = np.array([False, False, True, True])
    LIKELIHOOD_MASK = np.array([True, True, False, False])
    JOINT_MASK = np.array([False, False, False, False])

    def test_training_ran(self, two_moons_model):
        """Training completes and the loss decreases."""
        _, stats, _, _ = two_moons_model
        assert stats["steps_run"] > 0
        assert stats["final_train_loss_ema"] < stats["train_loss"][0]

    def test_posterior_is_bimodal(self, two_moons_model):
        """The posterior at the origin covers both signs of theta_1 + theta_2."""
        model, _, _, _ = two_moons_model
        gen = torch.Generator().manual_seed(0)
        samples = model.sample(
            2000,
            x_o=[0.0, 0.0],
            condition_mask=self.POSTERIOR_MASK,
            num_steps=100,
            generator=gen,
        )
        s = samples[:, 0] + samples[:, 1]
        frac_positive = (s > 0).mean()
        assert 0.1 < frac_positive < 0.9, f"Posterior lost a mode: {frac_positive:.2f} positive"

    def test_posterior_calibration_tarp(self, two_moons_model):
        """TARP expected coverage stays close to nominal at mid-credibility."""
        import tarp

        model, _, _, (test_theta, test_x) = two_moons_model
        gen = torch.Generator().manual_seed(1)
        samples = model.sample_batched(
            250,
            test_x,
            condition_mask=self.POSTERIOR_MASK,
            num_steps=100,
            batch_size=25,
            generator=gen,
        )
        # tarp expects (num_samples, num_sims, num_dims).
        ecp, alpha = tarp.get_tarp_coverage(samples.transpose(1, 0, 2), test_theta, norm=True)
        mid = np.argmin(np.abs(alpha - 0.5))
        assert abs(ecp[mid] - 0.5) < 0.15

    def test_arbitrary_conditionals(self, two_moons_model):
        """Likelihood and joint conditionals reproduce simulator statistics."""
        model, _, (theta_train, x_train), _ = two_moons_model
        gen = torch.Generator().manual_seed(2)
        rng = np.random.default_rng(3)

        theta_o = np.array([0.4, -0.3])
        x_model = model.sample(
            2000,
            x_o=theta_o,
            condition_mask=self.LIKELIHOOD_MASK,
            num_steps=100,
            generator=gen,
        )
        x_true = two_moons_simulator(np.tile(theta_o, (2000, 1)), rng)
        assert np.allclose(x_model.mean(axis=0), x_true.mean(axis=0), atol=0.1)
        assert np.allclose(x_model.std(axis=0), x_true.std(axis=0), atol=0.1)

        joint = model.sample(
            2000, x_o=[], condition_mask=self.JOINT_MASK, num_steps=100, generator=gen
        )
        train_data = np.hstack([theta_train, x_train])
        assert np.allclose(joint.mean(axis=0), train_data.mean(axis=0), atol=0.15)
        assert np.allclose(joint.std(axis=0), train_data.std(axis=0), atol=0.15)

    def test_interval_guidance_respects_box(self, two_moons_model):
        """Interval-constrained samples stay inside the box while conditioning on x."""
        model, _, _, _ = two_moons_model
        gen = torch.Generator().manual_seed(4)
        constraint = np.array([True, False, False, False])
        samples = model.sample_intervals(
            1000,
            x_o=[0.0, 0.0],
            condition_mask=self.POSTERIOR_MASK,
            constraint_mask=constraint,
            a=[0.0],
            b=[0.6],
            scale_bias=1e-2,
            num_steps=100,
            generator=gen,
        )
        inside = ((samples[:, 0] >= 0.0) & (samples[:, 0] <= 0.6)).mean()
        assert inside > 0.95

    def test_log_prob_beats_prior(self, two_moons_model):
        """Mean posterior log-prob of the true parameters beats the prior density."""
        model, _, _, (test_theta, test_x) = two_moons_model
        log_probs = []
        for i in range(10):
            lp = model.log_prob(test_theta[i], test_x[i], self.POSTERIOR_MASK, num_steps=100)
            log_probs.append(float(lp))
        prior_log_prob = -np.log(4.0)  # U(-1,1)^2
        assert np.isfinite(log_probs).all()
        assert np.mean(log_probs) > prior_log_prob


@pytest.fixture(scope="module")
def trained_fitter(tmp_path_factory):
    """Train a small Simformer_Fitter on the galaxy test library (shared)."""
    from synference import Simformer_Fitter
    from synference.utils import test_data_dir

    out_dir = tmp_path_factory.mktemp("simformer_models")
    fitter = Simformer_Fitter.init_from_hdf5(
        model_name="test_simformer",
        hdf5_path=f"{test_data_dir}/sbi_test_library.hdf5",
    )
    fitter.create_feature_array_from_raw_photometry()

    model, stats = fitter.run_single_sbi(
        name_append="pytest",
        out_dir=str(out_dir),
        random_seed=0,
        load_existing_model=False,
        model_config_dict_overrides={"num_layers": 2},
        train_config_dict_overrides={
            "min_number_steps": 600,
            "max_number_steps": 600,
            "training_batch_size": 128,
        },
        num_posterior_draws_per_sample=100,
        evaluate_model=True,
        verbose=False,
    )
    return fitter, model, stats, out_dir


class TestSimformerFitterLibrary:
    """End-to-end Simformer_Fitter run on the galaxy test library."""

    def test_artifacts_written(self, trained_fitter):
        """Training writes the posterior, params, and metrics files plus plots."""
        fitter, model, stats, out_dir = trained_fitter
        model_dir = out_dir / "test_simformer"
        assert (model_dir / "test_simformer_pytest_posterior.pkl").exists()
        assert (model_dir / "test_simformer_pytest_params.pkl").exists()
        assert (model_dir / "test_simformer_pytest_metrics.json").exists()
        plots = list((model_dir / "plots" / "pytest").glob("*"))
        assert len(plots) >= 2
        assert 0 < stats["steps_run"] <= 600

    def test_sample_posterior_contract(self, trained_fitter):
        """sample_posterior returns finite samples with the documented shapes."""
        fitter, _, _, _ = trained_fitter
        n_theta = len(fitter.fitted_parameter_names)

        multi = fitter.sample_posterior(fitter._X_test[:3], num_samples=50, num_steps=50)
        assert multi.shape == (3, 50, n_theta)
        assert np.isfinite(multi).all()

        single = fitter.sample_posterior(fitter._X_test[0], num_samples=50, num_steps=50)
        assert single.shape == (50, n_theta)

    def test_log_prob_paired_contract(self, trained_fitter):
        """log_prob(X, y) pairs rows and returns one finite value per observation."""
        fitter, _, _, _ = trained_fitter
        lp = fitter.log_prob(fitter._X_test[:3], fitter._y_test[:3], num_steps=25)
        assert lp.shape == (3,)
        assert np.isfinite(lp).all()

    def test_load_saved_model_roundtrip(self, trained_fitter):
        """A model reloaded via load_saved_model reproduces seeded samples."""
        from synference import Simformer_Fitter
        from synference.utils import test_data_dir

        fitter, _, _, out_dir = trained_fitter
        reloaded = Simformer_Fitter.load_saved_model(
            model_name="test_simformer_pytest",
            library_path=f"{test_data_dir}/sbi_test_library.hdf5",
            model_file=str(out_dir / "test_simformer"),
        )
        assert reloaded.posteriors is not None
        assert list(reloaded.posteriors.meta["param_names"]) == list(fitter.fitted_parameter_names)
        x_obs = fitter._X_test[:1]
        s1 = fitter.sample_posterior(x_obs, num_samples=20, num_steps=25, rng_seed=11)
        s2 = reloaded.sample_posterior(x_obs, num_samples=20, num_steps=25, rng_seed=11)
        assert np.allclose(s1, s2)

    def test_load_in_fresh_process(self, trained_fitter):
        """A saved model loads and samples in a brand-new Python interpreter."""
        import subprocess
        import sys

        _, _, _, out_dir = trained_fitter
        model_path = out_dir / "test_simformer" / "test_simformer_pytest_posterior.pkl"
        script = (
            "import numpy as np\n"
            "from synference.simformer import SimformerModel\n"
            f"model = SimformerModel.load({str(model_path)!r})\n"
            "mask = np.array([False] * model.theta_dim + [True] * model.x_dim)\n"
            "x_o = np.zeros(model.x_dim) + 25.0\n"
            "s = model.sample(10, x_o=x_o, condition_mask=mask, num_steps=10)\n"
            "assert s.shape == (10, model.theta_dim)\n"
            "assert np.isfinite(s).all()\n"
            "print('FRESH_LOAD_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
        )
        assert "FRESH_LOAD_OK" in result.stdout, result.stderr

    def test_fit_catalogue_smoke(self, trained_fitter):
        """fit_catalogue produces quantile columns for a small catalogue."""
        from astropy.table import Table

        fitter, _, _, _ = trained_fitter
        table = Table()
        for i, name in enumerate(fitter.feature_names):
            table[name] = np.asarray(fitter._X_test[:3, i])

        result = fitter.fit_catalogue(
            table,
            columns_to_feature_names={name: name for name in fitter.feature_names},
            flux_units="AB",
            num_samples=50,
            check_out_of_distribution=False,
        )
        for param in fitter.simple_fitted_parameter_names:
            assert f"{param}_50" in result.colnames
            assert np.isfinite(result[f"{param}_50"]).all()


class TestUniformBoxPrior:
    """Box prior behaviour."""

    def test_sample_and_log_prob(self):
        """Samples stay inside the box; log_prob is -log(volume) inside, -inf outside."""
        prior = UniformBoxPrior({"a": (0.0, 2.0), "b": (-1.0, 1.0)}, ["a", "b"])
        samples = prior.sample(100, generator=torch.Generator().manual_seed(0))
        assert samples.shape == (100, 2)
        assert (samples[:, 0] >= 0).all() and (samples[:, 0] <= 2).all()
        assert (samples[:, 1] >= -1).all() and (samples[:, 1] <= 1).all()

        lp = prior.log_prob(torch.tensor([[1.0, 0.0], [3.0, 0.0]]))
        assert torch.isclose(lp[0], torch.tensor(-np.log(4.0).astype(np.float32)))
        assert torch.isinf(lp[1]) and lp[1] < 0
