"""Two-moons demo of the native PyTorch Simformer.

Trains a Simformer on the classic two-moons SBI benchmark and demonstrates:
posterior sampling (bimodal), likelihood and joint conditionals, interval-constrained
sampling via guidance, log probabilities, and save/load round-tripping.

Run with: python two_moons_demo.py [--quick]
"""

import argparse
import os

import corner
import matplotlib.pyplot as plt
import numpy as np
import torch

from synference.simformer import SimformerModel, train_simformer


def two_moons_simulator(theta, rng):
    """Two-moons simulator (Lueckmann et al. 2021 parametrization)."""
    n = theta.shape[0]
    alpha = rng.uniform(-np.pi / 2, np.pi / 2, n)
    r = rng.normal(0.1, 0.01, n)
    p = np.stack([r * np.cos(alpha) + 0.25, r * np.sin(alpha)], axis=1)
    shift = np.stack(
        [-np.abs(theta[:, 0] + theta[:, 1]), (-theta[:, 0] + theta[:, 1])], axis=1
    ) / np.sqrt(2)
    return p + shift


def main():
    """Train and exercise a Simformer on the two-moons task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small training budget.")
    parser.add_argument("--out-dir", default="two_moons_output", help="Output directory.")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    n_train = 10_000 if args.quick else 100_000
    steps = 3_000 if args.quick else 30_000

    theta = rng.uniform(-1, 1, (n_train, 2))
    x = two_moons_simulator(theta, rng)

    model, stats = train_simformer(
        theta,
        x,
        train_config={
            "min_number_steps": steps,
            "max_number_steps": steps,
            "training_batch_size": 512 if args.quick else 1000,
        },
        seed=0,
    )
    print(f"Trained for {stats['steps_run']} steps in {stats['training_time_s']:.0f} s.")

    model_path = os.path.join(args.out_dir, "two_moons_simformer.pt")
    model.save(model_path)
    model = SimformerModel.load(model_path)  # prove the round trip
    print(f"Model saved to and reloaded from {model_path}.")

    gen = torch.Generator().manual_seed(1)
    posterior_mask = np.array([False, False, True, True])
    x_o = [0.0, 0.0]

    # 1. Posterior at the origin: two crescent-shaped modes.
    samples = model.sample(
        20_000, x_o=x_o, condition_mask=posterior_mask, num_steps=200, generator=gen
    )
    fig = corner.corner(
        samples,
        labels=[r"$\theta_1$", r"$\theta_2$"],
        bins=80,
        plot_datapoints=False,
        plot_density=True,
    )
    fig.suptitle("Two-moons posterior at x = (0, 0)")
    fig.savefig(os.path.join(args.out_dir, "posterior_corner.png"), dpi=150)
    plt.close(fig)

    # 2. Interval-guided posterior: constrain theta_1 to [0, 0.6].
    constraint = np.array([True, False, False, False])
    guided = model.sample_intervals(
        20_000,
        x_o=x_o,
        condition_mask=posterior_mask,
        constraint_mask=constraint,
        a=[0.0],
        b=[0.6],
        scale_bias=1e-2,
        num_steps=200,
        generator=gen,
    )
    inside = ((guided[:, 0] >= 0) & (guided[:, 0] <= 0.6)).mean()
    fig = corner.corner(
        guided,
        labels=[r"$\theta_1$", r"$\theta_2$"],
        bins=80,
        plot_datapoints=False,
        plot_density=True,
    )
    fig.suptitle(rf"Guided posterior, $\theta_1 \in [0, 0.6]$ ({inside:.1%} inside)")
    fig.savefig(os.path.join(args.out_dir, "posterior_interval_corner.png"), dpi=150)
    plt.close(fig)

    # 3. Likelihood conditional x | theta and the joint.
    likelihood_mask = np.array([True, True, False, False])
    x_samples = model.sample(
        5_000, x_o=[0.4, -0.3], condition_mask=likelihood_mask, num_steps=200, generator=gen
    )
    x_true = two_moons_simulator(np.tile([0.4, -0.3], (5_000, 1)), rng)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(*x_true.T, s=2, alpha=0.3, label="Simulator")
    ax.scatter(*x_samples.T, s=2, alpha=0.3, label="Simformer likelihood")
    ax.legend()
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    fig.savefig(os.path.join(args.out_dir, "likelihood_conditional.png"), dpi=150)
    plt.close(fig)

    # 4. Log probabilities of the true parameters for a few test pairs.
    test_theta = rng.uniform(-1, 1, (5, 2))
    test_x = two_moons_simulator(test_theta, rng)
    for i in range(5):
        lp = model.log_prob(test_theta[i], test_x[i], posterior_mask, num_steps=100)
        print(f"log p(theta_true | x_{i}) = {float(lp):.2f} (prior: {-np.log(4):.2f})")

    print(f"Plots written to {args.out_dir}/.")


if __name__ == "__main__":
    main()
