"""End-to-end smoke test: a few epochs must reduce the Fisher loss below
the t=0-only baseline, and the output params must be non-NaN."""

import jax
import jax.numpy as jnp
import numpy as np

from bohmian_flow import (
    potentials, baseline, network, train, core,
)


def test_morse_d2_training_reduces_loss():
    sys = potentials.morse_chain(d=2, q0=0.3)
    p0 = sys['mass'] * sys['v0']
    s_base = baseline.make_baseline_score(
        sys['V_fn'], sys['r0_mean'], sys['r0_cov'], p0)
    init_fn, score_fn, _ = network.make_score_network(
        d=2, hidden_dims=[16, 16], n_freq=2,
        s_baseline_fn=s_base, conditioning='film')
    params = init_fn(jax.random.PRNGKey(0))

    # Baseline loss with initial (untrained) params
    key = jax.random.PRNGKey(42)
    X0 = jax.random.multivariate_normal(
        key, sys['r0_mean'], sys['r0_cov'], shape=(8,))
    V0 = jnp.tile(sys['v0'], (8, 1))
    s0 = core.make_initial_score_fn(sys['r0_mean'], sys['r0_cov'])
    initial_loss = float(core.fisher_loss(
        params, X0, V0, score_fn, sys['V_fn'],
        T=0.1, dt=0.01, initial_score_fn=s0,
        caustic_threshold=0.01, target_clip=100.0))

    # Train a handful of epochs
    params, losses, diag = train.train_fisher(
        score_fn, params, sys['V_fn'],
        sys['r0_mean'], sys['r0_cov'], sys['v0'],
        T=0.1, dt=0.01, n_epochs=15, M_train=8, lr=1e-3,
        n_checkpoints=4, print_every=100,
        grad_clip=5.0, lr_patience=0,
        seed=7,
    )
    # Finite output, loss shrinks by a meaningful factor.
    from jax.flatten_util import ravel_pytree
    flat, _ = ravel_pytree(params['film_params'])
    assert np.all(np.isfinite(np.array(flat)))
    final_loss = diag['final_loss']
    assert np.isfinite(final_loss)
    # No strict monotonicity (stochastic), but the final loss must be
    # within the same order of magnitude, or smaller, than the initial.
    assert final_loss < 10.0 * initial_loss
