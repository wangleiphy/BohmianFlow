"""evaluate_trajectories: shapes + basic sanity."""

import jax
import jax.numpy as jnp
import numpy as np

from bohmian_score_matching import evaluate, network, potentials


def test_evaluate_shapes():
    sys = potentials.morse_chain(d=2, q0=0.3)
    init_fn, score_fn, _ = network.make_score_network(
        d=2, hidden_dims=[8, 8], n_freq=1,
        s_baseline_fn=lambda x, t: jnp.zeros(2), conditioning='concat')
    params = init_fn(jax.random.PRNGKey(0))

    key = jax.random.PRNGKey(1)
    X0, V0 = evaluate.sample_initial_conditions(
        key, 6, sys['r0_mean'], sys['r0_cov'], sys['v0'])
    out = evaluate.evaluate_trajectories(
        score_fn, params, sys['V_fn'], X0, V0,
        T=0.05, dt=0.01, n_checkpoints=3)
    assert out['mean_x'].shape == (3, 2)
    assert out['sigma'].shape == (3, 2)
    assert out['energies'].shape == (6, 3)
    assert out['t'].shape == (3,)
