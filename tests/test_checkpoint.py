"""Checkpoint round-trip via pickle."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import optax

from bohmian_score_matching import checkpoint


def test_save_load_roundtrip():
    params = {
        'mlp_params': [(jnp.ones((3, 2)), jnp.zeros(2))],
        'freqs': jnp.array([1.0, 2.0]),
    }
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params['mlp_params'])
    losses = [0.5, 0.4, 0.3]
    epoch = 7

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'ckpt.pkl')
        checkpoint.save_checkpoint(path, params, opt_state, losses, epoch)
        ck = checkpoint.load_checkpoint(path)

        np.testing.assert_allclose(
            np.array(ck['params']['mlp_params'][0][0]),
            np.ones((3, 2)))
        assert ck['epoch'] == epoch
        assert ck['losses'] == [0.5, 0.4, 0.3]
