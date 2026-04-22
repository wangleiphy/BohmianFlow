"""Checkpoint I/O: pickle-based save/load of params + optimiser state."""

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np


def save_checkpoint(path, params, opt_state, losses, epoch, args=None):
    """Save a training checkpoint.

    Args:
        path: destination file path (.pkl).
        params: full params dict (converted to numpy before pickling).
        opt_state: optax optimiser state (also converted to numpy).
        losses: sequence of floats.
        epoch: epoch index at which the checkpoint was taken.
        args: optional argparse.Namespace, saved for provenance.
    """
    def to_numpy(x):
        return np.array(x) if isinstance(x, jnp.ndarray) else x

    payload = {
        'params': jax.tree.map(to_numpy, params),
        'opt_state': jax.tree.map(to_numpy, opt_state),
        'losses': [float(l) for l in losses],
        'epoch': epoch,
        'args': vars(args) if args is not None else None,
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"  Checkpoint saved: {path} (epoch {epoch})")


def load_checkpoint(path):
    """Load a checkpoint saved by :func:`save_checkpoint`."""
    def to_jax(x):
        return jnp.array(x) if isinstance(x, np.ndarray) else x

    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    ckpt['params'] = jax.tree.map(to_jax, ckpt['params'])
    ckpt['opt_state'] = jax.tree.map(to_jax, ckpt['opt_state'])
    print(f"  Checkpoint loaded: {path} (epoch {ckpt['epoch']}, "
          f"{len(ckpt['losses'])} losses)")
    return ckpt
