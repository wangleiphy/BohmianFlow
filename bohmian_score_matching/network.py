"""Score network s_theta(x, t) = s_base(x, t) + grad_x phi_theta(x, t).

The scalar potential phi_theta is an MLP with softplus activations.  Time
enters through a Fourier embedding fourier(t) = [sin(2^k t), cos(2^k t)] that
is either concatenated to x at the input ('concat' conditioning) or used to
modulate each hidden layer via Feature-wise Linear Modulation
(FiLM)~[Perez et al., 2018]:

    h = spatial_layer(h)
    gamma = W_gamma fourier(t) + b_gamma
    beta  = W_beta  fourier(t) + b_beta
    h = (1 + gamma) * softplus(h) + beta

The score is obtained as s_theta = grad_x phi_theta, so it is a gradient
field by construction (curl-free by architecture).  The baseline
``s_base(x, t)`` adds a fixed (non-trainable) Gaussian/harmonic score; set
it to zero for pure MLP models.

The quantum potential helper

    Q(x, t) = -(hbar^2 / 4 m) (div s + 1/2 |s|^2)

shares a single ``jax.linearize`` of s_theta, so the divergence is computed
in the same linearisation as the score.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _init_mlp(dims, key, output_scale=0.01):
    """Xavier init, with a small output layer."""
    params = []
    n_layers = len(dims) - 1
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        key, k = jax.random.split(key)
        scale = output_scale if i == n_layers - 1 else jnp.sqrt(1.0 / d_in)
        W = jax.random.normal(k, (d_in, d_out)) * scale
        b = jnp.zeros(d_out)
        params.append((W, b))
    return params


def _mlp_forward(params, x):
    for W, b in params[:-1]:
        x = jax.nn.softplus(x @ W + b)
    W, b = params[-1]
    out = x @ W + b
    return out[0] if out.shape == (1,) else out


def _fourier_embed(t, freqs):
    if freqs.shape[0] == 0:
        return jnp.array([])
    phases = freqs * t
    return jnp.concatenate([jnp.sin(phases), jnp.cos(phases)])


# --- FiLM MLP --------------------------------------------------------------

def _init_film_mlp(d, hidden_dims, n_time_features, key, output_scale=0.01):
    spatial = []
    film = []
    dims = [d] + list(hidden_dims)
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        key, k = jax.random.split(key)
        spatial.append((
            jax.random.normal(k, (d_in, d_out)) * jnp.sqrt(1.0 / d_in),
            jnp.zeros(d_out),
        ))
        key, k1 = jax.random.split(key)
        key, k2 = jax.random.split(key)
        # Small init so the baseline score dominates at t = 0.
        film.append((
            jax.random.normal(k1, (n_time_features, d_out)) * 0.01,
            jnp.zeros(d_out),
            jax.random.normal(k2, (n_time_features, d_out)) * 0.01,
            jnp.zeros(d_out),
        ))
    key, k = jax.random.split(key)
    d_last = hidden_dims[-1]
    W_out = jax.random.normal(k, (d_last, 1)) * output_scale
    b_out = jnp.zeros(1)
    return {'spatial': spatial, 'film': film, 'output': (W_out, b_out)}


def _film_mlp_forward(params, x, time_embed):
    h = x
    for (W, b), (W_g, b_g, W_b, b_b) in zip(params['spatial'], params['film']):
        h = h @ W + b
        gamma = time_embed @ W_g + b_g
        beta = time_embed @ W_b + b_b
        h = (1.0 + gamma) * jax.nn.softplus(h) + beta
    W_out, b_out = params['output']
    out = h @ W_out + b_out
    return out[0] if out.shape == (1,) else out


def count_params(params):
    """Count trainable MLP parameters in either concat or FiLM form."""
    n = 0
    if 'film_params' in params:
        p = params['film_params']
        for W, b in p['spatial']:
            n += W.size + b.size
        for Wg, bg, Wb, bb in p['film']:
            n += Wg.size + bg.size + Wb.size + bb.size
        W, b = p['output']
        n += W.size + b.size
    else:
        for W, b in params['mlp_params']:
            n += W.size + b.size
    return n


# --- Factory ---------------------------------------------------------------

def make_score_network(d, hidden_dims, n_freq, s_baseline_fn,
                       conditioning='film', output_scale=0.01):
    """Construct a score network s_theta(x, t) = s_base(x, t) + grad_x phi_theta.

    Args:
        d: spatial dimension.
        hidden_dims: list of hidden layer widths for phi_theta.
        n_freq: number of Fourier frequencies for the time embedding; the
            embedding concatenates [sin(2^k t), cos(2^k t)] for k in 0..n_freq-1.
        s_baseline_fn: fixed baseline score (e.g. Gaussian harmonic); signature
            ``(x, t) -> (d,)``.  Pass ``lambda x, t: jnp.zeros(d)`` to disable.
        conditioning: ``'concat'`` (time features concatenated at input) or
            ``'film'`` (per-layer FiLM modulation, recommended).
        output_scale: initialization scale for the output layer, kept small
            so the network starts close to the baseline.

    Returns:
        init_fn(key) -> params,
        score_fn(params, x, t) -> (d,),
        potential_fn(params, x, t) -> scalar phi_theta.
    """
    freqs = (2.0 ** jnp.arange(n_freq, dtype=jnp.float64)
             if n_freq > 0 else jnp.array([], dtype=jnp.float64))
    n_time_features = 2 * n_freq

    if conditioning == 'film':
        def init_fn(key):
            return {
                'film_params': _init_film_mlp(
                    d, hidden_dims, n_time_features, key,
                    output_scale=output_scale),
                'freqs': freqs,
            }

        def potential_fn(params, x, t):
            te = (_fourier_embed(t, params['freqs'])
                  if params['freqs'].shape[0] > 0 else jnp.zeros(0))
            return _film_mlp_forward(params['film_params'], x, te)

    elif conditioning == 'concat':
        mlp_dims = [d + n_time_features] + list(hidden_dims) + [1]

        def init_fn(key):
            return {
                'mlp_params': _init_mlp(mlp_dims, key,
                                        output_scale=output_scale),
                'freqs': freqs,
            }

        def potential_fn(params, x, t):
            te = (_fourier_embed(t, params['freqs'])
                  if params['freqs'].shape[0] > 0 else jnp.zeros(0))
            inp = jnp.concatenate([x, te]) if te.shape[0] > 0 else x
            return _mlp_forward(params['mlp_params'], inp)

    else:
        raise ValueError(f"Unknown conditioning '{conditioning}'")

    def score_fn(params, x, t):
        s_base = s_baseline_fn(x, t)
        s_corr = jax.grad(potential_fn, argnums=1)(params, x, t)
        return s_base + s_corr

    return init_fn, score_fn, potential_fn


# --- Quantum potential -----------------------------------------------------

def _score_and_divergence(score_fn, params, x, t):
    """Return (s, div s) in a single linearisation of s_theta."""
    d = x.shape[0]
    s, pushfwd = jax.linearize(lambda x_: score_fn(params, x_, t), x)
    basis = jnp.eye(d, dtype=x.dtype)
    jac_cols = jax.vmap(pushfwd)(basis)
    return s, jnp.trace(jac_cols)


def quantum_potential(score_fn, params, x, t, hbar=1.0, mass=1.0):
    """Q(x, t) = -(hbar^2 / 4 m) (div s + 1/2 |s|^2) at a single x."""
    s, div_s = _score_and_divergence(score_fn, params, x, t)
    return -(hbar ** 2) / (4.0 * mass) * (div_s + 0.5 * jnp.sum(s ** 2))


def quantum_potential_batch(score_fn, params, X, t, hbar=1.0, mass=1.0):
    """Q at all M particles, X: (M, d) -> (M,)."""
    return jax.vmap(
        lambda x: quantum_potential(score_fn, params, x, t, hbar, mass)
    )(X)
