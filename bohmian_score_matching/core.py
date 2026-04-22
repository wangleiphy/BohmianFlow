"""Fisher divergence loss.

L_Fisher = E_{rho_theta}[|s_theta - grad ln rho_theta|^2]

The target ``grad ln rho_theta`` is computed from the deformation gradient F
along each Bohmian trajectory:

    grad_{x(t)} ln rho_theta(x(t), t)
        = F^{-T} . [ s_0(x(0)) - grad_{x(0)} ln|det F(t)| ].

Differentiating the identity ln rho_theta(x(t), t) = ln rho_0(x(0)) -
ln|det F(t)| with respect to x(0) and transforming to the current frame.

The loss is computed per particle, averaged over time checkpoints, and
averaged over the batch.  Timesteps where |det F| < caustic_threshold are
masked out to avoid divergent targets during transient classical focusing;
the remaining targets are clipped to ``target_clip`` in magnitude as a mild
safeguard.  At convergence, the learned quantum force lifts det F away from
zero and the masked fraction goes to zero; see the PRL paper for the
self-healing analysis.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def make_initial_score_fn(r0_mean, r0_cov):
    """Score s_0(x) = -Sigma_0^{-1} (x - mu_0) of a Gaussian initial state."""
    Sigma_inv = jnp.linalg.inv(r0_cov)

    def s0(x):
        return -Sigma_inv @ (x - r0_mean)

    return s0


def fisher_loss(params, X0, V0, score_fn, V_fn, T, dt, initial_score_fn,
                ckpt_idx=None, hbar=1.0, mass=1.0,
                caustic_threshold=0.1, target_clip=100.0,
                grad_mode='jacfwd'):
    """Compute the Fisher divergence over an ensemble of Bohmian trajectories.

    Args:
        params: full parameter dict (must contain the trainable key and any
            non-trainable entries consumed by ``score_fn``, e.g. ``'freqs'``).
        X0: (M, d) initial positions sampled from rho_0.
        V0: (M, d) initial velocities (``mass * v0 = grad S_0``).
        score_fn: (params, x, t) -> (d,).
        V_fn: (d,) -> scalar classical potential.
        T: final time.
        dt: leapfrog timestep (must divide T).
        initial_score_fn: (x,) -> (d,) exact score at t = 0.
        ckpt_idx: (K,) integer step indices at which the loss is evaluated.
            None = all n_steps+1 steps.  The first entry should be 0 to
            anchor the target with s_0.
        hbar, mass: physical constants.
        caustic_threshold: per-particle mask; a particle is dropped entirely
            if |det F| falls below this value at any checkpoint.
        target_clip: maximum absolute value per component of the score
            target, to soften caustic-neighbourhood spikes.
        grad_mode: ``'jacfwd'`` (d forward passes, best when d < K) or
            ``'jacrev'`` (K backward passes, best when d > K).

    Returns:
        Scalar loss averaged over (particles, non-masked checkpoints).
    """
    from bohmian_score_matching.trajectory import propagate_with_F

    d = X0.shape[1]
    eye_d = jnp.eye(d, dtype=X0.dtype)

    def single_particle_loss(x0, v0):
        def lndetF_of_x0(x0_):
            x_traj, _, F_traj, t_traj = propagate_with_F(
                x0_, v0, params, score_fn, V_fn, T, dt, hbar, mass)
            det_F = jnp.linalg.det(F_traj)
            lndetF = jnp.log(jnp.maximum(jnp.abs(det_F), 1e-30))
            sub = lndetF[ckpt_idx] if ckpt_idx is not None else lndetF
            return sub, (x_traj, F_traj, t_traj, det_F)

        jac_fn = jax.jacrev if grad_mode == 'jacrev' else jax.jacfwd
        grad_lndetF, (x_traj, F_traj, t_traj, det_F) = jac_fn(
            lndetF_of_x0, has_aux=True)(x0)

        if ckpt_idx is not None:
            x_ck = x_traj[ckpt_idx]
            t_ck = t_traj[ckpt_idx]
            F_ck = F_traj[ckpt_idx]
            det_F_ck = det_F[ckpt_idx]
        else:
            x_ck, t_ck, F_ck, det_F_ck = x_traj, t_traj, F_traj, det_F

        s_theta = jax.vmap(lambda x, t: score_fn(params, x, t))(x_ck, t_ck)

        # Score target:  F^{-T} . [s_0 - grad_{x0} ln|det F|]
        rhs = initial_score_fn(x0)[None, :] - grad_lndetF
        F_reg = F_ck + 1e-8 * eye_d[None]
        s_target = jnp.linalg.solve(
            jnp.swapaxes(F_reg, -2, -1), rhs[..., None]
        )[..., 0]
        s_target = jnp.clip(s_target, -target_clip, target_clip)

        # Per-particle caustic mask: drop this particle entirely if any
        # checkpoint has |det F| < threshold (per-timestep masking misses
        # near-threshold points that still produce large F^{-T} solves).
        any_caustic = jnp.any(jnp.abs(det_F_ck) < caustic_threshold)
        mask = jnp.where(any_caustic,
                         jnp.zeros_like(det_F_ck),
                         jnp.ones_like(det_F_ck))

        delta = s_theta - s_target       # (K, d)
        return jnp.sum(delta ** 2, axis=-1), mask   # both (K,)

    sq_err, mask = jax.vmap(single_particle_loss)(X0, V0)  # (M, K)
    per_particle = jnp.sum(sq_err * mask, axis=-1) / jnp.maximum(
        jnp.sum(mask, axis=-1), 1.0)
    return jnp.mean(per_particle)
