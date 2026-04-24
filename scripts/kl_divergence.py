"""Reverse KL divergence of the learned density against FFT reference.

For each time t in a supplied FFT snapshot pickle, evaluate

    KL(rho_theta || rho_exact)(t)
      ~= M^{-1} sum_i [log rho_theta(x_i(t), t)
                       - log rho_exact(x_i(t), t)]

where the learned density is obtained from the change of variables

    log rho_theta(x_i(t), t) = log rho_0(x_i(0)) - log |det F_i(t)|,

and rho_exact = |psi|^2 is multilinearly interpolated from the FFT grid at
the propagated particle positions.

Example:

    python scripts/kl_divergence.py \
        --checkpoint checkpoints/morse_d4.pkl \
        --psi-pkl data/fft_psi_snapshots.pkl \
        --M 20000 \
        --data-out data/kl_morse_d4.npz

The snapshot pickle is produced by
``scripts/compute_fft_reference.py --save-psi-at ...``.
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def interp_linear_nd(grids, field, pts):
    """Multilinear interpolation on a regular d-dimensional grid.

    Points outside the grid are returned as NaN so callers can exclude them
    explicitly instead of silently clipping density tails.
    """
    d = len(grids)
    M = pts.shape[0]
    axes = [np.asarray(g) for g in grids]
    dx = np.array([g[1] - g[0] for g in axes])
    lo = np.array([g[0] for g in axes])
    n = np.array([g.size for g in axes])

    u = (pts - lo[None, :]) / dx[None, :]
    i0 = np.floor(u).astype(np.int64)
    frac = u - i0
    inside = np.all((i0 >= 0) & (i0 < n - 1), axis=1)

    out = np.full(M, np.nan, dtype=np.float64)
    if not inside.any():
        return out

    sel = np.where(inside)[0]
    i0s = i0[sel]
    fs = frac[sel]
    acc = np.zeros(sel.size, dtype=np.float64)
    for corner in range(1 << d):
        weight = np.ones(sel.size, dtype=np.float64)
        idx = [None] * d
        for k in range(d):
            bit = (corner >> k) & 1
            weight *= fs[:, k] if bit else (1.0 - fs[:, k])
            idx[k] = i0s[:, k] + bit
        acc += weight * field[tuple(idx)]
    out[sel] = acc
    return out


def compute_kl(checkpoint_path, psi_pkl, M, dt, seed, floor):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    from bohmian_flow.baseline import make_baseline_score
    from bohmian_flow.checkpoint import load_checkpoint
    from bohmian_flow.evaluate import sample_initial_conditions
    from bohmian_flow.network import make_score_network
    from bohmian_flow.potentials import morse_chain
    from bohmian_flow.trajectory import propagate_with_F

    ckpt = load_checkpoint(checkpoint_path)
    ck_args = ckpt.get('args', {}) or {}
    params = ckpt['params']

    d = int(ck_args.get('d', 4))
    lam = float(ck_args.get('lam', 0.3))
    hidden_dims = list(ck_args.get('hidden_dims', [128, 128]))
    n_freq = int(ck_args.get('n_freq', 4))
    conditioning = ck_args.get('conditioning', 'film')

    system = morse_chain(d=d, lam=lam)
    p0 = system['mass'] * system['v0']
    s_base = make_baseline_score(
        system['V_fn'], system['r0_mean'], system['r0_cov'], p0)
    _, score_fn, _ = make_score_network(
        d, hidden_dims, n_freq, s_base, conditioning=conditioning)

    with open(psi_pkl, 'rb') as f:
        psi_data = pickle.load(f)
    snapshots = psi_data['psi_snapshots']
    grids = psi_data['grids']
    times = [float(t) for _, t in snapshots]
    print(f"FFT snapshot times: {times}")

    key = jax.random.PRNGKey(seed)
    r0_mean = jnp.array(system['r0_mean'])
    r0_cov = jnp.array(system['r0_cov'])
    v0 = jnp.array(system['v0'])
    X0, V0 = sample_initial_conditions(key, M, r0_mean, r0_cov, v0)

    T_max = max(times)

    def single_traj(x0, v0_):
        x_traj, _, F_traj, _ = propagate_with_F(
            x0, v0_, params, score_fn, system['V_fn'], T_max, dt,
            system['hbar'], system['mass'])
        return x_traj, F_traj

    print(f"Propagating M={M} particles to T={T_max:.4f} ...")
    x_all, F_all = jax.vmap(single_traj)(X0, V0)
    x_all = np.array(x_all)
    F_all = np.array(F_all)

    mean_np = np.array(r0_mean)
    cov_np = np.array(r0_cov)
    sign, logdet_cov = np.linalg.slogdet(cov_np)
    if sign <= 0:
        raise ValueError("Initial covariance must be positive definite")
    inv_cov = np.linalg.inv(cov_np)
    diff0 = np.array(X0) - mean_np[None, :]
    log_rho_0 = -0.5 * (
        np.einsum('mi,ij,mj->m', diff0, inv_cov, diff0)
        + logdet_cov + d * np.log(2.0 * np.pi)
    )

    print(f"\n{'t':>8} {'KL':>12} {'H(rho_theta)':>14} "
          f"{'H_cross':>12} {'out-of-grid':>12}")
    results = []
    for psi_nd, t_snap in snapshots:
        t_snap = float(t_snap)
        step = min(int(round(t_snap / dt)), x_all.shape[1] - 1)
        x_t = x_all[:, step, :]
        F_t = F_all[:, step, :, :]

        _, logabsdet_F = np.linalg.slogdet(F_t)
        log_rho_theta = log_rho_0 - logabsdet_F

        rho_exact = np.abs(psi_nd) ** 2
        log_rho_exact = interp_linear_nd(
            grids, np.log(rho_exact + floor), x_t)
        valid = ~np.isnan(log_rho_exact)
        n_out = int((~valid).sum())

        kl = float(np.mean(log_rho_theta[valid] - log_rho_exact[valid]))
        H_theta = float(-np.mean(log_rho_theta[valid]))
        H_cross = float(-np.mean(log_rho_exact[valid]))
        print(f"{t_snap:>8.4f} {kl:>12.4f} {H_theta:>14.4f} "
              f"{H_cross:>12.4f} {n_out:>12d}")
        results.append({
            't': t_snap,
            'kl': kl,
            'H_theta': H_theta,
            'H_cross': H_cross,
            'n_out': n_out,
            'M': M,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--psi-pkl', required=True)
    parser.add_argument('--M', type=int, default=20000)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--floor', type=float, default=1e-12,
                        help='Floor on rho_exact before taking log')
    parser.add_argument('--data-out', default=None,
                        help='Optional NPZ output path')
    args = parser.parse_args()

    results = compute_kl(
        args.checkpoint, args.psi_pkl, args.M, args.dt, args.seed,
        args.floor)

    if args.data_out:
        os.makedirs(os.path.dirname(args.data_out) or '.', exist_ok=True)
        keys = ['t', 'kl', 'H_theta', 'H_cross', 'n_out']
        arrs = {k: np.array([r[k] for r in results]) for k in keys}
        arrs['M'] = np.array(results[0]['M'])
        np.savez(args.data_out, **arrs)
        print(f"Saved: {args.data_out}")


if __name__ == '__main__':
    main()
