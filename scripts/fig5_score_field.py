"""Figure 5: learned score field s_theta(x, t) on a 2D slice of the 4D Morse chain.

Each panel shows the exact slice density |psi(x0, x1, 0, 0)|^2 from the FFT
reference (log-scale colormap) overlaid with streamlines of the learned
score field on the same (x0, x1) plane (non-plotted dims held at 0).

Two-step workflow:
  1. Load the trained checkpoint and the FFT psi snapshots, evaluate
     s_theta on a 2D grid at each requested time, save to an NPZ.
  2. Plot from the NPZ.

Examples:
    # Compute (requires FFT snapshots from compute_fft_reference.py --save-psi-at):
    python scripts/fig5_score_field.py \\
        --checkpoint checkpoints/morse_d4.pkl \\
        --psi-pkl data/fft_psi_snapshots.pkl \\
        --times 0.0 3.14 \\
        --output figures/fig5_score_field.png

    # Replot from cached NPZ:
    python scripts/fig5_score_field.py \\
        --data data/fig5_score_data.npz \\
        --output figures/fig5_score_field.png
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def compute_data(checkpoint_path, psi_pkl, times, dims, data_out):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    from bohmian_flow.potentials import morse_chain
    from bohmian_flow.network import make_score_network
    from bohmian_flow.baseline import make_baseline_score
    from bohmian_flow.checkpoint import load_checkpoint

    ckpt = load_checkpoint(checkpoint_path)
    ck_args = ckpt.get('args', {}) or {}
    params = ckpt['params']

    # Reconstruct the score network matching the checkpoint.
    d = int(ck_args.get('d', 4))
    lam = float(ck_args.get('lam', 0.3))
    hidden_dims = list(ck_args.get('hidden_dims', [128, 128]))
    n_freq = int(ck_args.get('n_freq', 4))
    conditioning = ck_args.get('conditioning', 'film')

    system = morse_chain(d=d, lam=lam)
    p0 = system['mass'] * system['v0']
    s_base = make_baseline_score(system['V_fn'], system['r0_mean'],
                                 system['r0_cov'], p0)
    _, score_fn, _ = make_score_network(
        d, hidden_dims, n_freq, s_base, conditioning=conditioning)

    with open(psi_pkl, 'rb') as f:
        psi_data = pickle.load(f)
    grids = psi_data['grids']
    snapshots = psi_data['psi_snapshots']   # list of (psi, t)
    saved_t = [t for _, t in snapshots]

    di, dj = dims
    gi, gj = grids[di], grids[dj]
    n_i, n_j = len(gi), len(gj)

    snap_out = {}
    x_center = np.zeros(d)    # non-plotted dims held at 0
    for t_req in times:
        idx = int(np.argmin([abs(t_req - t) for t in saved_t]))
        psi_nd, t_val = snapshots[idx]
        # Slice density at x2 = x3 = 0 (nearest grid index).
        idx_slice = [slice(None)] * d
        for k in range(d):
            if k not in (di, dj):
                idx_slice[k] = int(np.argmin(np.abs(grids[k])))
        rho = np.abs(psi_nd[tuple(idx_slice)]) ** 2

        # Evaluate learned score on the 2D grid (non-plotted dims = 0).
        points = np.tile(x_center, (n_i * n_j, 1))
        Gi, Gj = np.meshgrid(gi, gj, indexing='ij')
        points[:, di] = Gi.ravel()
        points[:, dj] = Gj.ravel()
        print(f'  evaluating learned score at t={t_val:.3f} '
              f'({n_i}x{n_j} grid)')
        S = np.array(jax.vmap(lambda x: score_fn(params, x,
                                                  jnp.float64(t_val)))(
            jnp.array(points)))
        s_i = S[:, di].reshape(n_i, n_j)
        s_j = S[:, dj].reshape(n_i, n_j)
        snap_out[t_val] = (rho, s_i, s_j)

    os.makedirs(os.path.dirname(data_out) or '.', exist_ok=True)
    payload = {
        'gi': np.array(gi), 'gj': np.array(gj),
        'di': di, 'dj': dj, 'd': d,
        'n_times': len(snap_out),
    }
    for k, (t_val, (rho, s_i, s_j)) in enumerate(sorted(snap_out.items())):
        payload[f't_{k}'] = t_val
        payload[f'rho_{k}'] = rho
        payload[f's_i_{k}'] = s_i
        payload[f's_j_{k}'] = s_j
    np.savez(data_out, **payload)
    print(f'Saved score-field data to {data_out}')


def _zoom(rho, gi, gj, frac=1e-2, pad=5):
    r_max = rho.max()
    present = rho > r_max * frac
    ri = np.where(np.any(present, axis=1))[0]
    ci = np.where(np.any(present, axis=0))[0]
    return (max(ri[0] - pad, 0), min(ri[-1] + pad + 1, len(gi)),
            max(ci[0] - pad, 0), min(ci[-1] + pad + 1, len(gj)))


def plot(npz_path, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    d = np.load(npz_path, allow_pickle=True)
    gi = d['gi']
    gj = d['gj']
    di = int(d['di'])
    dj = int(d['dj'])
    n_times = int(d['n_times'])

    blues = LinearSegmentedColormap.from_list('custom_blues', [
        (1.0, 1.0, 1.0), (0.75, 0.88, 1.0),
        (0.35, 0.55, 0.9), (0.1, 0.2, 0.6),
    ])

    # Shared zoom window covering all snapshots.
    i0, i1, j0, j1 = len(gi), 0, len(gj), 0
    for k in range(n_times):
        a, b, c, e = _zoom(d[f'rho_{k}'], gi, gj)
        i0, i1 = min(i0, a), max(i1, b)
        j0, j1 = min(j0, c), max(j1, e)
    gi_z = gi[i0:i1]
    gj_z = gj[j0:j1]

    fig, axes = plt.subplots(1, n_times,
                             figsize=(3.6 * n_times + 0.6, 3.4))
    if n_times == 1:
        axes = [axes]
    labels = 'abcdef'
    im = None
    for k in range(n_times):
        rho_z = d[f'rho_{k}'][i0:i1, j0:j1]
        si_z = d[f's_i_{k}'][i0:i1, j0:j1]
        sj_z = d[f's_j_{k}'][i0:i1, j0:j1]
        t_val = float(d[f't_{k}'])

        rho_rel = rho_z / rho_z.max()
        score_mask = rho_rel > 5e-3
        si_m = np.where(score_mask, si_z, 0.0)
        sj_m = np.where(score_mask, sj_z, 0.0)

        log_rho = np.log10(np.maximum(rho_z, rho_z.max() * 1e-6) / rho_z.max())
        extent = [gi_z[0], gi_z[-1], gj_z[0], gj_z[-1]]
        im = axes[k].imshow(log_rho.T, origin='lower', extent=extent,
                            cmap=blues, vmin=-4, vmax=0,
                            interpolation='bicubic',
                            aspect='equal', rasterized=True)

        speed = np.sqrt(si_m ** 2 + sj_m ** 2)
        p95 = max(np.percentile(speed[score_mask], 95), 1e-6)
        lw = np.clip(0.4 + 0.8 * speed.T / p95, 0.3, 1.2)
        axes[k].streamplot(gi_z, gj_z, si_m.T, sj_m.T,
                           color='0.1', linewidth=lw,
                           density=1.8, arrowsize=0.6, arrowstyle='->')
        axes[k].set_xlabel(f'$x_{di}$', fontsize=17)
        axes[k].set_ylabel(f'$x_{dj}$', fontsize=17)
        axes[k].set_title(f'({labels[k]})  $t = {t_val:.2f}$', fontsize=15)
        axes[k].tick_params(labelsize=13)
        if k > 0:
            axes[k].set_ylabel('')

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r'$\log_{10}(\rho / \rho_{\max})$', fontsize=15)
    cb.ax.tick_params(labelsize=13)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved figure to {output_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--psi-pkl', default=None,
                   help='Pickle of FFT psi snapshots from '
                        'compute_fft_reference.py --save-psi-at T0 T1 ...')
    p.add_argument('--data', default=None, help='Cached NPZ')
    p.add_argument('--data-out', default='data/fig5_score_data.npz')
    p.add_argument('--times', type=float, nargs='+', default=[0.0, np.pi])
    p.add_argument('--dims', type=int, nargs=2, default=[0, 1])
    p.add_argument('--output', default='figures/fig5_score_field.png')
    args = p.parse_args()

    if args.data is None:
        if args.checkpoint is None or args.psi_pkl is None:
            p.error('--checkpoint and --psi-pkl are required when '
                    '--data is omitted')
        compute_data(args.checkpoint, args.psi_pkl, args.times,
                     tuple(args.dims), args.data_out)
        plot(args.data_out, args.output)
    else:
        plot(args.data, args.output)


if __name__ == '__main__':
    main()
