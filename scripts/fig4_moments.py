"""Figure 4: per-mode means and widths on the d=4 Morse chain.

Two-panel figure:
    (a) <x_i>(t) for i = 0..3,
    (b) sigma_i(t) for i = 0..3.
Solid lines: FFT reference.  Open circles: learned trajectories from M_test
particles propagated with the trained score network.

Two-step workflow:
  1. Compute learned moments from the checkpoint and cache to an NPZ
     (default M_test = 20000, matching the PRL figure).
  2. Plot learned (markers) vs FFT (lines).

Examples:
    # Compute + plot in one go:
    python scripts/fig4_moments.py \\
        --checkpoint checkpoints/morse_d4.pkl \\
        --fft-ref data/fft_ref_morse_d4.npz \\
        --output figures/fig4_moments.png

    # Replot from cached NPZ:
    python scripts/fig4_moments.py \\
        --data data/fig4_moments.npz \\
        --output figures/fig4_moments.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def compute_moments(checkpoint_path, fft_ref_path, M_test, data_out):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    from bohmian_score_matching.potentials import morse_chain
    from bohmian_score_matching.network import make_score_network
    from bohmian_score_matching.baseline import make_baseline_score
    from bohmian_score_matching.checkpoint import load_checkpoint
    from bohmian_score_matching.evaluate import (
        sample_initial_conditions, evaluate_trajectories,
    )

    ckpt = load_checkpoint(checkpoint_path)
    ck_args = ckpt.get('args', {}) or {}
    params = ckpt['params']
    epoch = ckpt['epoch']

    fft = np.load(fft_ref_path)
    d = int(fft['d'])
    T = float(fft['T'])
    dt = float(fft['dt'])
    morse_kwargs = {'d': d}
    for key, default in [('lam', 0.3), ('D', 12.5), ('beta', 0.2),
                         ('q0', 1.0)]:
        if key in fft.files:
            morse_kwargs[key] = float(fft[key])
    system = morse_chain(**morse_kwargs)

    hidden_dims = tuple(ck_args.get('hidden_dims', [128, 128]))
    n_freq = int(ck_args.get('n_freq', 4))
    conditioning = ck_args.get('conditioning', 'film')
    p0 = system['mass'] * system['v0']
    s_base = make_baseline_score(system['V_fn'], system['r0_mean'],
                                 system['r0_cov'], p0)
    _, score_fn, _ = make_score_network(
        d, list(hidden_dims), n_freq, s_base, conditioning=conditioning)

    n_ck = min(101, int(T / dt) + 1)
    key = jax.random.PRNGKey(999)
    X0, V0 = sample_initial_conditions(
        key, M_test, system['r0_mean'], system['r0_cov'], system['v0'])
    print(f'Evaluating learned trajectories: M={M_test}, T={T:.4f}, n_ck={n_ck}')
    diag = evaluate_trajectories(
        score_fn, params, system['V_fn'], X0, V0, T, dt, n_ck,
        hbar=system['hbar'], mass=system['mass'])

    os.makedirs(os.path.dirname(data_out) or '.', exist_ok=True)
    np.savez(data_out,
             t_learn=np.array(diag['t']),
             learned_means=np.array(diag['mean_x']),
             learned_sigmas=np.array(diag['sigma']),
             fft_t=fft['t'],
             fft_means=fft['means'],
             fft_sigmas=fft['sigmas'],
             d=d, T=T, epoch=epoch,
             checkpoint=os.path.basename(checkpoint_path),
             M_test=M_test)
    print(f'Saved moments to {data_out}')
    return data_out


def plot(npz_path, output_path):
    d = np.load(npz_path)
    t_learn = d['t_learn']
    l_means = d['learned_means']
    l_sigmas = d['learned_sigmas']
    fft_t = d['fft_t']
    fft_means = d['fft_means']
    fft_sigmas = d['fft_sigmas']
    n_modes = int(d['d'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 15, 'axes.labelsize': 17,
        'legend.fontsize': 12,
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'font.family': 'serif', 'mathtext.fontset': 'cm',
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.16, top=0.94,
                        wspace=0.32)
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

    for i in range(n_modes):
        c = colors[i % len(colors)]
        ax1.plot(fft_t, fft_means[:, i], '-', color=c, lw=2.2, zorder=2,
                 label=fr'$i={i}$')
        ax1.plot(t_learn, l_means[:, i], 'o', color=c, ms=4.0,
                 markerfacecolor='none', markeredgewidth=0.9, zorder=3)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel(r'$\langle x_i \rangle$')
    ax1.text(0.04, 0.94, '(a)', transform=ax1.transAxes,
             fontsize=17, fontweight='bold', va='top')
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='best', ncol=2, frameon=False, handlelength=1.4,
               columnspacing=1.0)

    for i in range(n_modes):
        c = colors[i % len(colors)]
        ax2.plot(fft_t, fft_sigmas[:, i], '-', color=c, lw=2.2, zorder=2)
        ax2.plot(t_learn, l_sigmas[:, i], 'o', color=c, ms=4.0,
                 markerfacecolor='none', markeredgewidth=0.9, zorder=3)
    ax2.set_xlabel('$t$')
    ax2.set_ylabel(r'$\sigma_i$')
    ax2.text(0.04, 0.94, '(b)', transform=ax2.transAxes,
             fontsize=17, fontweight='bold', va='top')
    ax2.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved figure to {output_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--fft-ref', default=None)
    p.add_argument('--data', default=None, help='Cached moments NPZ')
    p.add_argument('--data-out', default='data/fig4_moments.npz')
    p.add_argument('--output', default='figures/fig4_moments.png')
    p.add_argument('--M-test', type=int, default=20000)
    args = p.parse_args()

    if args.data is None:
        if args.checkpoint is None or args.fft_ref is None:
            p.error('--checkpoint and --fft-ref are required when --data is omitted')
        compute_moments(args.checkpoint, args.fft_ref, args.M_test, args.data_out)
        plot(args.data_out, args.output)
    else:
        plot(args.data, args.output)


if __name__ == '__main__':
    main()
