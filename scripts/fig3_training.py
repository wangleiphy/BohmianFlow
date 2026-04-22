"""Figure 3: training convergence on the d=4 Morse chain.

Two-panel figure with raw per-epoch traces on log scale:
    (a) Fisher loss vs epoch,
    (b) |<E> - E_exact| vs epoch.

Input is an NPZ produced by ``scripts/extract_training_log.py`` with fields
``epochs``, ``losses``, ``e_mean``.  The exact energy is taken from the
FFT reference NPZ (``E_exact`` field), e.g. ``data/fft_ref_morse_d4.npz``.

Example:
    python scripts/fig3_training.py --data data/fig3_training.npz \\
        --fft-ref data/fft_ref_morse_d4.npz \\
        --output figures/fig3_training.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True,
                   help='NPZ from extract_training_log.py')
    p.add_argument('--fft-ref', default=None,
                   help='FFT reference NPZ (for E_exact); alternatively, '
                        'pass --E-exact.')
    p.add_argument('--E-exact', type=float, default=None)
    p.add_argument('-o', '--output', default='figures/fig3_training.png')
    args = p.parse_args()

    if args.E_exact is None and args.fft_ref is None:
        p.error('Provide --fft-ref or --E-exact')

    d = np.load(args.data)
    epochs = d['epochs']
    losses = d['losses']
    e_mean = d['e_mean']

    if args.E_exact is not None:
        E_exact = args.E_exact
    else:
        E_exact = float(np.load(args.fft_ref)['E_exact'])
    err = np.abs(e_mean - E_exact)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 15, 'axes.labelsize': 17,
        'legend.fontsize': 13,
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'font.family': 'serif', 'mathtext.fontset': 'cm',
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.4))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.94,
                        wspace=0.32)

    c = '#2166ac'
    ax1.semilogy(epochs, losses, '-', color=c, lw=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Fisher loss')
    ax1.text(0.04, 0.94, '(a)', transform=ax1.transAxes,
             fontsize=17, fontweight='bold', va='top')
    ax1.grid(True, alpha=0.2)

    ax2.semilogy(epochs, err, '-', color=c, lw=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(r'$|\langle E\rangle - E_{\mathrm{exact}}|$')
    ax2.text(0.04, 0.94, '(b)', transform=ax2.transAxes,
             fontsize=17, fontweight='bold', va='top')
    ax2.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
