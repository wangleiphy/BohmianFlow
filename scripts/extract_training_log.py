"""Parse Fisher training log(s) into an NPZ for Fig. 3.

Extracts ``(epoch, loss, E_mean)`` triples from each ``Epoch N/T: ...`` line.
Losses are present on every log line; ``E_mean`` only appears on
``print_every`` lines, so we linearly interpolate it to cover the missing
epochs.  Multiple logs can be concatenated in chronological order (for
resume chains).

Example:
    python scripts/extract_training_log.py -o data/fig3_training.npz \\
        logs/run_a.log logs/run_b.log logs/run_c.log
"""

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


EPOCH_RE = re.compile(
    r'^Epoch\s+(\d+)/\d+:\s+loss=([\dEe\-+.]+)'
    r'.*?(?:\|\s+E_mean=([\dEe\-+.]+))?'
)


def parse_log(path):
    """Return (epochs, losses, e_mean) arrays.  ``e_mean`` is NaN where the
    log line had no diagnostics block."""
    epochs, losses, e_means = [], [], []
    with open(path) as f:
        for line in f:
            m = EPOCH_RE.match(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            e_means.append(float(m.group(3)) if m.group(3) else np.nan)
    return np.array(epochs), np.array(losses), np.array(e_means)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+')
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    all_e, all_l, all_em = [], [], []
    for path in args.logs:
        e, l, em = parse_log(path)
        if len(e) == 0:
            print(f'WARNING: no epochs parsed from {path}')
            continue
        print(f'{path}: {len(e)} epochs, {e[0]}..{e[-1]}')
        all_e.append(e)
        all_l.append(l)
        all_em.append(em)

    epochs = np.concatenate(all_e) if all_e else np.array([])
    losses = np.concatenate(all_l) if all_l else np.array([])
    e_mean = np.concatenate(all_em) if all_em else np.array([])

    # Interpolate E_mean over epochs without it.
    if len(e_mean) and np.any(~np.isnan(e_mean)):
        mask = ~np.isnan(e_mean)
        e_mean = np.interp(epochs, epochs[mask], e_mean[mask])

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez(args.output,
             epochs=epochs, losses=losses, e_mean=e_mean)
    print(f'Saved {len(epochs)} rows to {args.output}')


if __name__ == '__main__':
    main()
