"""Figure 2: self-consistent Fisher training loop schematic.

This is a lightweight, code-native version of the PRL workflow diagram.  It
does not depend on LaTeX; the script emits a PNG suitable for documentation
and release reproducibility checks.

Example:
    python scripts/fig2_framework.py --output figures/fig2_framework.png
"""

import argparse
import os


def draw(output_path, dpi=300):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    def box(x, y, w, h, title, body, fc):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=1.0,
            edgecolor="#1f3f5b",
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.64, title, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color="#16324a")
        ax.text(x + w / 2, y + h * 0.30, body, ha='center', va='center',
                fontsize=9.0, color="#16324a")

    def arrow(x0, y0, x1, y1, label=None, color="#244c66",
              dashed=False, rad=0.0):
        arr = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle='-|>',
            mutation_scale=12,
            linewidth=1.2,
            color=color,
            linestyle='--' if dashed else '-',
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arr)
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, label,
                    ha='center', va='center', fontsize=7.7,
                    color=color, backgroundcolor='white')

    x, w, h = 0.22, 0.56, 0.105
    ys = [0.82, 0.62, 0.42, 0.22]
    box(x, ys[0], w, h, "Score network",
        r"$s_\theta(x,t)=s_{\rm base}+\nabla\phi_\theta$", "#e9f4ff")
    box(x, ys[1], w, h, "Bohmian trajectories",
        r"integrate $(x,v,F,G)$ with $Q[s_\theta]$", "#edf9f1")
    box(x, ys[2], w, h, "Transported score target",
        r"$F^{-T}[s_0-\nabla_{x_0}\log|\det F|]$", "#eef7f5")
    box(x, ys[3], w, h, "Fisher divergence",
        r"$E_{\rho_\theta}|s_\theta-\nabla\log\rho_\theta|^2$", "#f6eefb")

    arrow(0.50, ys[0], 0.50, ys[1] + h, "quantum force")
    arrow(0.50, ys[1], 0.50, ys[2] + h, "score transport")
    arrow(0.50, ys[2], 0.50, ys[3] + h)
    arrow(x, ys[3] + h / 2, x - 0.12, ys[3] + h / 2,
          color="#b03a2e", dashed=True)
    arrow(x - 0.12, ys[3] + h / 2, x - 0.12, ys[0] + h / 2,
          label=r"$dL/d\theta$ via BPTT", color="#b03a2e",
          dashed=True)
    arrow(x - 0.12, ys[0] + h / 2, x, ys[0] + h / 2,
          color="#b03a2e", dashed=True)

    ax.text(0.50, 0.96, "Self-consistent score matching",
            ha='center', va='center', fontsize=12.5, fontweight='bold',
            color="#16324a")
    ax.text(0.50, 0.075,
            r"Convergence: $L=0 \Leftrightarrow s_\theta=\nabla\log\rho_\theta$",
            ha='center', va='center', fontsize=9.0, color="#16324a")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='figures/fig2_framework.png')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    draw(args.output, args.dpi)


if __name__ == '__main__':
    main()
