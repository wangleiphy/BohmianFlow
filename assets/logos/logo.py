"""Regenerate the BohmianFlow logo (light + dark).

Produces:
    assets/logos/logo-light.{png,svg}   -- custom paper rainbow on white
    assets/logos/logo-dark.{png,svg}    -- turbo rainbow on midnight

Both variants share the same geometry: 60 non-crossing Bohmian
trajectories evolving in the 1D double-well from the unimodal Gaussian
initial state at t = 0 to the bimodal final state at t = T = pi,
rendered from the data in ``data/fig1_doublewell.npz``.

Each trajectory is coloured by its initial position x_0, so its colour is
a conserved label -- a visual nod to non-crossing.

Run from the repo root:
    python assets/logos/logo.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
NPZ = os.path.join(REPO, 'data', 'fig1_doublewell.npz')


# Custom "paper rainbow": full hue sweep but every stop has enough
# luminance contrast against white (the conventional pale yellow in
# Spectral_r is replaced by a darker amber).
PAPER_RAINBOW = LinearSegmentedColormap.from_list('paper_rainbow', [
    '#6a3a9e',   # deep violet
    '#2c58a0',   # royal blue
    '#1f8f8f',   # teal
    '#2e8b3e',   # forest green
    '#c9951a',   # amber gold
    '#e26a2c',   # orange
    '#b1283e',   # crimson
])
try:
    plt.colormaps.register(PAPER_RAINBOW, name='paper_rainbow')
except ValueError:
    pass


def _geometry(x_range=(-4.0, 4.0), n_traj=60, density_height_frac=0.10):
    d = np.load(NPZ)
    x = d['x_grid']; t = d['t_arr']; rho = d['rho']; traj = d['traj_x']
    T = float(d['T'])

    xm = (x >= x_range[0]) & (x <= x_range[1])
    x_plot = x[xm]

    h = density_height_frac
    margin = 0.05
    baselines = np.array([margin + h, 1.0 - margin - h])
    snap_idx = [int(np.argmin(np.abs(t - ts))) for ts in (0.0, T)]

    rho_max = max(rho[si, xm].max() for si in snap_idx)
    silhouettes = []
    for si, base in zip(snap_idx, baselines):
        flip = -1.0 if base < baselines.mean() else 1.0
        rho_scaled = rho[si, xm] / rho_max * h
        silhouettes.append({'base': base, 'upper': base + flip * rho_scaled})

    t_mask = (t >= 0) & (t <= T)
    t_sel = t[t_mask]
    y_of_t = baselines[0] + (baselines[1] - baselines[0]) * \
        (t_sel - t_sel[0]) / (t_sel[-1] - t_sel[0])
    idx = np.linspace(0, traj.shape[1] - 1, n_traj).astype(int)
    traj_xs = [traj[t_mask, j] for j in idx]

    return {'x_plot': x_plot, 'x_range': x_range,
            'silhouettes': silhouettes, 'traj_xs': traj_xs,
            'y_of_t': y_of_t}


def _render(out_stem, cmap_name, bg, density_color,
            traj_lw=1.0, traj_alpha=0.95, density_lw=0.8,
            figsize=(4.0, 4.0), dpi=300):
    g = _geometry()
    cmap = plt.get_cmap(cmap_name)
    x0 = np.array([xp[0] for xp in g['traj_xs']])
    norm = (x0 - x0.min()) / (x0.max() - x0.min() + 1e-12)
    colors = [cmap(n) for n in norm]

    transparent = (bg is None) or (isinstance(bg, str) and bg.lower() == 'none')

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*g['x_range'])
    ax.set_ylim(0, 1)
    ax.axis('off')
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
    else:
        fig.patch.set_facecolor(bg)

    for s in g['silhouettes']:
        ax.plot(g['x_plot'], s['upper'], color=density_color,
                lw=density_lw, zorder=3, solid_capstyle='round')

    for xp, col in zip(g['traj_xs'], colors):
        ax.plot(xp, g['y_of_t'], color=col, lw=traj_lw,
                alpha=traj_alpha, zorder=5, solid_capstyle='round')

    png_path = os.path.join(HERE, f'{out_stem}.png')
    svg_path = os.path.join(HERE, f'{out_stem}.svg')
    save_kwargs = {'transparent': True} if transparent else {'facecolor': bg}
    fig.savefig(png_path, dpi=dpi, **save_kwargs)
    fig.savefig(svg_path, **save_kwargs)
    plt.close(fig)
    print(f'wrote {png_path} and {svg_path}')


def _render_social(out_stem, cmap_name, bg, density_color, text_color,
                   subtitle_color, W=1280, H=640, dpi=100,
                   traj_lw=1.3, traj_alpha=0.95, density_lw=1.1,
                   subtitle_weight='regular'):
    """Render the GitHub social-preview card (1280 x 640).

    Layout: logo mark on the left (square region), title + subtitle
    stacked on the right, with generous white space.
    """
    g = _geometry()
    cmap = plt.get_cmap(cmap_name)
    x0 = np.array([xp[0] for xp in g['traj_xs']])
    norm = (x0 - x0.min()) / (x0.max() - x0.min() + 1e-12)
    colors = [cmap(n) for n in norm]

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.patch.set_facecolor(bg)

    # Left half: logo mark, centred in a square region.
    ax = fig.add_axes([0.03, 0.08, 0.34, 0.84])  # x0, y0, w, h in fig coords
    ax.set_xlim(*g['x_range'])
    ax.set_ylim(0, 1)
    ax.axis('off')
    for s in g['silhouettes']:
        ax.plot(g['x_plot'], s['upper'], color=density_color,
                lw=density_lw, zorder=3)
    for xp, col in zip(g['traj_xs'], colors):
        ax.plot(xp, g['y_of_t'], color=col, lw=traj_lw,
                alpha=traj_alpha, zorder=5, solid_capstyle='round')

    # Right half: title + subtitle.
    text_x = 0.41
    fig.text(text_x, 0.62, 'BohmianFlow', ha='left', va='center',
             fontsize=66, fontweight='bold', color=text_color,
             family='serif')
    fig.text(text_x, 0.44,
             'Quantum Dynamics via Score Matching',
             ha='left', va='center',
             fontsize=24, color=subtitle_color,
             family='serif', style='italic')
    fig.text(text_x, 0.38,
             'on Bohmian Trajectories',
             ha='left', va='center',
             fontsize=24, color=subtitle_color,
             family='serif', style='italic')
    fig.text(text_x, 0.22,
             'github.com/wangleiphy/BohmianFlow',
             ha='left', va='center',
             fontsize=18, color=subtitle_color, family='monospace')

    png = os.path.join(HERE, f'{out_stem}.png')
    fig.savefig(png, dpi=dpi, facecolor=bg)
    plt.close(fig)
    print(f'wrote {png}  ({W}x{H})')


def main():
    _render('logo-light',
            cmap_name='paper_rainbow', bg='white',
            density_color='#1a3a63')
    _render('logo-dark',
            cmap_name='turbo', bg='#0b1220',
            density_color='#dde7f2')
    # Transparent variants: paper_rainbow trajectories on a transparent
    # canvas; silhouette colour/width chosen for each target background.
    _render('logo-light-transparent',
            cmap_name='paper_rainbow', bg='none',
            density_color='#1a3a63', density_lw=0.8)
    _render('logo-dark-transparent',
            cmap_name='paper_rainbow', bg='none',
            density_color='white', density_lw=2.5)
    _render_social('social-preview',
                   cmap_name='paper_rainbow', bg='white',
                   density_color='#1a3a63',
                   text_color='#0d2f4f', subtitle_color='#4a5a72')


if __name__ == '__main__':
    main()
