# -*- coding: utf-8 -*-
"""Report generation: PNG figures (Bode, transient), JSON, Markdown.

Figures follow the dataviz defaults: fixed-order categorical palette
(blue then orange), thin 2px lines, hairline grid behind data, ink-colored
text, one y-scale per panel.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np

# palette (dataviz reference instance, light mode)
C1 = '#2a78d6'      # series 1 (blue)
C2 = '#eb6834'      # series 2 (orange)
C3 = '#1baf7a'      # series 3 (aqua)
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASE = '#c3c2b7'
SURF = '#fcfcfb'


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ax.spines.values():
        s.set_color(BASE)
        s.set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, which='major', color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def plot_bode(freqs, T, path, ref=None, status=None):
    """Bode magnitude/phase.  With per-point `status` (review §7.5): solid
    line segments connect only ORIGINAL-adjacent PASS points (no bridging
    across invalid gaps); non-PASS points appear as individual markers --
    NOT_SETTLED hollow circles, FOLDED crosses, FAILED triangles."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.4), dpi=150,
                                   sharex=True)
    fig.patch.set_facecolor(SURF)
    for ax in (ax1, ax2):
        _style(ax)
    freqs = np.asarray(freqs, dtype=float)
    T = np.asarray(T)
    ok = np.isfinite(T)
    # magnitude/phase per point (NaN where the point has no value); the
    # ORIGINAL index order is preserved -- compressing the finite points
    # first would bridge lines across FAILED gaps and hide their markers
    mag_db = np.full(len(T), np.nan)
    ph_deg = np.full(len(T), np.nan)
    mag_db[ok] = 20 * np.log10(np.abs(T[ok]))
    ph_deg[ok] = np.degrees(np.unwrap(np.angle(T[ok])))
    ax1.margins(y=0.12)
    ax2.margins(y=0.12)
    if status is None:
        ax1.semilogx(freqs[ok], mag_db[ok], color=C1, lw=2,
                     label='switching (injection fit)', zorder=3)
        ax2.semilogx(freqs[ok], ph_deg[ok], color=C1, lw=2, zorder=3)
    else:
        st = np.asarray(list(status))
        pass_mask = (st == 'PASS') & ok
        ax1.semilogx([], [], color=C1, lw=2, label='PASS', zorder=3)
        start = None
        for i in range(len(freqs) + 1):
            on = i < len(freqs) and pass_mask[i]
            if on and start is None:
                start = i
            elif not on and start is not None:
                sl = slice(start, i)
                ax1.semilogx(freqs[sl], mag_db[sl], color=C1, lw=2,
                             zorder=3)
                ax2.semilogx(freqs[sl], ph_deg[sl], color=C1, lw=2,
                             zorder=3)
                start = None
        marks = {'NOT_SETTLED': ('o', 'none', C2),
                 'FOLDED': ('x', None, MUTED),
                 'FAILED': ('^', 'none', C3)}
        for kind, (mk, mfc, col) in marks.items():
            m = (st == kind) & ok
            if m.any():
                ax1.semilogx(freqs[m], mag_db[m], marker=mk, ls='', mfc=mfc,
                             mec=col, mew=1.4, ms=5, color=col, label=kind,
                             zorder=4)
                ax2.semilogx(freqs[m], ph_deg[m], marker=mk, ls='', mfc=mfc,
                             mec=col, mew=1.4, ms=5, color=col, zorder=4)
            # FAILED points usually carry no phasor at all (NaN): mark the
            # frequency with a vertical line so the gap is visible and no
            # line can bridge it invisibly
            nf = (st == kind) & ~ok
            if nf.any():
                for fx in freqs[nf]:
                    ax1.axvline(fx, color=col, lw=1.0, alpha=0.5, zorder=2)
                    ax2.axvline(fx, color=col, lw=1.0, alpha=0.5, zorder=2)
                ax1.semilogx([], [], color=col, lw=1.0, alpha=0.7,
                             label=f'{kind} (no fit)', zorder=2)
    if ref is not None:
        fr, Tr = ref
        ax1.semilogx(fr, 20 * np.log10(np.abs(Tr)), color=C2, lw=1.4, ls='--',
                     label='average model (Ridley)', zorder=2)
        ax2.semilogx(fr, np.degrees(np.unwrap(np.angle(Tr))), color=C2,
                     lw=1.4, ls='--', zorder=2)
    else:
        ax1.set_title('|T| from switching-level injection fit', fontsize=9,
                      color=INK2, loc='left')
    if status is not None or ref is not None:
        ax1.legend(fontsize=8, frameon=False, labelcolor=INK2)
    ax1.axhline(0, color=MUTED, lw=0.8)
    ax1.set_ylabel('magnitude (dB)', fontsize=8, color=INK2)
    ax2.axhline(-180, color=MUTED, lw=0.8)
    ax2.set_ylabel('phase (deg)', fontsize=8, color=INK2)
    ax2.set_xlabel('frequency (Hz)', fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def plot_transient(samples, probes: List[str], path,
                   groups: Optional[Dict[str, List[str]]] = None):
    """One panel per probe group (one y-scale per panel; digital probes that
    share a scale are grouped)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = groups or {p: [p] for p in probes}
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.6 * n), dpi=150,
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURF)
    ts = np.array([t for t, _ in samples])
    for ax, (gname, plist) in zip(axes[:, 0], groups.items()):
        _style(ax)
        colors = [C1, C2, C3]
        for i, p in enumerate(plist):
            y = np.array([v[p] for _, v in samples])
            ax.plot(ts, y, lw=1.6 if len(plist) > 1 else 2,
                    color=colors[i % len(colors)],
                    label=p if len(plist) > 1 else None, zorder=3)
        ax.margins(y=0.15)
        if len(plist) > 1:
            ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
        else:
            ax.set_title(plist[0], fontsize=8, color=INK2, loc='left')
        ax.set_ylabel(gname, fontsize=8, color=INK2)
    axes[-1, 0].set_xlabel('time (s)', fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())       # recurse: complex/bool scalars
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (complex, np.complexfloating)):
        return {'re': float(o.real), 'im': float(o.imag)}
    return o


def write_outputs(outdir, data: Dict, figures: Dict[str, str],
                  md_sections: List[str]):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(_jsonable(data), f, indent=2, ensure_ascii=False)
    fig_lines = [f'![{k}](./{os.path.basename(v)})'
                 for k, v in figures.items() if os.path.exists(v)]
    md = ['# popac analysis report\n',
          '\n'.join(fig_lines), '\n'] + md_sections
    with open(os.path.join(outdir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
