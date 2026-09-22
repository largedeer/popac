"""P4 diagnostic #2: per-midpoint component comparison an vs fd (fixed
construction, nominal-time sampling).

Usage: python debug/p4_ztrack2.py [col] [eps] [components...]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from p4_ztrack import setup, fd_track, analytic_track

if __name__ == '__main__':
    model = os.environ.get('ZT_MODEL', 'models/boost24.yaml')
    col = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
    comps = [int(a) for a in sys.argv[3:]] or [1, 6, 7, 8]

    ckt, ana, pop, snap, pac, records, eng, T = setup(model)
    n = len(pop.x)
    comps = [j for j in comps if j < n] or [0]   # small models (VCO etc.)

    z0 = np.zeros(n)
    z0[col] = 1.0
    xp = pop.x.copy(); xp[col] += eps
    xm = pop.x.copy(); xm[col] -= eps
    zfd = fd_track(ckt, snap, pop.x, xp, xm, records)
    zan, zmids, events = analytic_track(pac, records, eng, z0, n)
    print(f'model={model} col={col} eps={eps:g} nrec={len(records)}')
    print(f'  {"mid":>4} {"ends-at event":22s} ' + ' '.join(
        f'{"an"+str(j):>12} {"fd"+str(j):>12}' for j in comps))
    for k in range(min(len(zmids), len(zfd) - 1)):
        a, b = zmids[k], zfd[k + 1] / (2 * eps)
        tag = f"{records[k]['ev_kind']}:{records[k]['ev_tag']}"
        cells = ' '.join(f'{a[j]:+12.4e} {b[j]:+12.4e}' for j in comps)
        print(f'  {k:4d} {tag:22s} {cells}')
    a, b = zan[-1], zfd[-1] / (2 * eps)
    cells = ' '.join(f'{a[j]:+12.4e} {b[j]:+12.4e}' for j in comps)
    print(f'  {"END":>4} {"":22s} {cells}')
