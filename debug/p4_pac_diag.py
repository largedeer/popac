"""P4 diagnostic: why does PAC's Psi diverge on the original-fidelity model?

Phase 1 evidence gathering:
(1) |df| audit on BOTH models (simplified vs original) — is a huge df per se
    the discriminator?  (The simplified model's S5 flip has tau=10 fs.)
(2) state-layout consistency across topologies — do consecutive records
    share the same reduced state basis?
(3) column-by-column Psi vs finite-difference DP on the original model —
    localize WHICH state components diverge.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from popac.model_loader import load_yaml
from popac.engine import Engine
from popac.pop import PopSolver
from popac.pac import PacSolver


def setup(model):
    ckt, ana = load_yaml(model)
    e = Engine(ckt, {'max_events': 50_000_000})
    pop = PopSolver(e, dict(ana.get('pop', {}))).solve()
    e.restore(getattr(pop, 'snap0', None) or e.snapshot())
    e.x = pop.x.copy()
    snap = e.snapshot()
    pac = PacSolver(ckt, snap, pop.x, {})
    records, eng, T = pac._walk()
    return ckt, ana, pop, snap, pac, records, eng, T


def df_audit(records, eng, n, tag):
    print(f'--- |df| audit ({tag}, n={n}) ---')
    mx = 0.0
    for i, r in enumerate(records):
        kind = r['ev_kind']
        if kind in ('CMP', 'D', 'DELAY') and i + 1 < len(records):
            w_ev = eng._propagate(r['Aug'], r['w0'], r['t1'] - r['t0'])
            f_pre = (r['Aug'] @ w_ev)[:n]
            f_post = (records[i + 1]['Aug'] @ w_ev)[:n]
            dfn = np.linalg.norm(f_post - f_pre)
            mx = max(mx, dfn)
            if dfn > 1e6:
                print(f'  rec{i} {kind}:{r["ev_tag"]} |df|={dfn:.2e}')
    print(f'  max|df| = {mx:.2e}')


def layout_check(records, tag):
    print(f'--- state layout consistency ({tag}) ---')
    lay = [tuple(r['topo'].states) if hasattr(r['topo'], 'states')
           else tuple(getattr(r['topo'], 'state_names', ()))
           for r in records]
    ref = lay[0]
    diffs = {i for i in range(len(lay)) if lay[i] != ref}
    print(f'  layout differing records: {sorted(diffs) if diffs else "none"}')
    if diffs:
        for i in sorted(diffs)[:6]:
            print(f'   rec{i} {records[i]["ev_kind"]}:{records[i]["ev_tag"]}:'
                  f' {lay[i]}')
    return diffs


def psi_vs_fd(ckt, pop, snap, pac, records, eng, T, n):
    """Column-by-column analytic Psi vs central-difference DP of the map."""
    print('--- Psi vs FD DP (original) ---')
    events, nslots, comp_of, comp_coef = pac._build_events(records, eng)
    nrec = len(records)
    from scipy.linalg import expm
    Phi = [expm(r['Aug'][:n, :n] * (r['t1'] - r['t0'])) for r in records]
    Psi = np.eye(nslots)
    for i, ev in enumerate(events):
        M = np.eye(nslots)
        M[:n, :n] = Phi[i]
        Psi = M @ Psi
        if 'df' in ev:
            if 'g' in ev:
                S = np.eye(nslots)
                S[:n, :n] += np.outer(ev['df'], ev['g'])
                Psi = S @ Psi
                for item in ev.get('push', ()):
                    if item[0] == 'g':
                        R = np.eye(nslots)
                        R[item[1], :n] = ev['g']
                        Psi = R @ Psi
            elif ev.get('fire_slot') is not None:
                S = np.eye(nslots)
                S[:n, ev['fire_slot']] -= ev['df']
                Psi = S @ Psi
                for item in ev.get('push', ()):
                    R = np.eye(nslots)
                    R[item[1], item[2]] = 1.0
                    Psi = R @ Psi
    A = Psi[:n, :n]

    # FD DP: perturb x0, run exactly one period on a fresh engine
    def cycle(x0):
        e2 = Engine(ckt, {'max_events': 50_000_000})
        e2.restore(snap)
        e2.x = x0.copy()
        e2.run(e2.t + T)
        return e2.x[:n]

    eps = 1e-9
    cols = []
    for j in range(n):
        xp = pop.x.copy(); xp[j] += eps
        xm = pop.x.copy(); xm[j] -= eps
        cols.append((cycle(xp) - cycle(xm)) / (2 * eps))
    DP = np.column_stack(cols)

    print('  j  |Psi col|    |DP col|    rel.err  worst_row')
    for j in range(n):
        a, b = A[:, j], DP[:, j]
        rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
        w = int(np.argmax(np.abs(a - b)))
        print(f'  {j}  {np.linalg.norm(a):9.3e} {np.linalg.norm(b):9.3e} '
              f'  {rel:8.2e}  {w}')


if __name__ == '__main__':
    for model in ('models/buck5v.yaml', 'models/boost24.yaml'):
        ckt, ana, pop, snap, pac, records, eng, T = setup(model)
        n = len(pop.x)
        df_audit(records, eng, n, model)
        layout_check(records, model)
        print()
    # column localization on the diverging model
    ckt, ana, pop, snap, pac, records, eng, T = setup(
        'models/boost24.yaml')
    psi_vs_fd(ckt, pop, snap, pac, records, eng, T, len(pop.x))
