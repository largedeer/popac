"""P4 diagnostic: track the variational state z record-by-record through one
cycle, FD (two perturbed engine runs) vs the analytic PAC _forward pass.

Goal: localize the FIRST record where the analytic construction departs from
the true linearization.  Columns of interest are the zero-FD-sensitivity
sampled-echo states (C16/C3/C5) that PAC amplifies into the inductor row.

Usage: python debug/p4_ztrack.py [col] [eps]
"""
import os
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


def fd_track(ckt, snap, x0, xp, xm, records):
    """Two perturbed runs sampled at the NOMINAL interval midpoints plus the
    nominal final time.  Sampling at fixed times (not at each run's own event
    times) keeps the comparison convention-consistent with the saltation
    formula: state-at-own-event-time carries an extra f*dt term that the
    same-time saltation convention folds into (f- - f+)*dt."""
    mids = [(r['t0'] + r['t1']) / 2 for r in records]
    tend = records[-1]['t1']
    out = []
    for x in (xp, xm):
        e2 = Engine(ckt, {'max_events': 50_000_000})
        e2.restore(snap)
        e2.x = x.copy()
        xs = [e2.x.copy()]
        for tm in mids:
            e2.run(tm)
            xs.append(e2.x.copy())
        e2.run(tend)
        xs.append(e2.x.copy())
        out.append(xs)
    (xs1, xs2) = out
    n = min(len(xs1), len(xs2))
    z = [(np.asarray(xs1[k]) - np.asarray(xs2[k])) for k in range(n)]
    return z


def analytic_track(pac, records, eng, z0, n):
    """Returns (zs_ends, zs_mids, events): state after each event, and state
    at each interval midpoint (same sampling points as fd_track)."""
    events, nslots, comp_of, comp_coef = pac._build_events(records, eng)
    from scipy.linalg import expm
    nrec = len(records)
    Phi = [expm(r['Aug'][:n, :n] * (r['t1'] - r['t0'])) for r in records]
    pac._comp_coef = comp_coef
    z = np.zeros(nslots)
    z[:n] = z0[:n]
    zs = [z[:n].copy()]
    zmids = []
    for i, ev in enumerate(events):
        r = records[i]
        zmids.append(expm(r['Aug'][:n, :n] * (r['t1'] - r['t0']) / 2)
                     @ z[:n])
        z[:n] = Phi[i] @ z[:n]
        if 'df' in ev:
            if 'g' in ev:
                dt = ev['g'] @ z[:n]
                z[:n] -= ev['df'] * dt
            elif ev.get('fire_slot') is not None:
                dt = z[ev['fire_slot']]
                z[:n] -= ev['df'] * dt
        for item in ev.get('push', ()):
            if item[0] == 'g':
                _, slot = item
                z[slot] = dt if 'df' in ev and dt is not None else 0.0
            else:
                _, slot, src = item
                z[slot] = z[src]
        zs.append(z[:n].copy())
    return zs, zmids, events


if __name__ == '__main__':
    model = os.environ.get('ZT_MODEL', 'models/boost24.yaml')
    col = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-9
    ckt, ana, pop, snap, pac, records, eng, T = setup(model)
    n = len(pop.x)

    if len(sys.argv) > 3:
        # anatomy mode: dump per-event quantities around given record indices
        from scipy.linalg import expm
        events, nslots, comp_of, comp_coef = pac._build_events(records, eng)
        pac._comp_coef = comp_coef
        z = np.zeros(nslots)
        z[col] = 1.0
        Phi = [expm(r['Aug'][:n, :n] * (r['t1'] - r['t0']))
               for r in records]
        for i, ev in enumerate(events):
            r = records[i]
            h = r['t1'] - r['t0']
            z[:n] = Phi[i] @ z[:n]
            dt = None
            if 'df' in ev:
                dfn = np.linalg.norm(ev['df'])
                parts = []
                if 'g' in ev:
                    dt = ev['g'] @ z[:n]
                    parts.append(f'g@z-={dt:+.3e}')
                if ev.get('fire_slot') is not None:
                    dt = z[ev['fire_slot']]
                    parts.append(f'slot={dt:+.3e}')
                pre = np.linalg.norm(z[:n])
                if dt is None:
                    parts.append('PINNED(no saltation)')
                else:
                    z[:n] -= ev['df'] * dt  # fixed sign: z+ = z- - df*dt
                post = np.linalg.norm(z[:n])
                top = np.argsort(-np.abs(ev['df']))[:4]
                dfs = ','.join(f'{j}:{ev["df"][j]:+.2e}' for j in top)
                print(f'rec{i:3d} {ev["kind"]:5s}:{ev["tag"]:14s} h={h:.3e} '
                      f'|df|={dfn:.2e} [{dfs}] {" ".join(parts)} '
                      f'|z| {pre:.3e}->{post:.3e}')
                if i + 1 < len(records):
                    A1 = records[i + 1]['Aug'][:n, :n]
                    h1 = records[i + 1]['t1'] - records[i + 1]['t0']
                    lam = np.linalg.eigvals(A1)
                    damped = expm(A1 * h1) @ ev['df']
                    print(f'        next: h={h1:.3e} lam_re=[{lam.real.min():+.2e},'
                          f'{lam.real.max():+.2e}] '
                          f'|Phi@df|={np.linalg.norm(damped):.3e} '
                      f'|df|={dfn:.3e}')
            # slot capture uses the PRE-jump crossing perturbation
            for item in ev.get('push', ()):
                if item[0] == 'g':
                    z[item[1]] = 0.0 if dt is None else dt
                else:
                    z[item[1]] = z[item[2]]
        sys.exit(0)

    z0 = np.zeros(n)
    z0[col] = 1.0
    xp = pop.x.copy(); xp[col] += eps
    xm = pop.x.copy(); xm[col] -= eps
    zfd = fd_track(ckt, snap, pop.x, xp, xm, records)
    zan, zmids, events = analytic_track(pac, records, eng, z0, n)

    # compare at nominal interval midpoints: fd[k+1] is mid of interval k,
    # zmids[k] the analytic prediction there; fd[-1] vs zan[-1] is the
    # cycle end (same convention as the trigger map)
    print(f'model={model} col={col} eps={eps:g} nrec={len(records)}')
    scale = max(1e-30, np.linalg.norm(pop.x))
    bad = None
    rows = [(k, zmids[k], zfd[k + 1] / (2 * eps),
             f'mid{ k }@before {records[k]["ev_kind"]}:{records[k]["ev_tag"]}')
            for k in range(min(len(zmids), len(zfd) - 1))]
    rows.append((len(records), zan[-1], zfd[-1] / (2 * eps), 'END'))
    for k, a, b, tag in rows:
        err = np.linalg.norm(a - b)
        rel = err / max(np.linalg.norm(b), 1e-30)
        mark = ''
        if (rel > 1e-3 and err > 1e-9 * scale) or err > 1e-6:
            if bad is None:
                bad = k
            mark = '  <<<< FIRST DIVERGENCE'
        if k < 2 or mark or (bad is not None and k < bad + 3) \
                or k == len(records):
            print(f' {tag:38s} |an|={np.linalg.norm(a):9.3e} '
                  f'|fd|={np.linalg.norm(b):9.3e} rel={rel:9.2e}{mark}')
    if bad is None:
        print('  no divergence found (all records agree)')
    else:
        a, b = rows[bad][1], rows[bad][2]
        d = np.abs(a - b)
        print(f'  detail at {rows[bad][3]}:')
        for j in np.argsort(-d)[:6]:
            print(f'    state{j}: an={a[j]:+.3e} fd={b[j]:+.3e}')
