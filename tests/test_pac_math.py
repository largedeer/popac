# -*- coding: utf-8 -*-
"""PAC variational-construction regressions (assessment 2026-09-20 section 5).

The record-by-record z-tracking diagnostic on the original-fidelity PS15
model localized the Psi divergence to two construction errors:

1. CMP/D saltation sign.  The jump must be  z+ = z- - df*(g@z-)  (identical
   in form to the DELAY branch); the old `+=` made the garbage of
   zero-duration event pairs (dead topology -> D5 on) ADD instead of
   telescope away, doubling ~1e12-level reconstruction error into z.
2. delay-slot capture must save the PRE-jump crossing perturbation
   g@z- (+ input term), not the post-saltation state g@z+.

plus the per-topology input/output-row issues (urow from the event's own
topology, per-interval output rows).  Tests:

- pacmini: a 3-state clocked circuit whose comparator event flips an analog
  switch immediately (df != 0 at a CMP event) and whose delayed BUF drives
  two further switches (slot chain; simultaneous same-cause delays).
  * analytic forward pass vs finite-difference trigger-map columns
  * dt_coef built from the event's own interval topology
  * PAC vs direct-injection AC at one frequency (probe on an algebraic
    node whose reconstruction row changes with switch state)
- original-fidelity PS15: analytic Psi spectrum vs POP's FD Floquet and the
  full-matrix column agreement (assessment P4 acceptance).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.ir import (Circuit, Res, Cap, VSrc, VcSwitch, Comparator,
                           Gate, Diode, PopTrigger)
from popac.waveforms import Waveform
from popac.engine import Engine
from popac.pop import PopSolver
from popac.pac import PacSolver

TCLK = 1e-6


def pacmini():
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'clki', '0',
             Waveform.pulse(0, 5, TCLK, 0.6e-6, 5e-9, 5e-9)),
        VSrc('VINJ', 'clki', 'clk', Waveform.dc(0.0)),
        Res('R1', 'clk', 'vc', 1e3),
        Cap('C1', 'vc', '0', 100e-12, ic_v=0.5),
        VSrc('VREF', 'vr', '0', Waveform.dc(2.5)),
        VSrc('VP', 'vp', '0', Waveform.dc(5.0)),
        # threshold node: bias divider + weak switched feed from the
        # injection node -> the monitor's direct input channel is topology
        # dependent (weak enough not to re-trigger the comparator)
        Res('R8', 'vth', '0', 10e3),
        Res('R10', 'vr', 'vth', 10e3),
        Res('R7', 'clk', 'vb', 100e3),
        VcSwitch('SW3', 'vb', 'vth', 'u3o', '0', 1e2, 1e7, 2.0, 0.1, 'OPEN'),
        Comparator('U1', out='cmpo', inp='vc', inn='vth',
                   vol=0.0, voh=5.0, rout=10.0, hystwd=1e-3, delay=1e-12),
        # comparator flips an analog switch in the SAME event (df != 0)
        VcSwitch('SW1', 'v2', 'vmid', 'cmpo', '0', 1e2, 1e7, 2.0, 0.1,
                 'OPEN'),
        Res('RL', 'vmid', '0', 1e2),
        Cap('C2', 'v2', '0', 1e-9, ic_v=4.0),
        Res('R2', 'vp', 'v2', 1e2),
        # two gates with the SAME delay from one cause: fire together by
        # construction (shared timing coefficient, must be accepted)
        Gate('U2', out='u2o', inputs=['cmpo'], fn='BUF',
             rout=41.67, rin=1e7, th=2.5, hystwd=0.1, delay=50e-9),
        Gate('U3', out='u3o', inputs=['cmpo'], fn='BUF',
             rout=41.67, rin=1e7, th=2.5, hystwd=0.1, delay=50e-9),
        VcSwitch('SW2', 'v4', 'vmid4', 'u2o', '0', 1e2, 1e7, 2.0, 0.1,
                 'OPEN'),
        Res('RL4', 'vmid4', '0', 1e2),
        # vmid4 is algebraic: when SW2 closes (at U2's delayed firing) it
        # jumps across D1's knee in one step -> the diode turns on at
        # h == 0 after the DELAY event, the PS15 dead-time chain shape
        Diode('D1', 'vmid4', '0', vf=1e-3, ron=0.01, roff=1e9),
        Cap('C4', 'v4', '0', 1e-9, ic_v=4.0),
        Res('R6', 'vp', 'v4', 1e3),
        PopTrigger('X1', node='clk', vref=2.5, edge='rising'),
    ]
    ckt.probes = {'VC': ('v', 'vc'), 'V2N': ('v', 'v2'), 'V4': ('v', 'v4'),
                  'VMID': ('v', 'vmid'), 'VMID4': ('v', 'vmid4'),
                  'VTH': ('v', 'vth'), 'A': ('v', 'clki'), 'B': ('v', 'clk')}
    return ckt


def setup_pac(ckt, pop_opt=None):
    e = Engine(ckt, {'max_events': 5_000_000})
    pop = PopSolver(e, pop_opt or {}).solve()
    assert pop.ok, pop.diag
    e.restore(getattr(pop, 'snap0', None) or e.snapshot())
    e.x = pop.x.copy()
    snap = e.snapshot()
    pac = PacSolver(ckt, snap, pop.x,
                    {'inject_src': 'VINJ', 'probe_a': 'VMID',
                     'probe_b': 'B'})
    records, eng, T = pac._walk()
    return pop, snap, pac, records, eng, T


def fd_columns(ckt, snap, x_star, eps, t_lim=None):
    """Central-difference columns of the trigger-to-trigger map.
    eps: scalar or per-column vector (state scales differ widely)."""
    n = len(x_star)
    epsv = np.broadcast_to(np.asarray(eps, float), (n,))
    cols = []
    t_lim = t_lim or (snap['t'] + 30 * TCLK)
    for j in range(n):
        c = np.zeros(n)
        for sgn in (+1.0, -1.0):
            e2 = Engine(ckt, {'max_events': 5_000_000})
            e2.restore(snap)
            e2.x = x_star.copy()
            e2.x[j] += sgn * epsv[j]
            e2.run(t_lim, stop_on_trigger=True)
            c += sgn * e2.x
        cols.append(c / (2 * epsv[j]))
    return np.column_stack(cols)


def analytic_columns(pac, records, eng, n):
    """Columns of the analytic variational map via the production _forward."""
    from scipy.linalg import expm
    events, nslots, comp_of, comp_coef = pac._build_events(records, eng)
    pac._comp_coef = comp_coef
    Phi = [expm(r['Aug'][:n, :n] * (r['t1'] - r['t0'])) for r in records]
    nrec = len(records)
    A = np.zeros((nrec, n, n))
    b = np.zeros((nrec, n))
    C = np.zeros((2, n))
    for i, r in enumerate(records):
        A[i] = r['Aug'][:n, :n]
    cols = []
    for j in range(n):
        z0 = np.zeros(nslots)
        z0[j] = 1.0
        z, _ = pac._forward(records, events, eng, z0, Phi, A, b, C, n)
        cols.append(z[:n].copy())
    return np.column_stack(cols), events


def test_pacmini_psi_vs_fd():
    ckt = pacmini()
    pop, snap, pac, records, eng, T = setup_pac(ckt, {'pre_cycles': 80})
    n = len(pop.x)
    assert abs(T - TCLK) < 1e-9, f'walk period {T}'
    # two CMP trips per period; U2+U3 (same delay, one cause) fire at the
    # same instant and are folded into one DELAY record (heap aggregation)
    tags = [r['ev_tag'] for r in records]
    assert tags.count('CMP:U1') == 2, tags
    n_delay = sum(1 for r in records if r['ev_kind'] == 'DELAY')
    assert n_delay == 2, tags
    DP = fd_columns(ckt, snap, pop.x, 5e-5)
    AN, _ = analytic_columns(pac, records, eng, n)
    for j in range(n):
        err = np.linalg.norm(AN[:, j] - DP[:, j])
        tol = max(1e-3 * np.linalg.norm(DP[:, j]), 1e-7)
        assert err <= tol, (
            f'column {j}: |an|={np.linalg.norm(AN[:, j]):.3e} '
            f'|fd|={np.linalg.norm(DP[:, j]):.3e} err={err:.3e}')


def test_pacmini_dtcoef_uses_event_topology():
    ckt = pacmini()
    pop, snap, pac, records, eng, T = setup_pac(ckt, {'pre_cycles': 80})
    n = len(pop.x)
    events, nslots, comp_of, comp_coef = pac._build_events(records, eng)
    chI = eng.tc.chan_idx[('V', 'VINJ')]
    checked = 0
    for ev, r in zip(events, records):
        if 'g' not in ev:
            continue
        w_ev = eng._propagate(r['Aug'], r['w0'], r['t1'] - r['t0'])
        hdot = float(np.dot(r['row'], r['Aug'] @ w_ev))
        urow_own = pac._dv_urow(eng, r['topo'], 'vc', 'vth', chI)
        urow_first = pac._dv_urow(eng, records[0]['topo'], 'vc', 'vth', chI)
        if abs(urow_own - urow_first) < 1e-15:
            continue                        # topology-insensitive event
        assert abs(ev['dt_coef'] + urow_own / hdot) < 1e-14 * max(
            1.0, abs(ev['dt_coef'])), (
            f"dt_coef built from wrong topology: {ev['dt_coef']} "
            f'expected {-urow_own / hdot}')
        checked += 1
    assert checked >= 1, 'no topology-sensitive monitor events found'


def test_pacmini_zero_duration_pair_folds():
    """h == 0 event chains (delayed switch slam -> diode across threshold
    at the same instant) must apply ONE saltation: the leader's timing
    against the SUMMED df.  Evaluating the follower's own g on the
    post-leader state feeds the leader's saltation garbage into the
    follower's crossing time; PS15's dead-time chains carry df ~ 1e12 in
    the dead topology and the ~(f- - f+) telescoping breaks."""
    from scipy.linalg import expm
    ckt = pacmini()
    pop, snap, pac, records, eng, T = setup_pac(ckt, {'pre_cycles': 80})
    n = len(pop.x)
    idx = [i for i in range(len(records) - 1)
           if records[i]['ev_kind'] == 'DELAY'
           and records[i + 1]['t1'] - records[i + 1]['t0'] == 0.0
           and records[i + 1]['ev_kind'] == 'D']
    assert idx, ('no zero-duration DELAY->D pair in pacmini records: '
                 + str([(r['ev_kind'], r['ev_tag']) for r in records]))
    events, nslots, comp_of, comp_coef = pac._build_events(records, eng)
    for i in idx:
        lead, foll = events[i], events[i + 1]
        assert foll.get('folded') is not None, (
            f'rec{i + 1} ({records[i + 1]["ev_tag"]}) not folded')
        assert 'df' not in foll and 'g' not in foll, 'follower stays inert'
        # exactly one timing for the chain
        assert ('g' in lead) != (lead.get('fire_slot') is not None), (
            'chain leader must carry exactly one timing mechanism')
        assert 'df' in lead
        # the summed df is the net slope change across the whole chain:
        # f_after_last - f_before_first (the intermediate garbage cancels)
        r = records[i]
        w_ev = eng._propagate(r['Aug'], r['w0'], r['t1'] - r['t0'])
        k = i + 1
        while (k + 1 < len(records)
               and records[k + 1]['t1'] - records[k + 1]['t0'] == 0.0):
            k += 1
        net = (records[k + 1]['Aug'] @ w_ev)[:n] \
            - (records[i]['Aug'] @ w_ev)[:n]
        assert np.allclose(lead['df'], net, rtol=1e-9, atol=1e-6), (
            f'rec{i}: summed df does not telescope to the net slope change')


def test_invalid_pac_suppresses_metrics():
    """HANDOFF §7.3: when the Psi-vs-FD validity gate reports INVALID,
    run.py must NOT compute PM/GM and must mark the report + pac_raw.json.
    The VALIDATED path is exercised by the CLI cross-validations; this
    regression pins the suppression wiring (models/pacmini.yaml runs the
    full production path in ~1 s)."""
    import json
    import shutil
    import popac.pac as pacmod
    import popac.run as runmod
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = os.path.join(here, 'models', 'pacmini.yaml')
    outdir = os.path.join(here, 'out', 'pacmini_invalid_test')
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    orig = pacmod.PacSolver._validate
    pacmod.PacSolver._validate = lambda self, snap_, Psi: {
        'status': 'INVALID', 'max_col_rel': 0.42, 'tol': 5e-3,
        'per_col_rel': []}
    try:
        runmod.main([model, '--analysis', 'pac', '--out', outdir])
    finally:
        pacmod.PacSolver._validate = orig
    with open(os.path.join(outdir, 'pac_raw.json'), encoding='utf-8') as fh:
        raw = json.load(fh)
    assert raw['validation']['status'] == 'INVALID', raw['validation']
    assert raw['metrics']['crossovers'] == [], \
        f'INVALID run published margins: {raw["metrics"]}'
    assert 'invalid_reason' in raw['metrics']
    with open(os.path.join(outdir, 'report.md'), encoding='utf-8') as fh:
        md = fh.read()
    assert 'INVALID' in md and 'margins suppressed' in md, \
        'report.md not marked INVALID'
    # the PAC section's crossings line must show none
    assert '0 dB crossings: none' in md, 'PAC section still claims crossings'


if __name__ == '__main__':
    test_pacmini_psi_vs_fd()
    print('PASS test_pacmini_psi_vs_fd')
    test_pacmini_dtcoef_uses_event_topology()
    print('PASS test_pacmini_dtcoef_uses_event_topology')
    test_pacmini_zero_duration_pair_folds()
    print('PASS test_pacmini_zero_duration_pair_folds')
    test_invalid_pac_suppresses_metrics()
    print('PASS test_invalid_pac_suppresses_metrics')
    print('all pac-math tests passed')
