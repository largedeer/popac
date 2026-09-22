# -*- coding: utf-8 -*-
"""Third-topology regression (CPP_MIGRATION_AND_TOPOLOGY_PLAN T2 / HANDOFF
§15): two-phase interleaved peak-current-mode asynchronous buck.

Pins the DSL/POP/PAC pipeline on a multi-phase topology:

- DECLARED current sharing = independent peak loops (own H/CMP/ramp/SRFF
  per phase, shared EA): IL_A ~ IL_B ~ 1 A at 5 V / 2.5 ohm
- 180 deg interleave: IL_B(t + T/2) reproduces IL_A(t) to noise level
- ripple cancellation follows the duty/phase-count formula
  dI_sum = dIL*(1-2D)/(1-D) (D < 0.5), NOT a hardcoded "less than half":
  the phase-aligned variant (V3B delay 0, the single-phase equivalent at
  the same total load and inductance resource, L_A||L_B) must show the
  un-cancelled 2*dIL ripple
- POP converges with exact T = 2.5 us and a physical Floquet spectrum;
  PAC validates and the composite loop lands in the PS15-proven window
  (EA gm halved 24u -> 12u to cancel the x2 gain of two parallel peak
  loops)
"""
import os
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.model_loader import load_yaml
from popac.waveforms import Waveform
from popac.engine import Engine
from popac.pop import PopSolver
from popac.pac import PacSolver
from popac.ac import loop_metrics

_M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'models', 'buck2ph.yaml')

# fine grid: the interleaved ESR triangle (slopes ~2.1 A/us through 5 m)
# loses ~6% p2p at T/200 sampling; T/800 keeps the grid bias ~1.5%
_NPT = 800


def _pop_and_orbit(model=_M, patch=None):
    """POP solve + two periods of samples (stats over the second).

    e.run samples [t0, t0 + 2T] at T/NPT -> ~2*NPT samples total; the
    second period lives in indices [NPT, 2*NPT).
    """
    ckt, ana = load_yaml(model)
    for name, wave in (patch or {}).items():
        next(d for d in ckt.devices if getattr(d, 'name', '') == name).wave = wave
    e = Engine(ckt, {'max_events': 5_000_000})
    solver = PopSolver(e, dict(ana.get('pop', {})))
    pop = solver.solve()
    assert pop.ok, f'POP failed: {pop.diag}'
    e.restore(solver.snap0)
    e.x = pop.x.copy()
    t_end = e.t + 2 * pop.period
    s, _ = e.run(t_end, sample_dt=pop.period / _NPT,
                 probes=['VOUT', 'ILA', 'ILB', 'IIN'])
    per = [v for t, v in s if t >= t_end - pop.period - 1e-15]
    ila = np.array([v['ILA'] for v in per])
    ilb = np.array([v['ILB'] for v in per])
    return {
        'ckt': ckt, 'T': pop.period, 'res': pop.residual,
        'maxlam': pop.max_multiplier,
        'snap': solver.snap0, 'x': pop.x.copy(),
        'vout': float(np.mean([v['VOUT'] for v in per])),
        'dvout': float(np.ptp([v['VOUT'] for v in per])),
        'ila': ila, 'ilb': ilb,
        'iin': float(np.mean([v['IIN'] for v in per])),
        'raw_ila': np.array([v['ILA'] for t, v in s]),
        'raw_ilb': np.array([v['ILB'] for t, v in s]),
    }


def _aligned_patch():
    """Phase B clocks delayed by 0 (both phases switch together): the
    same-total-load, same-total-inductance single-phase equivalent."""
    return {
        'V3B': Waveform.pulse(0, 5, 2.5e-6, 25e-9, 2.5e-9, 2.5e-9, 0.0),
        'V4B': Waveform.pulse(0, 5, 2.5e-6, 10e-9, 2.5e-9, 2.5e-9, 2.4e-6),
    }


@lru_cache(maxsize=None)
def _interleaved():
    return _pop_and_orbit()


@lru_cache(maxsize=None)
def _aligned():
    return _pop_and_orbit(patch=_aligned_patch())


def _find(ckt, name):
    return next(d for d in ckt.devices if getattr(d, 'name', '') == name)


# ------------------------------------------------- DC, sharing, interleave
def test_buck2ph_dc_sharing_180deg():
    o = _interleaved()
    vin = _find(o['ckt'], 'VIN').wave.eval(0.0)     # 12 V
    vref = _find(o['ckt'], 'VREF').wave.eval(0.0)   # 1 V
    r17 = _find(o['ckt'], 'R17').r                  # 100k
    r18 = _find(o['ckt'], 'R18').r                  # 25k
    # regulation: divider feeds VREF -> VOUT = 5.0 V
    assert abs(o['vout'] - vref * (r17 + r18) / r18) < 0.05, o['vout']
    # acceptance center values (plan T2): IL_A ~ IL_B ~ 1 A, total 2 A
    assert abs(o['ila'].mean() - 1.0) < 0.03, o['ila'].mean()
    assert abs(o['ilb'].mean() - 1.0) < 0.03, o['ilb'].mean()
    assert abs(o['ila'].mean() - o['ilb'].mean()) < 0.05
    # power balance: VIN*|IIN| = Pout + 2 diode blocks + ron (<6%, boost-style
    # envelope; ripple-mean sampling bias ~1%)
    r1 = _find(o['ckt'], 'R1').r
    vf = _find(o['ckt'], 'DA').vf
    d = (o['vout'] + vf) / (vin + vf)               # volt-second duty
    pin = vin * abs(o['iin'])
    pout = o['vout'] ** 2 / r1 + 2.0 * (1.0 - d) * vf * 1.0
    assert abs(pin / pout - 1.0) < 0.06, (pin, pout)
    # 180 deg interleave: comparing IL_A with IL_B half a period LATER
    # reproduces the same triangle (noise level), while the direct
    # comparison shows the true half-period offset
    ila_p = o['raw_ila']
    ilb_p = o['raw_ilb']
    n = _NPT                                       # samples per period
    ila_q = ila_p[n:n + n // 2]                    # [T, 1.25T) of orbit
    ilb_later = ilb_p[n + n // 2:2 * n]            # ILB at t + T/2
    err_half = float(np.mean(np.abs(ila_q - ilb_later)))
    err_direct = float(np.mean(np.abs(ila_p[n:2 * n] - ilb_p[n:2 * n])))
    assert err_direct > 0.3, err_direct            # phases really offset
    assert err_half < 0.1 * err_direct, (err_half, err_direct)
    assert err_half < 0.05, err_half


# -------------------------------------------- ripple: formula, cancellation
def test_buck2ph_ripple_formula_and_cancellation():
    o = _interleaved()
    vin = _find(o['ckt'], 'VIN').wave.eval(0.0)
    vf = _find(o['ckt'], 'DA').vf
    la = _find(o['ckt'], 'LA').l
    esr = _find(o['ckt'], 'R8').r
    c6 = _find(o['ckt'], 'C6').c
    T = o['T']
    # volt-second balance (async buck): m1*D = m2*(1-D) with m2=(VOUT+vf)/L
    d = (o['vout'] + vf) / (vin + vf)
    dil = (vin - o['vout']) * d * T / la           # per-phase triangle
    # duty/phase-count formula (2 phases, D < 0.5), NOT a hardcoded bound
    di_sum = dil * (1.0 - 2.0 * d) / (1.0 - d)
    lo = di_sum * esr                              # ESR triangle floor
    hi = lo + di_sum / (8.0 * c6 * (2.0 / T))      # + cap charge ripple
    assert o['dvout'] > 0.95 * lo, (o['dvout'], lo, hi)
    assert o['dvout'] < 1.05 * hi, (o['dvout'], lo, hi)
    # interleaved sum current really is the cancelled one: measured p-p of
    # (IL_A + IL_B) matches di_sum at 2/T, not 2*dil at 1/T
    isum = float(np.ptp(o['ila'] + o['ilb']))
    assert abs(isum - di_sum) / di_sum < 0.08, (isum, di_sum)
    # phase-aligned variant (single-phase equivalent at same load and total
    # inductance): no cancellation, ripple = 2*dil through the same cap
    oa = _aligned()
    assert oa['dvout'] > 4.0 * o['dvout'], (oa['dvout'], o['dvout'])
    isum_a = float(np.ptp(oa['ila'] + oa['ilb']))
    assert abs(isum_a - 2.0 * dil) / (2.0 * dil) < 0.08, (isum_a, 2.0 * dil)


# -------------------------------------------------------- POP + PAC quality
def test_buck2ph_pop_pac():
    o = _interleaved()
    assert abs(o['T'] - 2.5e-6) < 1e-12, o['T']    # exact per-phase T_sw
    assert o['res'] < 1e-5, o['res']
    # same slow-mode family as the reference buck (0.847) on a 2 A load: stable, < 1
    assert 0.80 < o['maxlam'] < 0.90, o['maxlam']

    ckt, ana = load_yaml(_M)
    pac = PacSolver(ckt, o['snap'], o['x'], dict(ana.get('pac', {})))
    res = pac.solve()
    val = res['info']['validation']
    assert val['status'] == 'VALIDATED', val
    m = loop_metrics(res['freqs'], res['T'])
    assert m['crossovers'], 'no 0 dB crossing found'
    fc = m['crossovers'][0]
    # composite loop = PS15 shape (EA gm halved cancels the x2 two-phase
    # plant gain): fc near 27 kHz, generous PM
    assert 15e3 < fc['freq'] < 45e3, fc
    assert 50 < fc['pm_deg'] < 85, fc


if __name__ == '__main__':
    test_buck2ph_dc_sharing_180deg()
    print('PASS test_buck2ph_dc_sharing_180deg')
    test_buck2ph_ripple_formula_and_cancellation()
    print('PASS test_buck2ph_ripple_formula_and_cancellation')
    test_buck2ph_pop_pac()
    print('PASS test_buck2ph_pop_pac')
    print('all buck2ph tests passed')
