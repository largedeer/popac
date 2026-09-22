# -*- coding: utf-8 -*-
"""Sixth-topology regression (TOPOLOGY_CAMPAIGN_PLAN P1): inverting
buck-boost, 12 V -> -12 V / 1 A at 200 kHz, peak CMC.

Acceptance (closed-form from model parameters only):
- DC laws: |VOUT| = VREF*(R17+R18)/R18; IL_avg = IOUT/(1-D);
  volt-second ripple dIL = VIN*D*T/L (D includes the diode drop)
- ripple bounds (DIODE-FED output, boost family -- NOT the buck
  formula): cap current swings -IOUT (ON) to IL_pk-IOUT (OFF), so its
  p2p is IL_pk and dv_pp in [IL_pk*ESR, IL_pk*ESR + IOUT*D*T/C]
  (bring-up round 3: the buck-formula draft bounded 18.5-19.9 mV while
  the sim sat at 50.5 mV = IL_pk*ESR exactly -- family physics)
- POP N=1 exact T; PAC VALIDATED with fc in a low window (RHP zero at
  27 kHz keeps the design fc ~1.5 kHz) and PM > 45 deg
"""
import os
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.model_loader import load_yaml
from popac.engine import Engine
from popac.pop import PopSolver
from popac.pac import PacSolver
from popac.ac import loop_metrics

_M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'models', 'buckboost24.yaml')

_NPT = 200


def _orbit():
    ckt, ana = load_yaml(_M)
    e = Engine(ckt, {'max_events': 5_000_000, 'chunk': 500e-9})
    solver = PopSolver(e, dict(ana.get('pop', {})))
    pop = solver.solve()
    assert pop.ok, f'POP failed: {pop.diag}'
    e.restore(solver.snap0)
    e.x = pop.x.copy()
    t0 = e.t
    s, _ = e.run(t0 + 2 * pop.period, sample_dt=pop.period / _NPT,
                 probes=['VOUT', 'IL1', 'VSW', 'EA', 'PWM', 'IIN'])
    per = [v for t, v in s if t >= t0 + pop.period - 1e-15]
    vouts = np.array([v['VOUT'] for v in per])
    il1 = np.array([v['IL1'] for v in per])
    return {
        'ckt': ckt, 'ana': ana, 'T': pop.period, 'res': pop.residual,
        'maxlam': pop.max_multiplier,
        'snap': solver.snap0, 'x': pop.x.copy(),
        'vout': float(np.mean(vouts)), 'vouts': vouts, 'il1': il1,
        'vsw': np.array([v['VSW'] for v in per]),
        'iin': float(np.mean([v['IIN'] for v in per])),
    }


@lru_cache(maxsize=None)
def _base():
    return _orbit()


def _find(ckt, name):
    return next(d for d in ckt.devices if getattr(d, 'name', '') == name)


def _pars():
    ckt = _base()['ckt']
    vin = _find(ckt, 'VIN').wave.eval(0.0)
    vref = _find(ckt, 'VREF').wave.eval(0.0)
    r17 = _find(ckt, 'R17').r
    r18 = _find(ckt, 'R18').r
    r1 = _find(ckt, 'R1').r
    vf = _find(ckt, 'D1').vf
    ll = _find(ckt, 'L1').l
    return vin, vref, r17, r18, r1, vf, ll


def test_buckboost_dc_laws():
    o = _base()
    vin, vref, r17, r18, r1, vf, ll = _pars()
    T = o['T']
    # regulation: |VOUT| set by the divider (output NEGATIVE, VREF is
    # likewise negative in the inverting-feedback arrangement)
    assert abs(-o['vout'] - abs(vref) * (r17 + r18) / r18) < 0.25, o['vout']
    # duty from volt-seconds including the diode drop
    vabs = -o['vout']
    d = (vabs + vf) / (vin + vabs + vf)
    # average inductor current
    il_avg = float(np.mean(o['il1']))
    assert abs(il_avg - (vabs / r1) / (1 - d)) / il_avg < 0.03, (il_avg, d)
    # volt-second ripple
    dil = float(np.ptp(o['il1']))
    assert abs(dil - vin * d * T / ll) / (vin * d * T / ll) < 0.05, dil
    # CCM
    assert float(o['il1'].min()) > 0.0


def test_buckboost_ripple_bounds():
    o = _base()
    vin, vref, r17, r18, r1, vf, ll = _pars()
    esr = _find(o['ckt'], 'RESR').r
    c6 = _find(o['ckt'], 'C6').c
    vabs = -o['vout']
    d = (vabs + vf) / (vin + vabs + vf)
    iout = vabs / r1
    il_pk = float(o['il1'].max())
    dv_pp = float(np.ptp(o['vouts']))
    lo = il_pk * esr
    hi = lo + iout * d * o['T'] / c6
    # 1% slack on the floor: the p2p is step-dominated (ESR jump at the
    # switching instants) and the 200-point sampling can miss the exact
    # extremes by up to ~0.2% (measured gap 0.08 mV on 50.5 mV)
    assert lo * 0.99 - 1e-9 <= dv_pp <= hi * 1.05 + 1e-9, (dv_pp, lo, hi)


def test_buckboost_pop_pac():
    o = _base()
    assert abs(o['T'] - 5e-6) < 1e-12, o['T']        # N=1, exact T_sw
    assert o['res'] < 2e-5, o['res']
    assert 0.99 < o['maxlam'] < 0.9999, o['maxlam']
    ckt, ana = o['ckt'], o['ana']
    pac = PacSolver(ckt, o['snap'], o['x'], dict(ana.get('pac', {})))
    res = pac.solve()
    val = res['info']['validation']
    assert val['status'] == 'VALIDATED', val
    m = loop_metrics(res['freqs'], res['T'])
    assert m['crossovers'], 'no 0 dB crossing'
    fc = m['crossovers'][0]
    assert 200 < fc['freq'] < 8000, fc
    assert fc['pm_deg'] > 45.0, fc


if __name__ == '__main__':
    test_buckboost_dc_laws()
    print('PASS test_buckboost_dc_laws')
    test_buckboost_ripple_bounds()
    print('PASS test_buckboost_ripple_bounds')
    test_buckboost_pop_pac()
    print('PASS test_buckboost_pop_pac')
    print('all buck-boost tests passed')
