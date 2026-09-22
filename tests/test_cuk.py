# -*- coding: utf-8 -*-
"""Seventh-topology regression (TOPOLOGY_CAMPAIGN_PLAN P1): coupled-
inductor Cuk, 12 V -> -15 V / 1 A at 200 kHz.  The Tier-1 story is the
zero-ripple coupling: with v_L1 = -v_L2 exactly (ideal Cuk) and
SUBTRACTIVE coupling, the input-side ripple ratio is
R_in = L1*(L2-M)/(L1*L2-M^2) -> 0 at M = L2.
Design L1=47u / L2=33u / M=32.5u (k=-0.8248): R_in = 0.0475 (input
ripple /21).  The uncoupled variant of the same model (K removed) is
the control -- exactly the buck2ph ripple-cancellation test pattern.
If the uncoupled control fails POP on the exchange resonance (sepic
HANDOFF 16.1), patch damping resistors into the CONTROL variant only.
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
                  'models', 'cuk24.yaml')
_NPT = 400          # input ripple is the measured quantity: sample finer


def _orbit(coupled=True):
    ckt, ana = load_yaml(_M)
    if not coupled:
        ckt.devices = [d for d in ckt.devices
                       if getattr(d, 'name', '') != 'K1']
    e = Engine(ckt, {'max_events': 5_000_000, 'chunk': 500e-9})
    solver = PopSolver(e, dict(ana.get('pop', {})))
    pop = solver.solve()
    assert pop.ok, f'POP failed: {pop.diag}'
    e.restore(solver.snap0)
    e.x = pop.x.copy()
    t0 = e.t
    s, _ = e.run(t0 + 2 * pop.period, sample_dt=pop.period / _NPT,
                 probes=['VOUT', 'IL1', 'IL2', 'NA', 'NB', 'EA'])
    per = [v for t, v in s if t >= t0 + pop.period - 1e-15]
    return {
        'ckt': ckt, 'ana': ana, 'T': pop.period, 'res': pop.residual,
        'maxlam': pop.max_multiplier,
        'snap': solver.snap0, 'x': pop.x.copy(),
        'vout': float(np.mean([v['VOUT'] for v in per])),
        'il1': np.array([v['IL1'] for v in per]),
        'il2': np.array([v['IL2'] for v in per]),
        'vc': float(np.mean([v['NA'] - v['NB'] for v in per])),
    }


@lru_cache(maxsize=None)
def _base():
    return _orbit(True)


@lru_cache(maxsize=None)
def _uncoupled():
    return _orbit(False)


def _find(ckt, name):
    return next(d for d in ckt.devices if getattr(d, 'name', '') == name)


def test_cuk_dc_laws():
    o = _base()
    ckt = o['ckt']
    vref = _find(ckt, 'VREF').wave.eval(0.0)
    r17, r18 = _find(ckt, 'R17').r, _find(ckt, 'R18').r
    r1 = _find(ckt, 'R1').r
    vf = _find(ckt, 'D1').vf
    vin = _find(ckt, 'VIN').wave.eval(0.0)
    vabs = -o['vout']
    # inverting output: VREF is negative in the |VOUT|-sensing arrangement
    assert abs(vabs - abs(vref) * (r17 + r18) / r18) < 0.3, o['vout']
    d = (vabs + vf) / (vin + vabs + vf)
    i2 = float(np.mean(np.abs(o['il2'])))
    assert abs(i2 - vabs / r1) / i2 < 0.03, i2          # = IOUT exactly
    i1 = float(np.mean(o['il1']))
    assert abs(i1 - d / (1 - d) * i2) / i1 < 0.05, (i1, i2, d)
    # energy-transfer cap DC magnitude = VIN + |VOUT| (polarity per dots)
    assert abs(abs(o['vc']) - (vin + vabs)) / (vin + vabs) < 0.05, o['vc']


def test_cuk_coupling_cancels_input_ripple():
    o, u = _base(), _uncoupled()
    d_pp = float(np.ptp(o['il1']))
    u_pp = float(np.ptp(u['il1']))
    assert u_pp > 0.5, u_pp                  # control has real ripple
    assert d_pp < 0.2 * u_pp, (d_pp, u_pp)   # >=5x cancellation
    assert abs(float(np.mean(o['il1'])) - float(np.mean(u['il1']))) \
        / float(np.mean(u['il1'])) < 0.10    # operating point preserved


def test_cuk_pop_pac():
    o = _base()
    assert abs(o['T'] - 5e-6) < 1e-12, o['T']
    assert o['res'] < 2e-5, o['res']
    assert 0.99 < o['maxlam'] < 0.9999, o['maxlam']
    pac = PacSolver(o['ckt'], o['snap'], o['x'], dict(o['ana'].get('pac', {})))
    res = pac.solve()
    val = res['info']['validation']
    assert val['status'] == 'VALIDATED', val
    m = loop_metrics(res['freqs'], res['T'])
    assert m['crossovers'], 'no 0 dB crossing'
    fc = m['crossovers'][0]
    assert 100 < fc['freq'] < 5000, fc
    assert fc['pm_deg'] > 45.0, fc


if __name__ == '__main__':
    test_cuk_dc_laws()
    print('PASS test_cuk_dc_laws')
    test_cuk_coupling_cancels_input_ripple()
    print('PASS test_cuk_coupling_cancels_input_ripple')
    test_cuk_pop_pac()
    print('PASS test_cuk_pop_pac')
    print('all cuk tests passed')
