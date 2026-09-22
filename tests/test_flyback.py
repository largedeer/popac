# -*- coding: utf-8 -*-
"""Fifth-topology regression (CPP_MIGRATION_AND_TOPOLOGY_PLAN T5):
peak-current-mode DCM flyback, 12 V -> ~24.5 V / 0.5 A at 100 kHz — the
first application of the T4 mutual-inductance device.

Acceptance (plan T5, closed-form from model parameters only):
- DCM: the primary current starts each cycle at ~0 and the secondary
  current returns to zero before the cycle ends (idle fraction > 0)
- magnetizing-energy transport: 0.5*Lp*Ip^2/T = input power; output
  power + diode loss + clamp loss matches within tolerance
- transformer action (dots + turns): v(ds) during ON = -k*VIN (reflected
  input), v(ds) during demag = VOUT+vf
- RCD clamp: peak v(sw) is bounded by the clamp rail, the rail sits
  ABOVE the reflected voltage (no magnetizing-energy theft), and clamp
  dissipation GROWS with leakage (k 0.98 -> 0.95 trend)
- POP/PAC VALIDATED; the one-period identity P(x*) = x* is asserted to
  guard the pseudo-N=2 regression seen with an earlier EA (the N=1
  Newton bounced, N=2 converged to the same period-1 orbit — evidence
  out/flyback24_pop.log)

The leakage representation is DECLARED in the model header: coupled
windings, implicit leakage (1-k)*L, no separate leakage inductor (no
double counting).  DCM has no subharmonic mode -> no slope compensation.
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
                  'models', 'flyback24.yaml')

_NPT = 200


def _orbit(k=None):
    ckt, ana = load_yaml(_M)
    if k is not None:
        for d in ckt.devices:
            if getattr(d, 'name', '') == 'K1':
                d.k = k
    e = Engine(ckt, {'max_events': 5_000_000, 'chunk': 1e-6})
    solver = PopSolver(e, dict(ana.get('pop', {})))
    pop = solver.solve()
    assert pop.ok, f'POP failed: {pop.diag}'
    e.restore(solver.snap0)
    e.x = pop.x.copy()
    t0 = e.t
    s, _ = e.run(t0 + 2 * pop.period, sample_dt=pop.period / _NPT,
                 probes=['VOUT', 'ILP', 'ILS', 'VSW', 'VDS', 'VCL', 'IIN'])
    per = [v for t, v in s if t >= t0 + pop.period - 1e-15]
    ilp = np.array([v['ILP'] for v in per])
    ils = np.array([v['ILS'] for v in per])
    return {
        'ckt': ckt, 'T': pop.period, 'res': pop.residual,
        'maxlam': pop.max_multiplier,
        'snap': solver.snap0, 'x': pop.x.copy(),
        'vout': float(np.mean([v['VOUT'] for v in per])),
        'ilp': ilp, 'ils': ils,
        'vsw': np.array([v['VSW'] for v in per]),
        'vds': np.array([v['VDS'] for v in per]),
        'vcl': float(np.mean([v['VCL'] for v in per])),
        'iin': float(np.mean([v['IIN'] for v in per])),
    }


@lru_cache(maxsize=None)
def _base():
    return _orbit()


@lru_cache(maxsize=None)
def _leaky():
    return _orbit(k=0.95)


def _find(ckt, name):
    return next(d for d in ckt.devices if getattr(d, 'name', '') == name)


# ---------------------------------------------------- DCM + energy transport
def test_flyback_dcm_and_energy():
    o = _base()
    ckt = o['ckt']
    vin = _find(ckt, 'VIN').wave.eval(0.0)
    vref = _find(ckt, 'VREF').wave.eval(0.0)
    r17 = _find(ckt, 'R17').r
    r18 = _find(ckt, 'R18').r
    r1 = _find(ckt, 'R1').r
    vf = _find(ckt, 'D1').vf
    lp = _find(ckt, 'LP').l
    # regulation
    assert abs(o['vout'] - vref * (r17 + r18) / r18) < 0.3, o['vout']
    # DCM: the secondary fully demagnetizes and the trailing idle window
    # (from the last conducting sample to the cycle end) is nonempty with
    # the PRIMARY current ~0 there.  (The idle test cannot be a simple
    # |ILS|<eps mask: the secondary's blocked-phase ring crosses zero
    # mid-ON while i(Lp) is at full ramp.)
    ip = float(o['ilp'].max())
    # 20 mA threshold: above the mA-class blocked-phase ring (which is
    # part of the periodic orbit, tau ~ 52 us), far below conduction
    act = np.where(np.abs(o['ils']) > 0.02)[0]
    last_act = int(act.max())
    n_idle = len(o['ilp']) - 1 - last_act
    assert n_idle > 0.02 * _NPT, n_idle
    assert float(np.max(np.abs(o['ilp'][last_act + 1:]))) < 0.02 * ip, \
        (float(np.max(np.abs(o['ilp'][last_act + 1:]))), ip)
    # magnetizing energy transport: 0.5*Lp*Ip^2/T vs input power (the
    # reflected demag current returns some energy to VIN, so a 12%
    # envelope; output side checked against Pout + diode + clamp)
    pmag = 0.5 * lp * ip ** 2 / o['T']
    pin = vin * abs(o['iin'])
    assert abs(pmag / pin - 1.0) < 0.12, (pmag, pin)
    iout = o['vout'] / r1
    pout = o['vout'] * iout + vf * iout
    pcl = (o['vcl'] - vin) ** 2 / _find(ckt, 'RCL').r
    assert pin > pout + pcl, (pin, pout, pcl)      # losses one-sided
    assert pin / (pout + pcl) < 1.25, (pin, pout, pcl)


# ------------------------------------------- transformer action (dots/turns)
def test_flyback_transformer_action():
    o = _base()
    ckt = o['ckt']
    vin = _find(ckt, 'VIN').wave.eval(0.0)
    kk = _find(ckt, 'K1').k
    vf = _find(ckt, 'D1').vf
    ilp = o['ilp']
    # ON window: i(Lp) above half peak; demag window: |i(Ls)| > 10% pk
    on = ilp > 0.5 * ilp.max()
    demag = np.abs(o['ils']) > 0.1 * np.abs(o['ils']).max()
    vds_on = float(np.mean(o['vds'][on]))
    assert abs(vds_on + kk * vin) / (kk * vin) < 0.02, vds_on
    vds_dm = float(np.mean(o['vds'][demag]))
    assert abs(vds_dm - (o['vout'] + vf)) / (o['vout'] + vf) < 0.02, vds_dm


# ---------------------------------------------------- RCD clamp + trend
def test_flyback_rcd_clamp_and_trend():
    o = _base()
    ckt = o['ckt']
    vin = _find(ckt, 'VIN').wave.eval(0.0)
    vf = _find(ckt, 'D1').vf
    reflected = vin + o['vout'] + vf
    # the clamp rail sits ABOVE the reflected voltage (no magnetizing
    # theft) and bounds the switch-node peak
    assert o['vcl'] > reflected + 2.0, (o['vcl'], reflected)
    assert float(o['vsw'].max()) < o['vcl'] + vf + 3.0, (o['vsw'].max(), o['vcl'])
    # trend: more leakage -> more clamp dissipation (k 0.98 -> 0.95)
    o2 = _leaky()
    pcl1 = (o['vcl'] - vin) ** 2 / _find(ckt, 'RCL').r
    pcl2 = (o2['vcl'] - vin) ** 2 / _find(ckt, 'RCL').r
    assert pcl2 > 1.5 * pcl1, (pcl1, pcl2)
    assert o2['vsw'].max() > o['vsw'].max()          # spike grows too


# ------------------------------------------------ POP + PAC + pseudo-N guard
def test_flyback_pop_pac():
    o = _base()
    assert abs(o['T'] - 1e-5) < 1e-12, o['T']        # N=1, exact T_sw
    assert o['res'] < 2e-5, o['res']
    assert 0.99 < o['maxlam'] < 0.9999, o['maxlam']

    # one-period identity: guards the pseudo-N=2 regression (an earlier
    # EA made the N=1 Newton bounce while N=2 converged to the same
    # period-1 orbit; P(x*) at T_nom must return x*)
    ckt, ana = load_yaml(_M)
    e = Engine(ckt, {'max_events': 5_000_000, 'chunk': 1e-6})
    e.restore(o['snap'])
    e.x = o['x'].copy()
    t0 = e.t
    e.run(t0 + 1.1e-5, stop_on_trigger=True)
    d1 = float(np.max(np.abs(e.x - o['x']) / np.maximum(np.abs(o['x']), 1e-9)))
    assert d1 < 1e-6, d1

    pac = PacSolver(ckt, o['snap'], o['x'], dict(ana.get('pac', {})))
    res = pac.solve()
    val = res['info']['validation']
    assert val['status'] == 'VALIDATED', val
    m = loop_metrics(res['freqs'], res['T'])
    assert m['crossovers'], 'no 0 dB crossing'
    fc = m['crossovers'][0]
    assert 150 < fc['freq'] < 1200, fc
    assert fc['pm_deg'] > 60.0, fc


if __name__ == '__main__':
    test_flyback_dcm_and_energy()
    print('PASS test_flyback_dcm_and_energy')
    test_flyback_transformer_action()
    print('PASS test_flyback_transformer_action')
    test_flyback_rcd_clamp_and_trend()
    print('PASS test_flyback_rcd_clamp_and_trend')
    test_flyback_pop_pac()
    print('PASS test_flyback_pop_pac')
    print('all flyback tests passed')
