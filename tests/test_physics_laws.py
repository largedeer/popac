# -*- coding: utf-8 -*-
"""Stage R2 item 1 (CPP_MIGRATION_AND_TOPOLOGY_PLAN §6/R2): physics laws.

Independent of the engine's internal cross-checks (POP residual, FD, Psi):
every assertion here compares a steady-state orbit measurement against a
CLOSED-FORM prediction computed from the model parameters alone.

  buck  (buck5v):          EA integrator -> VOUT = VREF*(divider) exact;
        charge balance IL = VOUT*(1/R1 + 1/(R17+R18)); volt-second ripple
        dIL = (VIN-VOUT)*D*T/L with the async-buck duty including the
        freewheeler drop, D = (VOUT+vf)/(VIN+vf); output ripple bounded
        below by the ESR triangle and above by ESR + cap charge ripple
  boost (boost24):        regulation VOUT = VREF*(R17+R18)/R18; power
        balance VIN*IL = VOUT^2/R1 + losses (diode ~ Iout*vf, ESR, ron);
        volt-second ripple dIL = VIN*D*T/L with D = 1 - VIN/(VOUT+vf)

Tolerances carry ~10x the margins measured in out/physics_sens_r1_final.log
(typically 3-4 significant digits of agreement).  The ripple-mean sampling
has a small systematic bias (~0.4%, triangle mean over a grid), so current
balances get 1.5-2% rather than the raw residuals.

L2-doubling re-verifies dIL ~ 1/L as a scaling law, not just one point.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.model_loader import load_yaml
from popac.engine import Engine
from popac.pop import PopSolver

_M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'models', 'buck5v.yaml')
_MB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'models', 'boost24.yaml')


def _orbit(model, eopt=None, **perturb):
    """POP orbit + one-period means/ripples (physics_sens measurement
    convention: two periods of samples, statistics over the second)."""
    ckt, ana = load_yaml(model)
    for name, val in perturb.items():
        for d in ckt.devices:
            if getattr(d, 'name', '') == name:
                setattr(d, val[0], val[1])
    e = Engine(ckt, {'max_events': 2_000_000, **(eopt or {})})
    solver = PopSolver(e, dict(ana.get('pop', {})))
    pop = solver.solve()
    assert pop.ok, f'POP failed: {pop.diag}'
    e.restore(solver.snap0)
    e.x = pop.x.copy()
    t_end = e.t + 2 * pop.period
    pname = 'IL' if model.endswith('boost24.yaml') else 'IL1'
    s, _ = e.run(t_end, sample_dt=pop.period / 200,
                 probes=[pname, 'VOUT'])
    per = [v for t, v in s if t >= t_end - pop.period - 1e-15]
    return {
        'ckt': ckt, 'T': pop.period, 'maxlam': pop.max_multiplier,
        'vout': float(np.mean([v['VOUT'] for v in per])),
        'il': float(np.mean([v[pname] for v in per])),
        'dil': float(np.ptp([v[pname] for v in per])),
        'dvout': float(np.ptp([v['VOUT'] for v in per])),
    }


def _find(ckt, name):
    return next(d for d in ckt.devices if getattr(d, 'name', '') == name)


# ------------------------------------------------------------- buck: DC laws
def test_buck_dc_laws():
    o = _orbit(_M)
    vin = _find(o['ckt'], 'VIN').wave.eval(0.0)   # 12 V
    vref = _find(o['ckt'], 'VREF').wave.eval(0.0)  # 2.5 V
    r1 = _find(o['ckt'], 'R1').r              # 2.5 ohm load
    r17 = _find(o['ckt'], 'R17').r            # 100k
    r18 = _find(o['ckt'], 'R18').r            # 100k
    l1 = _find(o['ckt'], 'L1').l              # 33 u
    vf = _find(o['ckt'], 'D1').vf             # freewheeler drop
    # regulation: divider feeds VREF -> VOUT = VREF*(R17+R18)/R18 = 5.0
    assert abs(o['vout'] - vref * (r17 + r18) / r18) < 0.05, o['vout']
    # charge balance: the inductor carries load + feedback divider current
    il_pred = o['vout'] * (1.0 / r1 + 1.0 / (r17 + r18))
    assert abs(o['il'] - il_pred) / il_pred < 0.015, (o['il'], il_pred)
    # volt-second balance, async-buck duty including the freewheeler
    # drop: (VIN-VOUT)*D*T = (VOUT+vf)*(1-D)*T -> D = (VOUT+vf)/(VIN+vf)
    d = (o['vout'] + vf) / (vin + vf)
    dil_pred = (vin - o['vout']) * d * o['T'] / l1
    assert abs(o['dil'] - dil_pred) / dil_pred < 0.015, (o['dil'], dil_pred)
    # period is the switching period
    assert abs(o['T'] - 5e-6) < 1e-9, o['T']


# ----------------------------------------------------- buck: ripple bounds
def test_buck_ripple_bounds():
    o = _orbit(_M)
    esr = _find(o['ckt'], 'RESR').r           # Cout ESR, 20 m
    c6 = _find(o['ckt'], 'C6').c              # 220 u
    # below: the ESR triangle dIL*ESR (all ripple current flows through the
    # cap branch; the load takes a negligible share)
    lo = o['dil'] * esr
    # above: ESR triangle + cap charge ripple; p2p(f+g) <= p2p(f)+p2p(g)
    hi = o['dil'] * esr + o['dil'] * o['T'] / (8 * c6)
    assert o['dvout'] > 0.99 * lo, (o['dvout'], lo, hi)
    assert o['dvout'] < 1.01 * hi, (o['dvout'], lo, hi)


# ------------------------------------------------- buck: dIL ~ 1/L scaling
def test_buck_dil_scales_with_l():
    base = _orbit(_M)
    dbl = _orbit(_M, L1=('l', 66e-6))
    ratio = dbl['dil'] / base['dil']
    assert abs(ratio - 0.5) < 0.02, ratio
    # operating point unchanged
    assert abs(dbl['vout'] - base['vout']) < 0.05, (dbl['vout'], base['vout'])
    assert abs(dbl['il'] - base['il']) < 0.02, (dbl['il'], base['il'])


# ----------------------------------------------------------- boost: DC laws
def test_boost_dc_laws():
    o = _orbit(_MB, eopt={'chunk': 500e-9})
    vin = _find(o['ckt'], 'VIN').wave.eval(0.0)      # 12 V
    vref = _find(o['ckt'], 'VREF').wave.eval(0.0)    # 2.5 V
    r17 = _find(o['ckt'], 'R17').r                   # 100k
    r18 = _find(o['ckt'], 'R18').r                   # 11.36k
    r1 = _find(o['ckt'], 'R1').r                     # 12 ohm
    l1 = _find(o['ckt'], 'L1').l
    vf = _find(o['ckt'], 'D1').vf                    # 0.5 V
    # regulation
    assert abs(o['vout'] - vref * (r17 + r18) / r18) < 0.25, o['vout']
    # power balance: VIN*IL = Pout + losses (diode block ~ Iout*vf ~ 1 W,
    # ESR ~ 0.35 W, ron ~ 0.01 W on a 50 W converter); ripple-mean sampling
    # adds ~1% bias, so assert a two-sided 6% envelope
    pin = vin * o['il']
    pout = o['vout'] ** 2 / r1
    assert abs(pin / pout - 1.0) < 0.06, (pin, pout)
    # volt-second balance: D = 1 - VIN/(VOUT+vf) -> dIL = VIN*D*T/L
    d = 1.0 - vin / (o['vout'] + vf)
    dil_pred = vin * d * o['T'] / l1
    assert abs(o['dil'] - dil_pred) / dil_pred < 0.05, (o['dil'], dil_pred)
    assert abs(o['T'] - 5.0e-6) < 2e-9, o['T']


if __name__ == '__main__':
    test_buck_dc_laws()
    print('PASS test_buck_dc_laws')
    test_buck_ripple_bounds()
    print('PASS test_buck_ripple_bounds')
    test_buck_dil_scales_with_l()
    print('PASS test_buck_dil_scales_with_l')
    test_boost_dc_laws()
    print('PASS test_boost_dc_laws')
    print('all physics-law tests passed')
