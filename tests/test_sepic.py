# -*- coding: utf-8 -*-
"""Fourth-topology regression (CPP_MIGRATION_AND_TOPOLOGY_PLAN T3):
peak-current-mode SEPIC with UNCOUPLED inductors, 12 V -> ~24.5 V / 2 A at
200 kHz, same output point as boost24.

Acceptance (plan T3, all closed-form from model parameters only):

- input/output power balance with the loss inventory
- topology current relations: i(L2) avg = output current (EXACT from the
  cap charge balance + output KCL), i(L1) avg = D/(1-D) * i(L2), duty from
  volt-second balance D = (VOUT+vf)/(VIN+VOUT+vf)
- flying-cap mean voltage = VIN (charge balance)
- volt-second ripple dIL1 = dIL2 = VIN*D*T/L (both inductors charge from
  VIN during ON: L1 from the source, L2 from the cap at VIN)
- POP/PAC VALIDATED
- RHP-zero bandwidth gate COMPUTED, not hardcoded: the 4-state averaged
  model (i1,i2,vc,vo) built from the model's actual parameters gives the
  control-to-output zeros; the lowest RHP zero gates fc < f_RHP/3.
  (No fixed "fc < 8 kHz" universal check -- plan forbids it.)

Model-level physics findings pinned by this file's existence (HANDOFF
section 16): the ideal uncoupled CCM SEPIC has an undamped (u,vc)
exchange resonance (slightly UNSTABLE here, +2.37e4 1/s) that neither
the switch-current loop nor the voltage loop can see -- damped by the
RD/CD network across the flying cap; and switch-leg current sensing
makes the comparator immediate-fire at every pwm edge (algebraic vsense
jump), which PAC's folded-saltation bookkeeping can only carry to
~0.5% -- replaced by the trip-equivalent continuous sum sensing.
"""
import os
import sys
from functools import lru_cache

import numpy as np
import scipy.signal as sig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.model_loader import load_yaml
from popac.engine import Engine
from popac.pop import PopSolver
from popac.pac import PacSolver
from popac.ac import loop_metrics

_M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'models', 'sepic24.yaml')

_NPT = 400      # samples per period on the orbit


def _pop_and_orbit():
    ckt, ana = load_yaml(_M)
    e = Engine(ckt, {'max_events': 5_000_000, 'chunk': 500e-9})
    solver = PopSolver(e, dict(ana.get('pop', {})))
    pop = solver.solve()
    assert pop.ok, f'POP failed: {pop.diag}'
    e.restore(solver.snap0)
    e.x = pop.x.copy()
    t_end = e.t + 2 * pop.period
    s, _ = e.run(t_end, sample_dt=pop.period / _NPT,
                 probes=['VOUT', 'IL1', 'IL2', 'IIN', 'VSW', 'VCS'])
    per = [v for t, v in s if t >= t_end - pop.period - 1e-15]
    il1 = np.array([v['IL1'] for v in per])
    il2 = np.array([v['IL2'] for v in per])
    return {
        'ckt': ckt, 'T': pop.period, 'res': pop.residual,
        'maxlam': pop.max_multiplier,
        'snap': solver.snap0, 'x': pop.x.copy(),
        'vout': float(np.mean([v['VOUT'] for v in per])),
        'il1': il1, 'il2': il2,
        'iin': float(np.mean([v['IIN'] for v in per])),
        'vcp': float(np.mean([v['VSW'] - v['VCS'] for v in per])),
    }


@lru_cache(maxsize=None)
def _orbit():
    return _pop_and_orbit()


def _find(ckt, name):
    return next(d for d in ckt.devices if getattr(d, 'name', '') == name)


def _averaged(ckt, vout):
    """Operating point + duty->output zeros of the 4-state averaged SEPIC
    (i1, i2, vc, vo), all from the model's actual parameters.  DC check:
    the equations admit vc0 = VIN and D = (VOUT+vf)/(VIN+VOUT+vf)."""
    vin = _find(ckt, 'VIN').wave.eval(0.0)
    vf = _find(ckt, 'D1').vf
    l1 = _find(ckt, 'L1').l
    l2 = _find(ckt, 'L2').l
    cp = _find(ckt, 'CP').c
    co = _find(ckt, 'C6').c
    esr = _find(ckt, 'RESR').r
    r = _find(ckt, 'R1').r
    e = vout + vf
    d = e / (vin + e)
    i1 = d / (1 - d) * vout / r
    i2 = vout / r
    isd = i1 + i2
    A = np.array([
        [0, 0, -(1 - d) / l1, -(1 - d) / l1],
        [0, 0, d / l2, -(1 - d) / l2],
        [-(1 - d) / cp, d / cp, 0, 0],
        [(1 - d) / co, (1 - d) / co, 0, -1 / (r * co)]])
    B = np.array([[vin / ((1 - d) * l1)], [vin / ((1 - d) * l2)],
                  [isd / cp], [-isd / co]])
    C = np.array([[0.0, 0.0, 0.0, 1.0]])
    num, den = sig.ss2tf(A, B, C, np.zeros((1, 1)))
    num = np.atleast_1d(np.squeeze(num))
    zeros = np.roots(num)
    rhp = [z for z in zeros if z.real > 0]
    return {'d': d, 'i1': i1, 'i2': i2,
            'f_rhp': min(z.real for z in rhp) / (2 * np.pi) if rhp else np.inf}


# ------------------------------------------- DC laws and topology relations
def test_sepic_dc_and_topology_relations():
    o = _orbit()
    ckt = o['ckt']
    vin = _find(ckt, 'VIN').wave.eval(0.0)
    vref = _find(ckt, 'VREF').wave.eval(0.0)
    r17 = _find(ckt, 'R17').r
    r18 = _find(ckt, 'R18').r
    r1 = _find(ckt, 'R1').r
    vf = _find(ckt, 'D1').vf
    # regulation
    assert abs(o['vout'] - vref * (r17 + r18) / r18) < 0.25, o['vout']
    av = _averaged(ckt, o['vout'])
    # topology current relations (charge balance on the flying cap +
    # output KCL): i(L2) avg = output current, i(L1) = D/(1-D) * i(L2)
    iout = o['vout'] * (1.0 / r1 + 1.0 / (r17 + r18))
    assert abs(o['il2'].mean() - iout) / iout < 0.03, \
        (o['il2'].mean(), iout)
    assert abs(o['il1'].mean() - av['d'] / (1 - av['d']) * o['il2'].mean()) \
        / o['il1'].mean() < 0.04, (o['il1'].mean(), av['d'])
    # flying-cap mean voltage = VIN (charge balance)
    assert abs(o['vcp'] - vin) / vin < 0.02, o['vcp']


# ------------------------------------------------------------- power balance
def test_sepic_power_balance():
    o = _orbit()
    ckt = o['ckt']
    vin = _find(ckt, 'VIN').wave.eval(0.0)
    r1 = _find(ckt, 'R1').r
    r17 = _find(ckt, 'R17').r
    r18 = _find(ckt, 'R18').r
    vf = _find(ckt, 'D1').vf
    # diode block loss = vf * IOUT (avg diode current is the output
    # current); RD damper + ESR + ron ~ 0.7 W more; ripple-mean sampling
    # bias ~1% -> two-sided 8% envelope
    pin = vin * abs(o['iin'])
    iout = o['vout'] * (1.0 / r1 + 1.0 / (r17 + r18))
    ploss = vf * iout + 0.7
    pout = o['vout'] * iout
    assert abs(pin / (pout + ploss) - 1.0) < 0.08, (pin, pout, ploss)


# ------------------------------------------------- volt-second ripple laws
def test_sepic_voltsec_ripple():
    o = _orbit()
    ckt = o['ckt']
    vin = _find(ckt, 'VIN').wave.eval(0.0)
    vf = _find(ckt, 'D1').vf
    l1 = _find(ckt, 'L1').l
    l2 = _find(ckt, 'L2').l
    av = _averaged(ckt, o['vout'])
    d = av['d']
    # both inductors see ~VIN during ON (L1 from the source, L2 from the
    # flying cap charged to VIN): dIL = VIN*D*T/L each
    dil1_pred = vin * d * o['T'] / l1
    dil2_pred = vin * d * o['T'] / l2
    assert abs(np.ptp(o['il1']) - dil1_pred) / dil1_pred < 0.05, \
        (np.ptp(o['il1']), dil1_pred)
    assert abs(np.ptp(o['il2']) - dil2_pred) / dil2_pred < 0.05, \
        (np.ptp(o['il2']), dil2_pred)


# --------------------------------------------- POP + PAC + computed RHP gate
def test_sepic_pop_pac_rhp_gate():
    o = _orbit()
    assert abs(o['T'] - 5e-6) < 2e-9, o['T']      # exact T_sw
    assert o['res'] < 1e-5, o['res']
    assert 0.99 < o['maxlam'] < 0.9995, o['maxlam']   # slow mode, stable

    ckt, ana = load_yaml(_M)
    pac = PacSolver(ckt, o['snap'], o['x'], dict(ana.get('pac', {})))
    res = pac.solve()
    val = res['info']['validation']
    assert val['status'] == 'VALIDATED', val
    m = loop_metrics(res['freqs'], res['T'])
    assert m['crossovers'], 'no 0 dB crossing'
    fc = m['crossovers'][0]
    # the RHP zero gate is COMPUTED from the actual parameters (plan T3:
    # no fixed fc<8k).  The averaged model puts the lowest RHP zero of
    # this design at ~2.9 kHz; the measured crossover sits ~RHP/6.
    av = _averaged(ckt, o['vout'])
    assert 1e3 < av['f_rhp'] < 1e4, av['f_rhp']
    assert fc['freq'] < av['f_rhp'] / 3.0, (fc, av['f_rhp'])
    assert fc['pm_deg'] > 60.0, fc


if __name__ == '__main__':
    test_sepic_dc_and_topology_relations()
    print('PASS test_sepic_dc_and_topology_relations')
    test_sepic_power_balance()
    print('PASS test_sepic_power_balance')
    test_sepic_voltsec_ripple()
    print('PASS test_sepic_voltsec_ripple')
    test_sepic_pop_pac_rhp_gate()
    print('PASS test_sepic_pop_pac_rhp_gate')
    print('all sepic tests passed')
