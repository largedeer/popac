# -*- coding: utf-8 -*-
"""Phase 1 validation: linear circuits vs analytic solutions (plan §16.1)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.ir import Circuit, Res, Cap, Ind, VSrc
from popac.engine import Engine


def _run(ckt, tmax, dt, probes):
    e = Engine(ckt)
    samples, _ = e.run(tmax, sample_dt=dt, probes=probes)
    return samples


def test_rc_step():
    R, C = 1e3, 1e-6
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'in', '0', __import__('popac.waveforms', fromlist=['Waveform']).Waveform.dc(1.0)),
        Res('R1', 'in', 'out', R),
        Cap('C1', 'out', '0', C, ic_v=0.0),
    ]
    ckt.probes = {'vc': ('v', 'out')}
    s = _run(ckt, 5 * R * C, R * C / 100, ['vc'])
    tau = R * C
    worst = 0.0
    for t, vals in s[1:]:
        exact = 1 - math.exp(-t / tau)
        worst = max(worst, abs(vals['vc'] - exact))
    assert worst < 1e-9, f"RC worst err {worst}"


def test_rlc_step_underdamped():
    R, L, C = 1.0, 1e-3, 1e-6
    ckt = Circuit()
    from popac.waveforms import Waveform
    ckt.devices += [
        VSrc('V1', 'in', '0', Waveform.dc(1.0)),
        Res('R1', 'in', 'a', R),
        Ind('L1', 'a', 'out', L, ic_i=0.0),
        Cap('C1', 'out', '0', C, ic_v=0.0),
    ]
    ckt.probes = {'vc': ('v', 'out'), 'il': ('i', 'L1')}
    T = 2 * math.pi / math.sqrt(1 / (L * C))
    s = _run(ckt, 5 * T, T / 200, ['vc', 'il'])
    alpha = R / (2 * L)
    w0 = 1 / math.sqrt(L * C)
    wd = math.sqrt(w0 * w0 - alpha * alpha)
    worst = 0.0
    for t, vals in s[1:]:
        exact = 1 - math.exp(-alpha * t) * (math.cos(wd * t) + alpha / wd * math.sin(wd * t))
        worst = max(worst, abs(vals['vc'] - exact))
    assert worst < 1e-9, f"RLC worst err {worst}"


def test_vcvs_gear():
    """VCVS buffer x2 sanity (with a small cap so the engine has a state)."""
    from popac.ir import Vcvs
    from popac.waveforms import Waveform
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'in', '0', Waveform.dc(2.0)),
        Vcvs('E1', 'mid', '0', 'in', '0', 2.0),      # v(mid)=2*v(in)=4
        Res('R0', 'mid', 'out', 1.0),                # output resistance (index-1)
        Res('R1', 'out', '0', 100.0),
        Cap('C1', 'out', '0', 1e-12, ic_v=0.0),
    ]
    ckt.probes = {'vout': ('v', 'out')}
    e = Engine(ckt)
    samples, _ = e.run(1e-7, sample_dt=1e-8, probes=['vout'])
    v = samples[-1][1]['vout']
    assert abs(v - 4.0 * 100.0 / 101.0) < 1e-6, f"vout={v}"


def test_ccvs_current_sense():
    from popac.ir import Ccvs
    from popac.waveforms import Waveform
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'in', '0', Waveform.dc(1.0)),
        Res('R1', 'in', 'mid', 1.0),
        Ind('L1', 'mid', '0', 1e-9, ic_i=0.0),      # sense branch, closes the loop
        Ccvs('H1', 'vs', '0', 'L1', 0.5),
        Res('R2', 'vs', '0', 1.0),
    ]
    ckt.probes = {'vs': ('v', 'vs')}
    e = Engine(ckt)
    # at t=0 with il=0 -> vs=0; after L charges (tau~1ns), steady il=1A, vs=0.5
    samples, _ = e.run(1e-6, sample_dt=1e-8, probes=['vs'])
    v_final = samples[-1][1]['vs']
    assert abs(v_final - 0.5) < 1e-6, f"vs={v_final}"


def test_clamp_explicit_and_recorded():
    """Review §6: Ron/Roff clamping must be an explicit solver option with
    per-device declared/effective recording, not a silent constant.  With
    the clamps disabled the declared extreme values are honored; any clamp
    on an ORIGINAL_EQUIVALENT model downgrades the effective fidelity to
    CONDITIONED (visible, not silent)."""
    from popac.ir import VcSwitch, Diode
    from popac.mna import TopologyCompiler, effective_fidelity
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'a', '0',
             __import__('popac.waveforms', fromlist=['Waveform']).Waveform.dc(1.0)),
        VcSwitch('S1', 'a', 'b', 'a', '0',
                 ron=1e-6, roff=1e20, threshold=2.0, hystwd=0.1),
        Res('R1', 'b', '0', 1e3),
        Diode('D1', 'b', '0', vf=1e-3, ron=1e-2, roff=1e12),
    ]
    # defaults: floor 1 m on ron, ceilings 100 Meg (sw) / 1 G (diode) on roff
    tc = TopologyCompiler(ckt)
    assert tc.r_eff['S1']['ron'] == 1e-3 and tc.r_eff['S1']['roff'] == 1e8
    assert tc.r_eff['D1']['ron'] == 1e-2 and tc.r_eff['D1']['roff'] == 1e9
    rec = {(c['device'], c['param']): c for c in tc.clamps}
    assert set(rec) == {('S1', 'ron'), ('S1', 'roff'), ('D1', 'roff')}, \
        f'unexpected clamp set: {sorted(rec)}'
    assert rec[('S1', 'roff')]['declared'] == 1e20
    assert rec[('S1', 'roff')]['effective'] == 1e8
    assert rec[('D1', 'roff')]['effective'] == 1e9
    assert effective_fidelity('ORIGINAL_EQUIVALENT', tc.clamps) == 'CONDITIONED'
    assert effective_fidelity('SIMPLIFIED', tc.clamps) == 'SIMPLIFIED'
    assert effective_fidelity('ORIGINAL_EQUIVALENT', []) == 'ORIGINAL_EQUIVALENT'

    # clamps off: declared extremes honored, nothing recorded
    tc2 = TopologyCompiler(ckt, {'min_ron': None, 'max_roff': None,
                                 'max_roff_diode': None})
    assert tc2.clamps == [], tc2.clamps
    assert tc2.r_eff['S1']['ron'] == 1e-6 and tc2.r_eff['S1']['roff'] == 1e20
    assert tc2.r_eff['D1']['ron'] == 1e-2 and tc2.r_eff['D1']['roff'] == 1e12
    assert effective_fidelity('ORIGINAL_EQUIVALENT', tc2.clamps) == \
        'ORIGINAL_EQUIVALENT'


if __name__ == '__main__':
    for fn in [test_rc_step, test_rlc_step_underdamped, test_vcvs_gear,
               test_ccvs_current_sense, test_clamp_explicit_and_recorded]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all linear tests passed")
