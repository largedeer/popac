# -*- coding: utf-8 -*-
"""Phase 2 validation: switches, diodes, dead time, open-loop sync buck."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.ir import (Circuit, Res, Cap, Ind, VSrc, VcSwitch, Diode)
from popac.waveforms import Waveform
from popac.engine import Engine

T = 2.5e-6     # 400 kHz
DEAD = 0.15e-6
TON = 1.0e-6   # duty 0.4


def open_loop_buck(ic_v=4.8):
    ckt = Circuit()
    ckt.devices += [
        VSrc('VIN', 'vin', '0', Waveform.dc(12.0)),
        VcSwitch('S1', 'vin', 'sw', 'gh', '0', 1e-3, 10e6, 2.0, 0.1, 'OPEN'),
        VcSwitch('S4', 'sw', '0', 'gl', '0', 1e-3, 10e6, 2.0, 0.1, 'OPEN'),
        Diode('D5', '0', 'sw', vf=0.001, ron=0.01, roff=1e9),
        Ind('L1', 'sw', 'vc', 4.7e-6, ic_i=1.0),
        Res('RESR', 'vc', 'vout', 5e-3),
        Cap('C1', 'vc', '0', 88e-6, ic_v=ic_v),
        Res('RL', 'vout', '0', 5.0),
        VSrc('VGH', 'gh', '0', Waveform.pulse(0, 5, T, TON, 2.5e-9, 2.5e-9, DEAD)),
        VSrc('VGL', 'gl', '0', Waveform.pulse(0, 5, T, TON, 2.5e-9, 2.5e-9,
                                              DEAD + TON + DEAD)),
    ]
    ckt.probes = {'VOUT': ('v', 'vout'), 'SW': ('v', 'sw'), 'IL': ('i', 'L1'),
                  'GH': ('v', 'gh'), 'GL': ('v', 'gl')}
    return ckt


def test_open_loop_buck_steady():
    ckt = open_loop_buck()
    e = Engine(ckt)
    # tau = R*C = 440us -> need > 3 tau to settle: 450 cycles = 1.125ms
    samples, _ = e.run(450 * T, sample_dt=T / 20, probes=['VOUT', 'IL', 'SW'])
    vout = [s[1]['VOUT'] for s in samples[-400:]]
    il = [s[1]['IL'] for s in samples[-400:]]
    vavg = sum(vout) / len(vout)
    iavg = sum(il) / len(il)
    duty = TON / T
    vexp = duty * 12.0
    assert abs(vavg - vexp) < 0.05, f"vavg={vavg} exp={vexp}"
    assert abs(iavg - vexp / 5.0) < 0.05, f"iavg={iavg}"
    # ripple: dIL = (Vin-Vout)*D*T/L
    dil = (12 - vexp) * duty * T / 4.7e-6
    ipp = max(il) - min(il)
    assert abs(ipp - dil) / dil < 0.05, f"ipp={ipp} exp={dil}"


def test_deadtime_diode_conduction():
    ckt = open_loop_buck()
    e = Engine(ckt)
    samples, _ = e.run(60 * T, sample_dt=T / 400, probes=['SW'])
    # find a GH-off window (dead time) and check SW goes below ~ -0.5mV
    sw_min = min(s[1]['SW'] for s in samples[-4000:])
    assert sw_min < -5e-4, f"SW min={sw_min} (diode not conducting?)"


def test_diode_pw_lturnon():
    """Diode in series: ideal except vf; check clamping."""
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'in', '0', Waveform.dc(5.0)),
        Res('R1', 'in', 'a', 100.0),
        Diode('D1', 'a', 'out', vf=0.7, ron=1.0, roff=1e9),
        Res('R2', 'out', '0', 1e3),
        Cap('C1', 'out', '0', 1e-12, ic_v=0.0),
    ]
    ckt.probes = {'vout': ('v', 'out'), 'vd': ('dv', 'a', 'out')}
    e = Engine(ckt)
    samples, _ = e.run(1e-5, sample_dt=1e-7, probes=['vout', 'vd'])
    # steady: i = (5-0.7)/(100+1+1000), vout = i*1k
    iexp = (5.0 - 0.7) / 1101.0
    vout = samples[-1][1]['vout']
    assert abs(vout - iexp * 1000.0) < 1e-3, f"vout={vout}"
    vd = samples[-1][1]['vd']
    assert abs(vd - (0.7 + iexp * 1.0)) < 1e-3, f"vd={vd}"


if __name__ == '__main__':
    for fn in [test_diode_pw_lturnon, test_deadtime_diode_conduction,
               test_open_loop_buck_steady]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all switching tests passed")
