# -*- coding: utf-8 -*-
"""Phase 3 validation: comparator, gates, SR latch, delays."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.ir import (Circuit, Res, Cap, VSrc, Comparator, Gate, SrLatch)
from popac.waveforms import Waveform
from popac.engine import Engine


def _add_load(ckt, node):
    """Comparator/gate outputs need at least one dynamic element in the ckt."""
    ckt.devices.append(Cap(f'C_{node}', node, '0', 1e-15, ic_v=0.0))
    ckt.devices.append(Res(f'R_{node}', node, '0', 1e12))


def test_comparator_zero_crossings():
    ckt = Circuit()
    ckt.devices += [
        VSrc('VIN', 'in', '0', Waveform.sine(0, 2.0, 1e6)),
        Comparator('CMP1', 'out', 'in', '0', vol=0, voh=5, rout=10,
                   hystwd=0, delay=2e-9, ic=0),
    ]
    _add_load(ckt, 'out')
    ckt.probes = {'vout': ('v', 'out'), 'vin': ('v', 'in')}
    e = Engine(ckt)
    samples, _ = e.run(3e-6, sample_dt=1e-9, probes=['vout', 'vin'])
    # comparator out should be ~square at 1 MHz, transitions delayed 2 ns
    hi = [s for s in samples if s[1]['vout'] > 2.5]
    lo = [s for s in samples if s[1]['vout'] < 2.5]
    assert len(hi) > 10 and len(lo) > 10
    # measure transition times vs sine zero crossings (rising: sin>0 -> out=5)
    prev = samples[0][1]['vout']
    rises, falls = [], []
    for t, v in samples[1:]:
        cur = v['vout']
        if prev <= 2.5 < cur:
            rises.append(t)
        if prev >= 2.5 > cur:
            falls.append(t)
        prev = cur
    per = 1e-6
    assert len(rises) >= 2 and len(falls) >= 2
    # rising at sine zero-up + 2ns delay: t=0(+2n), 1us(+2n), ...
    for r in rises:
        k = round((r - 2e-9) / (per / 2))
        expected = k * per / 2 + 2e-9
        assert abs(r - expected) < 8e-9, f"rise at {r} expected {expected}"
    for f in falls:
        k = round((f - 2e-9) / (per / 2))
        expected = k * per / 2 + 2e-9
        assert abs(f - expected) < 8e-9, f"fall at {f} expected {expected}"


def test_inv_delay():
    ckt = Circuit()
    ckt.devices += [
        VSrc('VIN', 'in', '0', Waveform.pulse(0, 5, 100e-9, 40e-9,
                                              1e-9, 1e-9, 10e-9)),
        Gate('U1', 'out', ['in'], fn='INV', delay=20e-9),
    ]
    _add_load(ckt, 'out')
    ckt.probes = {'vout': ('v', 'out')}
    e = Engine(ckt)
    samples, _ = e.run(250e-9, sample_dt=1e-9, probes=['vout'])
    seq = [(t, v['vout']) for t, v in samples]
    # transport-delay semantics: initial eval (in=0 -> out=5) fires at 20ns;
    # input high [10,50]ns -> out low [30,70]ns
    def v_at(t):
        best = min(seq, key=lambda s: abs(s[0] - t))
        return best[1]
    assert v_at(5e-9) < 2.5       # initial low until first delayed eval
    assert v_at(25e-9) > 2.5      # init transition landed at 20ns
    assert v_at(40e-9) < 2.5      # went low 20ns after 10ns rise
    assert v_at(60e-9) < 2.5      # still low
    assert v_at(80e-9) > 2.5      # back high 20ns after 50ns fall


def test_sr_latch_set_dominant():
    ckt = Circuit()
    # S: pulse at t=10ns (20ns wide); R: pulse at t=60ns
    ckt.devices += [
        VSrc('VS', 's', '0', Waveform.pulse(0, 5, 1e-6, 20e-9, 1e-9, 1e-9, 10e-9)),
        VSrc('VR', 'r', '0', Waveform.pulse(0, 5, 1e-6, 20e-9, 1e-9, 1e-9, 60e-9)),
        SrLatch('FF1', 'q', 'nq', 's', 'r', th=2.5, ic=0),
    ]
    _add_load(ckt, 'q')
    ckt.probes = {'q': ('v', 'q')}
    e = Engine(ckt)
    samples, _ = e.run(200e-9, sample_dt=1e-9, probes=['q'])

    def q_at(t):
        return min(samples, key=lambda s: abs(s[0] - t))[1]['q']
    assert q_at(5e-9) < 2.5       # reset initially
    assert q_at(40e-9) > 2.5      # set at 10ns
    assert q_at(90e-9) < 2.5      # reset at 60ns
    assert q_at(150e-9) < 2.5     # stays reset


def test_and_or_truth():
    ckt = Circuit()
    ckt.devices += [
        VSrc('VA', 'a', '0', Waveform.pulse(0, 5, 200e-9, 80e-9, 1e-9, 1e-9, 20e-9)),
        VSrc('VB', 'b', '0', Waveform.pulse(0, 5, 200e-9, 80e-9, 1e-9, 1e-9, 60e-9)),
        Gate('UAND', 'yand', ['a', 'b'], fn='AND'),
        Gate('UOR', 'yor', ['a', 'b'], fn='OR'),
    ]
    _add_load(ckt, 'yand')
    _add_load(ckt, 'yor')
    ckt.probes = {'yand': ('v', 'yand'), 'yor': ('v', 'yor'),
                  'a': ('v', 'a'), 'b': ('v', 'b')}
    e = Engine(ckt)
    samples, _ = e.run(400e-9, sample_dt=2e-9,
                       probes=['yand', 'yor', 'a', 'b'])
    for t, v in samples[2:-2]:
        a = v['a'] > 2.5
        b = v['b'] > 2.5
        assert (v['yand'] > 2.5) == (a and b), f"AND wrong at t={t}"
        assert (v['yor'] > 2.5) == (a or b), f"OR wrong at t={t}"


if __name__ == '__main__':
    for fn in [test_comparator_zero_crossings, test_inv_delay,
               test_sr_latch_set_dominant, test_and_or_truth]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all digital tests passed")
