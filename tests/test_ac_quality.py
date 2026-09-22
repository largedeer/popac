# -*- coding: utf-8 -*-
"""AC fit quality regressions (review issue 6).

- the harmonic-aware LSQ must recover a known phasor from a signal
  contaminated with switching ripple over a NON-integer number of
  excitation cycles (integer switching cycles);
- window halves that disagree must gate the point to NOT_SETTLED.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.ac import AcSolver
from popac.engine import SimError




def _mk(harm_k=2, tol_db=0.25, tol_deg=1.5):
    s = AcSolver.__new__(AcSolver)
    s.harm_k = harm_k
    s.win_tol_db = tol_db
    s.win_tol_deg = tol_deg
    s.probe_a = 'a'
    s.probe_b = 'b'
    return s


def test_fit_rejects_switching_ripple():
    s = _mk()
    T_sw = 2.5e-6
    f = 7.3e3                          # NOT commensurate with fsw
    w = 2 * math.pi * f
    ws = 2 * math.pi / T_sw
    A, phi = 0.02, 0.4
    ts = np.arange(0, 120 * T_sw, T_sw / 16)
    def sig(p):
        ph = phi + (0.0 if p == 'a' else math.pi)   # b = -a (loop-like)
        y = A * np.cos(w * ts + ph)
        y += 0.05 * np.cos(ws * ts + 0.7)           # switching ripple LEAKAGE
        y += 0.05 * np.cos(2 * ws * ts)             # 2nd harmonic ripple
        y += 1.234                                  # DC offset
        return y
    samples = [(float(t), {'a': float(sig('a')[i]),
                           'b': float(sig('b')[i])})
               for i, t in enumerate(ts)]
    ph = s._fit(samples, f, T_sw)
    # recovered phasors: a ~ A e^{j phi};  b ~ A e^{j(phi+pi)}
    za, zb = ph['a'], ph['b']
    assert abs(abs(za) - A) / A < 1e-6, f'|a|={abs(za)}'
    assert abs(np.degrees(np.angle(za / (A * np.exp(1j * phi))))) < 1e-4
    assert abs(za + zb) < 1e-9 * A, 'a/b sign relation broken'


def test_fit_folding_guard():
    """f exactly at a switching harmonic folds a sideband onto DC; the fit
    must not go singular."""
    s = _mk()
    T_sw = 2.5e-6
    f = 1.0 / T_sw                     # f == fsw: omega - omega_s == 0
    w = 2 * math.pi * f
    ts = np.arange(0, 60 * T_sw, T_sw / 16)
    samples = [(float(t), {'a': float(np.cos(w * t)),
                           'b': float(-np.cos(w * t))}) for t in ts]
    ph = s._fit(samples, f, T_sw)
    assert abs(abs(ph['a']) - 1.0) < 1e-6


def test_point_status_gates_unsettled():
    s = _mk()
    pt = {'h1': {'a': 1 + 0j, 'b': -1 + 0j},
          'h2': {'a': 1.5 + 0j, 'b': -1.5 + 0j}}      # 3.5 dB apart
    assert s._point_status(pt) == 'NOT_SETTLED'
    pt2 = {'h1': {'a': 1 + 0.01j, 'b': -1 + 0.01j},
           'h2': {'a': 1 + 0.012j, 'b': -1 + 0.012j}}
    assert s._point_status(pt2) == 'PASS'


def test_ac_lti_benchmark():
    """Full production path (engine sine oscillator + sampling + LSQ fit)
    against an analytic LTI response: RC low-pass with the injection source
    upstream, T = -v(a)/v(b) = -(1 + j w R C).  Review P2 acceptance bar:
    < 0.05 dB magnitude, < 0.1 deg phase at every test frequency, and the
    half-window consistency gate must report PASS (the circuit is exactly
    linear, so any disagreement is settling/fit error)."""
    from popac.ir import Circuit, Res, Cap, VSrc
    from popac.waveforms import Waveform
    R, C = 1e3, 100e-9
    tau = R * C
    ckt = Circuit()
    ckt.devices += [
        VSrc('V16', 'a', '0', Waveform.sine(0.0, 0.02, 1e3)),
        Res('R1', 'a', 'b', R),
        Cap('C1', 'b', '0', C, ic_v=0.0),
    ]
    opt = {'inject_src': 'V16', 'probe_a': 'a', 'probe_b': 'b',
           'discard_cycles': 300, 'min_cycles': 100,
           'max_cycles': 1_000_000}
    solver = AcSolver(ckt, opt)
    for f in (200.0, 500.0, 1591.55, 5000.0, 50000.0):
        pt = solver.run_point(f)
        exact = -(1.0 + 1j * 2 * math.pi * f * tau)
        err_db = 20 * math.log10(abs(pt['T']) / abs(exact))
        err_deg = abs(np.degrees(np.angle(pt['T'] / exact)))
        assert abs(err_db) < 0.05, f'{f:g} Hz: {err_db * 1000:+.2f} mdB'
        assert err_deg < 0.1, f'{f:g} Hz: {err_deg * 1000:.1f} mdeg'
        assert solver._point_status(pt) == 'PASS', \
            f'{f:g} Hz not settled (linear circuit)'


def test_ac_degenerate_probe():
    """A flat probe_b phasor must raise the AC_DEGENERATE SimError (not a
    NameError — review issue 6).  Node b carries an undriven capacitor:
    it stays at exactly 0 V for all time."""
    from popac.ir import Circuit, Res, Cap, VSrc
    from popac.waveforms import Waveform
    ckt = Circuit()
    ckt.devices += [
        VSrc('V16', 'a', '0', Waveform.sine(0.0, 0.02, 1e3)),
        Cap('C2', 'b', '0', 1e-9, ic_v=0.0),
        Res('R2', 'b', '0', 1e6),      # bleeder: pencil needs a resistive path
    ]
    solver = AcSolver(ckt, {'inject_src': 'V16', 'probe_a': 'a',
                            'probe_b': 'b', 'discard_cycles': 20,
                            'min_cycles': 100})
    try:
        solver.run_point(1e3)
    except SimError as exc:
        assert 'AC_DEGENERATE' in str(exc) or 'flat' in str(exc), str(exc)
    else:
        raise AssertionError('degenerate probe_b did not raise SimError')


def test_segmented_metrics_gaps():
    """PM/GM must not interpolate across invalid points (review §7.4):
    PASS points split into contiguous segments, each measured
    independently; a magnitude/phase crossing provably hidden inside an
    invalid gap is flagged MARGIN_UNRESOLVED instead of being silently
    bridged by the old masked interpolation."""
    from popac.ac import segmented_metrics
    freqs = np.array([1e3, 2e3, 3e3, 4e3, 5e3])
    nan = complex(np.nan, np.nan)

    # gap separates a |T|>1 point from a |T|<1 point: a 0 dB crossing is
    # CERTAIN to hide in the gap -> flag, and no fake fc from the flanks
    T = np.array([2 + 0j, nan, nan, nan, 0.5 - 0.1j])
    status = ['PASS', 'FAILED', 'NOT_SETTLED', 'FAILED', 'PASS']
    m = segmented_metrics(freqs, T, status)
    assert m['crossovers'] == [], \
        f'crossover invented across a gap: {m["crossovers"]}'
    assert any(u['kind'] == 'crossover' for u in m['margin_unresolved']), \
        f'hidden 0 dB crossing not flagged: {m["margin_unresolved"]}'

    # same-sign flanks across the gap: no crossing provable, no flag
    T1 = np.array([2 + 0j, nan, nan, nan, 3 - 0.1j])
    m1 = segmented_metrics(freqs, T1, status)
    assert not any(u['kind'] == 'crossover'
                   for u in m1['margin_unresolved']), m1['margin_unresolved']

    # contiguous PASS run: normal log-interpolated crossover inside it
    T2 = np.array([2 + 0j, 0.8 - 0.1j, 0.5 - 0.2j, nan, nan])
    st2 = ['PASS', 'PASS', 'PASS', 'FAILED', 'FAILED']
    m2 = segmented_metrics(freqs, T2, st2)
    assert len(m2['crossovers']) == 1, m2['crossovers']
    assert 1e3 < m2['crossovers'][0]['freq'] < 2e3
    assert not any(u['kind'] == 'crossover'
                   for u in m2['margin_unresolved']), m2['margin_unresolved']

    # phase crossing -180 deg hidden inside a gap -> phase flag
    a_lo = 0.1 * np.exp(1j * math.radians(-170.0))
    a_hi = 0.1 * np.exp(1j * math.radians(170.0))     # = -190 deg
    T3 = np.array([a_lo, nan, nan, a_hi, nan])
    st3 = ['PASS', 'FAILED', 'FAILED', 'PASS', 'FAILED']
    m3 = segmented_metrics(freqs, T3, st3)
    assert any(u['kind'] == 'phase_180' for u in m3['margin_unresolved']), \
        m3['margin_unresolved']

    # within-segment -180 deg crossing: GM at the flanking point
    T4 = np.array([0.1 * np.exp(1j * math.radians(a))
                   for a in (-170.0, -190.0, -200.0)] + [nan, nan])
    st4 = ['PASS', 'PASS', 'PASS', 'FAILED', 'FAILED']
    m4 = segmented_metrics(freqs, T4, st4)
    assert m4['gain_margin_db'] is not None and \
        abs(m4['gain_margin_db'] - 20.0) < 1e-9, m4['gain_margin_db']
    assert m4.get('gm_at_freq') == 1e3


def test_loop_metrics_ignores_nan_points():
    """A NaN T point (PAC at a folded switching harmonic) must not
    fabricate a spurious 0 dB crossover with an NaN frequency."""
    from popac.ac import loop_metrics
    freqs = np.array([1e3, 3e3, 1e4, 3e4], float)
    T = np.array([2 + 0j, np.nan + np.nan * 1j, 0.5 - 0.1j, 0.5 - 0.3j])
    m = loop_metrics(freqs, T)
    assert len(m['crossovers']) == 1, m['crossovers']
    assert np.isfinite(m['crossovers'][0]['freq'])
    assert 3e3 < m['crossovers'][0]['freq'] < 1e4


if __name__ == '__main__':
    test_fit_rejects_switching_ripple()
    print('PASS test_fit_rejects_switching_ripple')
    test_fit_folding_guard()
    print('PASS test_fit_folding_guard')
    test_point_status_gates_unsettled()
    print('PASS test_point_status_gates_unsettled')
    test_segmented_metrics_gaps()
    print('PASS test_segmented_metrics_gaps')
    test_loop_metrics_ignores_nan_points()
    print('PASS test_loop_metrics_ignores_nan_points')
    test_ac_lti_benchmark()
    print('PASS test_ac_lti_benchmark')
    test_ac_degenerate_probe()
    print('PASS test_ac_degenerate_probe')
    print('all ac-quality tests passed')
