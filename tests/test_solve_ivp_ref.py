# -*- coding: utf-8 -*-
"""Stage R2 item 2 (CPP_MIGRATION_AND_TOPOLOGY_PLAN §6/R2): independent
numerical reference for the hybrid engine via scipy solve_ivp.

L0/L4 cross-check: a one-state RC network with a self-switching
hysteretic load (VcSwitch on its own node voltage) is integrated
 BOTH by the engine (exact per-topology matrix exponential + event
 search) AND by a hand-written piecewise solve_ivp reference (adaptive
 RK45 with terminal events, rtol 1e-10) that shares nothing with the
 engine except the circuit equations.  The trajectory comparison pins
 BOTH the state values and the event TIMES: at the switch slopes
 (~350 V/s charging, ~11 kV/s discharging) a 1 us event-time error
 would show as a ~3.5e-4 V trajectory kink -- two orders above the
 asserted 1e-5 V bound, so the bound implicitly locates every event to
 well under 100 ns.

The reference's hysteresis state machine mirrors the engine's SW monitor
(arm/fire at threshold +/- hystwd/2); the load's fast/slow time constants
(91 us vs 1 ms) produce chatter windows around the band during the sine
crest -- the hybrid stress this test exists for.
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.ir import Circuit, Res, Cap, VSrc, VcSwitch
from popac.waveforms import Waveform
from popac.engine import Engine

# circuit parameters (single source of truth for both sides)
F_SINE = 1e3
V_DC, V_AMP = 1.0, 0.5
R1, R2, C1 = 1e3, 100.0, 1e-6
RON, ROFF = 1e-3, 1e6
G_ON = 1.0 / (RON + R2)     # closed: ron + R2 to ground
G_OFF = 1.0 / (ROFF + R2)   # open: roff is HONORED (not ideal) -- the
                            # 1 uA leak shifts vc by ~1e-4 V and phases
                            # the chatter; an ideal-open reference
                            # desyncs by a full band within 20 ms
TH, HYST = 1.0, 0.02
T_END, DT = 20e-3, 2e-6


def _vs(t):
    return V_DC + V_AMP * np.sin(2 * np.pi * F_SINE * t)


def build_circuit():
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'vin', '0', Waveform.sine(V_DC, V_AMP, F_SINE)),
        Res('R1', 'vin', 'vc', R1),
        Cap('C1', 'vc', '0', C1, ic_v=0.0),
        VcSwitch('S1', 'vc', 'r2a', 'vc', '0', RON, ROFF, TH, HYST, 'OPEN'),
        Res('R2', 'r2a', '0', R2),
    ]
    ckt.probes = {'VC': ('v', 'vc')}
    return ckt


def engine_run():
    e = Engine(build_circuit())
    s, _ = e.run(T_END, sample_dt=DT, probes=['VC'])
    t = np.array([p[0] for p in s])
    v = np.array([p[1]['VC'] for p in s])
    return t, v


def reference_run():
    """Piecewise solve_ivp with the same hysteresis state machine."""
    th_hi, th_lo = TH + HYST / 2, TH - HYST / 2
    grid = np.arange(0.0, T_END + DT / 2, DT)
    out = np.empty_like(grid)
    out[0] = 0.0
    t, y, sw = 0.0, [0.0], 0
    i_grid = 1
    while t < T_END - 1e-15:
        g_load = G_ON if sw else G_OFF

        def f(tt, yy, g=g_load):
            return [(_vs(tt) - yy[0]) / (R1 * C1) - yy[0] * g / C1]

        # terminal event: closing (rising past th_hi) or opening (falling
        # past th_lo), matching the engine's armed-direction monitor
        if sw == 0:
            ev = lambda tt, yy: yy[0] - th_hi
            ev.terminal, ev.direction = True, 1.0
        else:
            ev = lambda tt, yy: th_lo - yy[0]
            ev.terminal, ev.direction = True, 1.0
        sol = solve_ivp(f, (t, T_END), y, method='RK45', events=ev,
                        rtol=1e-10, atol=1e-12, dense_output=True)
        t_stop = sol.t[-1] if sol.t_events[0].size else T_END
        while i_grid < len(grid) and grid[i_grid] <= t_stop + 1e-15:
            out[i_grid] = sol.sol(grid[i_grid])[0]
            i_grid += 1
        t = float(sol.t[-1])
        y = [float(sol.y[0, -1])]
        if sol.t_events[0].size:
            sw = 1 - sw
        else:
            break
    out[i_grid:] = out[i_grid - 1] if i_grid else 0.0
    return grid, out


def test_solve_ivp_reference_matches_engine():
    te, ve = engine_run()
    tr, vr = reference_run()
    n = min(len(te), len(tr))
    assert abs(te[-1] - T_END) < 1e-12, f'engine stopped early at {te[-1]}'
    # same sampling grid
    assert np.allclose(te[:n], tr[:n], atol=1e-12), 'sample grid mismatch'
    err = np.max(np.abs(ve[:n] - vr[:n]))
    assert err < 1e-5, f'max |engine - solve_ivp| = {err:.3e} V'
    # the trajectory must actually exercise the switch.  Closures clamp
    # vc AT th_hi (the event fires on the crossing, so samples never
    # exceed it): assert the ceiling sits on the upper band edge, and
    # that openings (falling through th_lo between samples) recur.
    th_hi, th_lo = TH + HYST / 2, TH - HYST / 2
    assert abs(ve.max() - th_hi) < 5e-3, ve.max()
    n_lo = int(np.sum((ve[:-1] > th_lo) & (ve[1:] <= th_lo)))
    assert n_lo >= 5, n_lo


if __name__ == '__main__':
    test_solve_ivp_reference_matches_engine()
    print('PASS test_solve_ivp_reference_matches_engine')
    print('all solve_ivp reference tests passed')
