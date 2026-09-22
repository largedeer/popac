# -*- coding: utf-8 -*-
"""Engine edge-case regressions.

test_find_crossing_domain: the crossing-search grid must span [t0, span]
when a dwell offset t0 is given.  The original `tt = i*dt` implementation
sampled [0, span-t0] instead, searching inside the dwell-forbidden region
and firing phantom device flips (PS15: a phantom comparator re-flip
trip+0.5ns that kept the orbit from ever settling and blocked POP).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.ir import Circuit, Res, Cap, VSrc
from popac.waveforms import Waveform
from popac.engine import Engine, SimError


def _engine(chunk=0.5e-9):
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'in', '0', Waveform.dc(1.0)),
        Res('R1', 'in', 'out', 1e3),
        Cap('C1', 'out', '0', 1e-6, ic_v=0.0),
    ]
    return Engine(ckt, {'chunk': chunk})


def test_find_crossing_domain():
    e = _engine(chunk=0.5e-9)
    # zero lies at 0.5 ns, BELOW t0 = 1 ns (inside the dwell window):
    # the search domain [t0, span] contains no zero -> None.  The old grid
    # (tt = i*dt, dt = (span-t0)/n) sampled tau = 0.5 ns and returned it.
    f = lambda tau: tau - 0.5e-9
    t0, span = 1e-9, 10e-9
    hit = e._find_crossing(f, span, f(t0), t0=t0)
    assert hit is None, f"crossing returned {hit} inside dwell-forbidden region"

    # zero inside [t0, span] must be found there
    f2 = lambda tau: tau - 5e-9
    hit2 = e._find_crossing(f2, span, f2(t0), t0=t0)
    assert hit2 is not None and abs(hit2 - 5e-9) < 1e-15, f"hit2={hit2}"

    # no t0: grid still covers [0, span]
    hit3 = e._find_crossing(lambda tau: tau - 7e-9, span, -7e-9)
    assert hit3 is not None and abs(hit3 - 7e-9) < 1e-15, f"hit3={hit3}"


def test_dwell_blocks_refire():
    """A comparator must not re-fire within its delay-dwell even when the
    input would cross back immediately (phantom re-flip regression)."""
    from popac.ir import Comparator
    ckt = Circuit()
    # triangle-ish drive: source ramps up through the comparator threshold,
    # then the topology change forces the input back down quickly
    ckt.devices += [
        VSrc('V1', 'in', '0',
             Waveform.pulse(-5, 5, period=1e-6, width=100e-9,
                            rise=50e-9, fall=1e-9)),
        Res('R1', 'in', 'out', 1e3),
        Cap('C1', 'out', '0', 1e-12, ic_v=0.0),
        Comparator('U1', 'cout', 'out', '0', vol=0, voh=5, rout=10,
                   hystwd=1e-12, delay=10e-9, ic=0),
    ]
    ckt.probes = {'cout': ('v', 'cout')}
    e = Engine(ckt, {'chunk': 1e-9})
    e.run(2e-6)
    # count comparator flips per pulse period: exactly 2 (up at the rising
    # crossing, down after the fall); phantom re-flips would add more
    flips = [t for (t, kind, tag) in e.event_log if tag == 'CMP:U1']
    assert len(flips) == 4, f"expected 4 cmp flips over 2 periods, got {len(flips)}"


def test_crossing_dense_oscillation():
    """Two (or more) zeros inside one grid step must not hide the first
    root: h(t) = sin(2pi*100MHz*t + 0.3), chunk=50ns, span=100ns; the true
    first zero is 4.5225 ns (review issue 1)."""
    import math
    e = _engine(chunk=50e-9)
    f = lambda tau: math.sin(2 * math.pi * 1e8 * tau + 0.3)
    hit = e._find_crossing(f, 100e-9, f(0.0))
    assert hit is not None, 'first zero missed inside grid step'
    assert abs(hit - 4.522535e-9) < 1e-9, f'hit={hit}'


def test_analog_controlled_switch():
    """V -> RC -> switch control must switch the SW on the analog threshold
    with hysteresis (review issue 2)."""
    from popac.ir import VcSwitch
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'in', '0', Waveform.dc(5.0)),
        Res('R1', 'in', 'ctl', 1e3),
        Cap('C1', 'ctl', '0', 1e-6, ic_v=0.0),
        VSrc('V2', 'pv', '0', Waveform.dc(5.0)),
        VcSwitch('S1', 'pv', 'out', 'ctl', '0',
                 ron=1e-3, roff=1e6, threshold=2.0, hystwd=0.1, ic='OPEN'),
        Res('R2', 'out', '0', 1e3),
    ]
    ckt.probes = {'vout': ('v', 'out'), 'ctl': ('v', 'ctl')}
    e = Engine(ckt, {'chunk': 1e-6})
    s, _ = e.run(5e-3, sample_dt=1e-4, probes=['vout', 'ctl'])
    vout_end = s[-1][1]['vout']
    ctl_end = s[-1][1]['ctl']
    assert ctl_end > 4.0, f'control node did not charge: {ctl_end}'
    assert vout_end > 4.0, f'analog-controlled switch never closed: {vout_end}'


def test_falling_trigger():
    """TRIG edge=falling must fire on the falling crossing, not the rising
    one (review issue 3)."""
    from popac.ir import PopTrigger
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'clk', '0',
             Waveform.pulse(0, 5, period=2e-6, width=400e-9,
                            rise=100e-9, fall=100e-9)),
        Res('R1', 'clk', 'out', 1e3),
        Cap('C1', 'out', '0', 1e-12, ic_v=0.0),
        PopTrigger('X1', 'out', vref=2.5, edge='falling'),
    ]
    ckt.probes = {'v': ('v', 'out')}
    e = Engine(ckt, {'chunk': 5e-9})
    _, trig = e.run(2e-6, stop_on_trigger=True)
    assert trig is not None, 'falling trigger never fired'
    # rising crossing ~50 ns, falling crossing ~550 ns
    assert abs(trig - 550e-9) < 50e-9, f'falling trigger fired at {trig}'


def _cmp_circuit(vth, hystwd, chunk, amp=5.0, freq=1e6, opt=None):
    """Source-driven comparator with threshold `vth` on the inn pin."""
    from popac.ir import Comparator
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'in', '0', Waveform.sine(0.0, amp, freq, phase_deg=90.0)),
        VSrc('VTH', 'vth', '0', Waveform.dc(vth)),
        Res('R1', 'in', 'out', 1e3),
        Cap('C1', 'out', '0', 1e-15, ic_v=0.0),
        Comparator('U1', 'cout', 'out', 'vth', vol=0, voh=5, rout=10,
                   hystwd=hystwd, delay=0, ic=0),
    ]
    ckt.probes = {'cout': ('v', 'cout')}
    o = {'chunk': chunk}
    o.update(opt or {})
    return Engine(ckt, o)


def test_narrow_pulse_hidden_pair():
    """A comparator pulse narrower than one grid step (root pair hidden
    between samples) must still be found, not silently missed (review
    issue 1: very narrow pulses).  v = 5cos(2pi*100MHz*t): peaks every
    10 ns, threshold 4.99 V -> 201 ps pulses (10 mV overdrive, above the
    eig-propagation noise floor).  Run length 103 ns (prime to the 3 ns
    chunk) so no peak lands on a grid point and every pulse must be found
    by the hidden-pair hunt, not the coarse scan.

    The pair straddling t=0 (width ~92 ps) is BELOW the engine's declared
    event bandwidth (dt/16 = 184 ps): it is legitimately dropped, but the
    drop must be counted in _cross_unresolved, never silent."""
    import math
    e = _cmp_circuit(4.99, 1e-12, chunk=3e-9, freq=1e8)
    e.run(103e-9)
    c = math.acos(4.99 / 5.0) / (2 * math.pi * 1e8)      # half width ~100.7ps
    ev = [t for (t, kind, tag) in e.event_log if tag == 'CMP:U1']
    assert len(ev) == 20, (
        f'expected 20 flips (10 hidden narrow pulses), got {len(ev)} '
        f'(unresolved={getattr(e, "_cross_unresolved", "n/a")})')
    for k in range(10):
        up, down = ev[2 * k], ev[2 * k + 1]
        assert abs(up - (10e-9 * (k + 1) - c)) < 0.02e-9, f'pulse {k} up at {up}'
        assert abs(down - (10e-9 * (k + 1) + c)) < 0.02e-9, f'pulse {k} down at {down}'
    assert getattr(e, '_cross_unresolved', 0) >= 1, \
        'sub-bandwidth t=0 pair must be counted, not silently dropped'


def test_grazing_inside_hysteresis():
    """A graze that stays inside the comparator hysteresis band must be a
    clean no-event (hysteresis is the spec'd behavior); a graze beyond the
    band but narrower than the root-pair depth limit must be COUNTED as
    unresolved, never silently swallowed (review issue 1: grazing)."""
    # (a) peak exceeds the threshold by 0.4 x (HYSTWD/2), both far above
    # the propagation noise floor: monitor function never crosses the
    # +/-hyst/2 boundary -> exactly zero events
    e = _cmp_circuit(5.0 - 4e-6, 20e-6, chunk=3e-9)
    e.run(3e-6)
    n_ev = sum(1 for (_, kind, tag) in e.event_log if tag == 'CMP:U1')
    assert n_ev == 0, f'graze inside hysteresis produced {n_ev} events'

    # (b) sub-noise/sub-resolution pair beyond the band: peak 1 nV over
    # threshold, pair half-width ~10 ps << dt/16.  Run terminates; a miss
    # (if any) is accounted in _cross_unresolved.
    e3 = _cmp_circuit(5.0 - 1e-9, 1e-12, chunk=3e-9)
    e3.run(1e-6)
    unresolved = getattr(e3, '_cross_unresolved', 0)
    n_ev3 = sum(1 for (_, kind, tag) in e3.event_log if tag == 'CMP:U1')
    assert n_ev3 in (0, 2), f'unexpected flip count {n_ev3}'
    if n_ev3 == 0:
        assert unresolved > 0, 'sub-resolution pair missed silently'


def test_multi_monitor_same_interval():
    """Two comparators crossing inside the same search interval: the engine
    must fire both, earliest first, at their analytic times (review
    issue 1: multiple monitors in one interval)."""
    from popac.ir import Comparator
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'in', '0',
             Waveform.pulse(0, 5, period=1e-6, width=500e-9,
                            rise=5e-9, fall=5e-9)),      # 1 V/ns ramp
        VSrc('VTH2', 'vth2', '0', Waveform.dc(2.5)),
        VSrc('VTH3', 'vth3', '0', Waveform.dc(3.5)),
        Res('R1', 'in', 'out', 1e3),
        Cap('C1', 'out', '0', 1e-15, ic_v=0.0),
        Comparator('U2V', 'c2', 'out', 'vth2', vol=0, voh=5, rout=10,
                   hystwd=1e-12, delay=0, ic=0),
        Comparator('U3V', 'c3', 'out', 'vth3', vol=0, voh=5, rout=10,
                   hystwd=1e-12, delay=0, ic=0),
    ]
    ckt.probes = {'c2': ('v', 'c2'), 'c3': ('v', 'c3')}
    e = Engine(ckt, {'chunk': 5e-9})   # both crossings inside one chunk
    e.run(1e-6)
    t2 = [t for (t, kind, tag) in e.event_log if tag == 'CMP:U2V']
    t3 = [t for (t, kind, tag) in e.event_log if tag == 'CMP:U3V']
    assert t2 and t3, f'monitors did not both fire: {t2} {t3}'
    # rising crossings: v=2.5 at 2.5 ns, v=3.5 at 3.5 ns (1 V/ns ramp)
    assert abs(t2[0] - 2.5e-9) < 0.2e-9, f'2.5V crossing at {t2[0]}'
    assert abs(t3[0] - 3.5e-9) < 0.2e-9, f'3.5V crossing at {t3[0]}'
    assert t2[0] < t3[0], 'earliest event must fire first'


def test_budget_starvation_resolves_brackets():
    """Under total hunt-budget starvation (event_hunt_budget=0) every
    definite grid-bracketed crossing must STILL be found: brackets resolve
    regardless of budget (resolution self-terminates), only sign-stable
    spans are counted and skipped.  The fired-event set for bracketed
    crossings must not depend on the budget or on interval ordering
    (review P0: no dropped events).  Hidden pairs inside one grid step
    genuinely need the subdivision budget; those stay covered by
    test_narrow_pulse_hidden_pair with the default formula."""
    import math
    # 1 MHz sine, threshold 2 V: crossings are slow, directly bracketed by
    # the 3 ns coarse grid -- no hidden-pair machinery needed
    e = _cmp_circuit(2.0, 1e-12, chunk=3e-9, freq=1e6,
                     opt=dict(event_hunt_budget=0))
    e.run(5e-6)                        # completes: no fatal drop
    ev = [t for (t, kind, tag) in e.event_log if tag == 'CMP:U1']
    assert len(ev) == 11, (
        f'starved hunt lost crossings: {len(ev)} flips '
        f'(report={e.event_report()})')
    # first flip: C1 (ic 0 V) charges through R1 with tau = 1 ps and
    # crosses 2 V at t = tau*ln(5/3) ~ 0.51 ps
    assert ev[0] < 1e-9, f'init charging crossing at {ev[0]}'
    # v = 5cos(2pi*1e6 t): falls through 2 V at acos(0.4)/w = 184.51 ns,
    # rises at T-184.51 ns, alternating each period
    tf = math.acos(0.4) / (2 * math.pi * 1e6)
    T = 1e-6
    expect = []
    for k in range(5):
        expect.append(tf + k * T)          # falling
        expect.append(T - tf + k * T)      # rising
    for got, want in zip(ev[1:], expect):
        assert abs(got - want) < 3e-9, f'crossing at {got}, want {want}'
    assert e._cross_budget == 0, 'bracket was dropped under starvation'


def test_gate_unresolved_policy():
    """The EVENTS_UNRESOLVED gate mechanism (unit): the budget counter is a
    tripwire that can only be set by a regression (brackets are never
    dropped by construction), so its policy handling is tested directly."""
    e = _engine(chunk=1e-9)
    e._cross_budget = 2
    e._cross_subband = 7
    try:
        e._gate_unresolved(0, 0, 0)
        raised = None
    except SimError as ex:
        raised = ex
    assert raised is not None and raised.code == 'EVENTS_UNRESOLVED', \
        'default policy must fail on a bracketed drop'

    e2 = _engine(chunk=1e-9)
    e2.event_unresolved_policy = 'warn'
    e2._cross_subband = 5
    e2._gate_unresolved(0, 0, 0)
    assert any('EVENTS_UNRESOLVED' in w for w in e2.warnings), \
        'warn policy must record a warning'

    e3 = _engine(chunk=1e-9)
    e3.event_unresolved_policy = 'ignore'
    e3._cross_budget = 1
    e3._gate_unresolved(0, 0, 0)
    assert not e3.warnings, 'ignore policy must stay silent'


def test_subband_reported_not_fatal():
    """The sub-bandwidth drop in the narrow-pulse scenario must be reported
    via event_report() but must NOT trip the default 'error' policy (it is
    below the engine's declared event bandwidth, not a missed event)."""
    import math
    e = _cmp_circuit(4.99, 1e-12, chunk=3e-9, freq=1e8)
    e.run(103e-9)                      # default policy: completes
    rep = e.event_report()
    assert rep['unresolved'] == rep['budget'] + rep['subband'], rep
    assert rep['subband'] >= 1, 't=0 sub-band pair must land in subband class'
    assert rep['budget'] == 0, \
        f'unexpected budget exhaustion (formula too small?): {rep}'
    # narrowest unresolved span is a positive width below the grid step
    assert 0 < rep['min_span'] <= 3e-9, rep


def test_budget_exhaustion_leftmost_root():
    """Review round-7 §13.2 P0 counterexample.  Under total budget
    starvation (event_hunt_budget=0) the hidden root pair (0.02, 0.04)
    lives in a WIDE sign-stable interval that gets skipped unexamined;
    the later bracketed root 0.25 then returns as valid.  The search must
    instead return the true leftmost root 0.02 -- retro-resolving the
    skipped spans that precede a found root -- and every return path must
    commit the unresolved counters (the old code lost them on root
    returns, reporting unresolved == 0)."""
    e = _engine(chunk=0.1)
    e.event_hunt_budget = 0
    f = lambda t: (t - 0.02) * (t - 0.04) * (t - 0.25)
    hit = e._find_crossing(f, 0.3, f(0.0))
    assert hit is not None and abs(hit - 0.02) < 1e-9, \
        f'returned {hit} as valid; the leftmost root is 0.02'
    # counters must be committed even though a root was returned
    assert e._cross_budget == 0, \
        'wide unexamined spans must not stand after a successful search'
    assert e._cross_subband >= 1, 'resolved-span accounting lost'


if __name__ == '__main__':
    test_find_crossing_domain()
    print('PASS test_find_crossing_domain')
    test_dwell_blocks_refire()
    print('PASS test_dwell_blocks_refire')
    test_crossing_dense_oscillation()
    print('PASS test_crossing_dense_oscillation')
    test_analog_controlled_switch()
    print('PASS test_analog_controlled_switch')
    test_falling_trigger()
    print('PASS test_falling_trigger')
    test_narrow_pulse_hidden_pair()
    print('PASS test_narrow_pulse_hidden_pair')
    test_grazing_inside_hysteresis()
    print('PASS test_grazing_inside_hysteresis')
    test_multi_monitor_same_interval()
    print('PASS test_multi_monitor_same_interval')
    test_budget_starvation_resolves_brackets()
    print('PASS test_budget_starvation_resolves_brackets')
    test_gate_unresolved_policy()
    print('PASS test_gate_unresolved_policy')
    test_subband_reported_not_fatal()
    print('PASS test_subband_reported_not_fatal')
    test_budget_exhaustion_leftmost_root()
    print('PASS test_budget_exhaustion_leftmost_root')
    print('all engine-edge tests passed')
