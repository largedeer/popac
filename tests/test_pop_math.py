# -*- coding: utf-8 -*-
"""POP math regressions (review issues 4 and 5).

- Newton must solve with DP-I: on the slow affine map P(x)=0.99x+1 the
  fixed-point iteration alone needs ~1000 iterations; a correct Newton
  converges in a handful.  With the old plain-DP step it diverges/stalls.
- The Floquet isolation classifier must be structural: a near-unity mode on
  MAIN states is never filtered; an isolated-island mode is.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.pop import PopSolver, _discrete_signature


def _fake_snap(x):
    return {'t': 0.0, 'x': np.asarray(x, float),
            'sw': {}, 'dio': {}, 'cmp': {}, 'srff_q': {},
            'dig': {}, 'src_logic': {}}


class _AffinePop(PopSolver):
    """P(x) = a x + b  with a constant discrete signature."""

    def __init__(self, a, b, n=3):
        self.a = a
        self.b = np.full(n, b)
        self.pre_cycles = 0
        self.max_iter = 25
        self.tol = 1e-6
        self.atol = 1e-9
        self.rtol = 1e-7
        self.n_mult = 1
        self._iso_rows = None

    def _map(self, snap, x, cycles, t_lim):
        x1 = self.a * np.asarray(x, float) + self.b
        return _fake_snap(x1), 1e-6 * cycles


def test_newton_dp_minus_i_slow_map():
    p = _AffinePop(0.99, 1.0)
    snap = _fake_snap(np.zeros(3))
    r = p._solve_N(snap, _discrete_signature(snap), 1e-3, 1)
    # fixed point = b/(1-a) = 100; pure fixed-point iteration from 0 has
    # 0.99^k damping: needs ~2000 iters for 1e-6 — only a correct Newton
    # (DP - I) can land within 25.
    assert r.ok, f'Newton failed on slow affine map: res={r.residual:.3e}'
    assert abs(r.x[0] - 100.0) < 1e-3, f'x*={r.x}'


def test_newton_dp_minus_i_alternating_map():
    p = _AffinePop(-0.9, 1.0)
    snap = _fake_snap(np.zeros(3))
    r = p._solve_N(snap, _discrete_signature(snap), 1e-3, 1)
    assert r.ok, f'Newton failed on alternating map: res={r.residual:.3e}'
    assert abs(r.x[0] - 1.0 / 1.9) < 1e-3


def test_isolated_mode_classification():
    """Synthetic monodromy + reconstruction rows: a 0.9999995 mode on MAIN
    storage must stay; the island mode (mass on isolated rows) is flagged."""
    p = _AffinePop(0.5, 0.0, n=3)
    # pretend states 0,1 are main storage, state 2 is an isolated island cap
    p._iso_rows = (np.array([[0.0, 0.0, 1.0]]),                 # P_iso
                   np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))  # P_main
    J = np.diag([0.9999995, 0.5, 1.0 + 1e-12])   # near-unity MAIN + island
    _, vecs = np.linalg.eig(J)
    mask = p._isolated_modes(None, vecs)
    assert mask is not None
    main_lam = np.abs(np.diag(J))[~mask]
    assert 0.9999995 in main_lam, 'near-unity MAIN mode was filtered out!'
    assert mask[2], 'island mode not classified isolated'


def test_signature_covers_delay_heap_and_source_phase():
    """Review §7.1: the periodic discrete signature must include the pending
    delay heap (channel / target value / remaining time) and the periodic
    sources' segment phase.  Identical continuous+logic states with a
    different pending-event queue are NOT the same boundary (must not be
    judged periodic), while the same queue/phase shifted by exactly one
    period IS (the relative encoding is what makes the signature periodic
    in the first place)."""
    base = {'t': 100e-6, 'x': np.zeros(2), 'sw': {}, 'dio': {}, 'cmp': {},
            'srff_q': {}, 'dig': {}, 'src_logic': {},
            'heap': [(100e-6 + 5e-9, 'U4', 1.0, 41),
                     (100e-6 + 12e-9, 'U10', 0.0, 42)],
            'seg': {'V3': (99.999e-6, 100.001e-6, 0.0, 5e6),
                    'V1': (0.0, float('inf'), 2.5, 0.0)}}

    # same queue/phase shifted by exactly one period T: signature MATCHES
    T = 2.5e-6
    shifted = dict(base)
    shifted['t'] = base['t'] + T
    shifted['heap'] = [(base['t'] + T + 5e-9, 'U4', 1.0, 97),
                       (base['t'] + T + 12e-9, 'U10', 0.0, 98)]
    shifted['seg'] = {'V3': (99.999e-6 + T, 100.001e-6 + T, 0.0, 5e6),
                      'V1': (0.0, float('inf'), 2.5, 0.0)}
    assert _discrete_signature(shifted) == _discrete_signature(base), \
        'one-period shift of the same queue must not change the signature'

    # different remaining time on a pending event: signature DIFFERS
    diff = dict(base)
    diff['heap'] = [(100e-6 + 6e-9, 'U4', 1.0, 41),
                    (100e-6 + 12e-9, 'U10', 0.0, 42)]
    assert _discrete_signature(diff) != _discrete_signature(base), \
        'different delay queue judged periodic (review §7.1 counterexample)'

    # different target value on a pending event: DIFFERS
    diff2 = dict(base)
    diff2['heap'] = [(100e-6 + 5e-9, 'U4', 0.0, 41),
                     (100e-6 + 12e-9, 'U10', 0.0, 42)]
    assert _discrete_signature(diff2) != _discrete_signature(base)

    # different source segment phase (ramp started elsewhere): DIFFERS
    diff3 = dict(base)
    diff3['seg'] = {'V3': (99.9995e-6, 100.0015e-6, 0.0, 5e6),
                    'V1': (0.0, float('inf'), 2.5, 0.0)}
    assert _discrete_signature(diff3) != _discrete_signature(base)


def test_gate_input_dependency_not_isolated():
    """Review round-7 §13.2: the isolation graph must union Gate.inputs
    with Gate.out, so storage influencing a gate input is classified as
    part of whatever the gate's output ultimately switches.  Tested at the
    graph level (_iso_setup walks devices structurally, no transient
    needed): a cap hanging on a gate input whose output controls a switch
    in the trigger component must NOT build an isolated row set.

    Note: in today's engine a purely analog gate input has no runtime
    monitor path (the gate output would stay frozen), and gate inputs in
    runnable circuits are always dig/source nodes already unioned through
    their driving device's own terminals -- so the missing edge is latent
    risk (it bites the day analog gate inputs get monitors), not a live
    misclassification.  The graph must still be explicit."""
    from popac.ir import (Circuit, Res, Cap, VSrc, VcSwitch, Gate,
                               PopTrigger)
    from popac.waveforms import Waveform
    from popac.engine import Engine
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'clk', '0',
             Waveform.pulse(0, 5, 10e-6, 5e-6, 1e-6, 1e-6)),
        Res('R1', 'clk', 'cl', 10e3),
        Cap('C1', 'cl', '0', 2e-9, ic_v=0.0),
        Gate('U1', out='g', inputs=['cl'], fn='BUF',
             vol=0, voh=5, rout=10, rin=1e7, th=2.5, hystwd=0.1, delay=0),
        VSrc('V2', 'vp', '0', Waveform.dc(5.0)),
        VcSwitch('S1', 'vp', 'vm', 'g', '0', 1e2, 1e7, 2.0, 0.1, 'OPEN'),
        Res('RL', 'vm', '0', 1e2),
        Cap('C2', 'vm', '0', 10e-9, ic_v=0.0),
        PopTrigger('X1', node='vm', vref=2.5, edge='rising'),
    ]
    e = Engine(ckt, {'chunk': 1e-7})
    ps = PopSolver(e, {})
    ps.e = e
    setup = ps._iso_setup(e.snapshot())
    # with Gate.inputs unioned, C1 joins the trigger component through
    # cl -> g (gate) -> vp/vm (switch): no isolated storage remains
    assert setup is None, \
        'cap on a gate input built an isolated row set (inputs not unioned)'


if __name__ == '__main__':
    test_newton_dp_minus_i_slow_map()
    print('PASS test_newton_dp_minus_i_slow_map')
    test_newton_dp_minus_i_alternating_map()
    print('PASS test_newton_dp_minus_i_alternating_map')
    test_isolated_mode_classification()
    print('PASS test_isolated_mode_classification')
    test_signature_covers_delay_heap_and_source_phase()
    print('PASS test_signature_covers_delay_heap_and_source_phase')
    test_gate_input_dependency_not_isolated()
    print('PASS test_gate_input_dependency_not_isolated')
    print('all pop-math tests passed')
