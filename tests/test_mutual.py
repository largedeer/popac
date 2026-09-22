# -*- coding: utf-8 -*-
"""T4 mutual-inductance device (plan §T4): IR + MNA E-block + diagnostics.

Plan verification list, item by item:
- |k| < 1 enforced (validation rejects both polarities)
- inductance matrix symmetric and positive definite (exposed via
  TopologyCompiler.inductance_matrix())
- dot polarity: BOTH solutions verified against the test's own exact
  matrix-exponential solution of the coupled RL step (engine and analytic
  solve the same linear system -> machine-precision agreement); flipping
  k's sign flips the secondary response
- k = 0 degenerates to two independent inductors BITWISE (identical E
  block -> identical reduction -> identical trajectory)
- k -> 1: cond_L diagnostic in Topology.cond_report, (1+k)/(1-k) for the
  equal-L pair, growing without bound
- repeated coupling of one winding rejected at validation; self-coupling,
  missing/non-L targets rejected

The E-block change adds NO state: rank(E) is nL+nC for every |k| < 1
(asserted via the compiled topology's n_x).
"""
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.ir import Circuit, Res, Ind, VSrc, Mutual
from popac.waveforms import Waveform
from popac.engine import Engine
from popac.mna import TopologyCompiler

_L1, _L2 = 1e-3, 4e-3


def _coupled_circuit(k=None, r2=1.0, l1=_L1, l2=_L2):
    """V(step)=1 -- R1=1 -- L1 || L2--R2, coupled with k when given."""
    ckt = Circuit()
    ckt.devices += [
        VSrc('V1', 'vin', '0', Waveform.dc(1.0)),
        Res('R1', 'vin', 'a', 1.0),
        Ind('L1', 'a', '0', l1, ic_i=0.0),
        Ind('L2', 'b', '0', l2, ic_i=0.0),
        Res('R2', 'b', '0', r2),
    ]
    if k is not None:
        ckt.devices.append(Mutual('K1', 'L1', 'L2', k))
    ckt.probes = {'il1': ('i', 'L1'), 'il2': ('i', 'L2'), 'vb': ('v', 'b')}
    return ckt


def _exact_step(l1, l2, k, r1, r2, ts):
    """Exact i1(t), i2(t) of the coupled RL step via the 2x2 matrix
    exponential: [L] di/dt = [1 - r1 i1; -r2 i2] (v_b = -r2 i2 from KCL),
    x(t) = A^-1 (expm(At) - I) b."""
    m = k * np.sqrt(l1 * l2)
    Lm = np.array([[l1, m], [m, l2]])
    Ai = np.linalg.inv(Lm)
    A = Ai @ np.array([[-r1, 0.0], [0.0, -r2]])
    b = Ai @ np.array([1.0, 0.0])
    Ainv = np.linalg.inv(A)
    out = []
    for t in ts:
        x = Ainv @ (expm(A * t) - np.eye(2)) @ b
        out.append(x)
    return np.array(out)


# ----------------------------------------------------------- IR validation
def test_mutual_rejections():
    def errs_of(ckt):
        return '\n'.join(ckt.validate())

    # |k| must be < 1 in both polarities
    for bad in (1.0, -1.0, 1.5):
        e = errs_of(_coupled_circuit(k=bad))
        assert 'must be < 1' in e, (bad, e)
    # self-coupling
    ckt = _coupled_circuit(k=0.5)
    ckt.devices.append(Mutual('K2', 'L1', 'L1', 0.5))
    assert 'cannot couple an inductor to itself' in errs_of(ckt)
    # missing target
    ckt = _coupled_circuit(k=0.5)
    ckt.devices.append(Mutual('K2', 'LX', 'L2', 0.5))
    assert "not found" in errs_of(ckt)
    # non-L target
    ckt = _coupled_circuit(k=0.5)
    ckt.devices.append(Mutual('K2', 'R1', 'L2', 0.5))
    assert 'not an inductor' in errs_of(ckt)
    # repeated coupling of one winding
    ckt = Circuit()
    ckt.devices += [
        Ind('L1', 'a', '0', 1e-3), Ind('L2', 'b', '0', 1e-3),
        Ind('L3', 'c', '0', 1e-3),
        Mutual('K1', 'L1', 'L2', 0.3), Mutual('K2', 'L2', 'L3', 0.3),
    ]
    e = errs_of(ckt)
    assert 'already coupled' in e, e
    # a legal circuit produces no errors
    assert _coupled_circuit(k=0.6).validate() == []


# ------------------------------------- matrix symmetric + positive definite
def test_mutual_matrix_symmetric_pd():
    k = 0.6
    tc = TopologyCompiler(_coupled_circuit(k=k))
    Lm = tc.inductance_matrix()
    assert Lm.shape == (2, 2)
    assert np.array_equal(Lm, Lm.T), Lm                     # symmetric
    assert np.all(np.linalg.eigvalsh(Lm) > 0), Lm           # positive definite
    m_ref = k * np.sqrt(_L1 * _L2)
    assert abs(Lm[0, 1] - m_ref) < 1e-18 and abs(Lm[1, 0] - m_ref) < 1e-18
    # k = 0 degenerates to the diagonal
    L0 = TopologyCompiler(_coupled_circuit(k=0.0)).inductance_matrix()
    assert np.array_equal(L0, np.diag([_L1, _L2])), L0


# --------------------------------------- dot polarity: both exact solutions
def test_mutual_dot_polarity_exact():
    runs = {}
    for k in (0.6, -0.6):
        ckt = _coupled_circuit(k=k)
        e = Engine(ckt)
        dt = 3e-3 / 49
        s, _ = e.run(3e-3, sample_dt=dt, probes=['il1', 'il2'])
        ts = np.array([t for t, _ in s])
        ref = _exact_step(_L1, _L2, k, 1.0, 1.0, ts)
        got = np.array([[v['il1'], v['il2']] for _, v in s])
        assert got.shape == ref.shape, (got.shape, ref.shape)
        err = np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12))
        assert err < 1e-9, f'k={k}: worst rel err {err:g}'
        runs[k] = got
    # the two dot polarities are exact mirrors (the sign of k flips the
    # secondary response and leaves the primary symmetric part unchanged)
    assert np.allclose(runs[0.6][:, 1], -runs[-0.6][:, 1], atol=1e-15, rtol=0)
    assert np.allclose(runs[0.6][:, 0], runs[-0.6][:, 0], atol=1e-15, rtol=0)


# ----------------------------------------------- k=0 bitwise degeneration
def test_mutual_k0_bitwise_degenerate():
    ts = 2e-3
    s_ref, _ = Engine(_coupled_circuit(k=None)).run(
        ts, sample_dt=ts / 50, probes=['il1', 'il2', 'vb'])
    s_k0, _ = Engine(_coupled_circuit(k=0.0)).run(
        ts, sample_dt=ts / 50, probes=['il1', 'il2', 'vb'])
    assert len(s_ref) == len(s_k0)
    for (t1, v1), (t2, v2) in zip(s_ref, s_k0):
        assert t1 == t2
        for key in v1:
            assert v1[key] == v2[key], (key, t1, v1[key], v2[key])


# ------------------------------------------------ k->1 condition diagnostic
def test_mutual_cond_diagnostic():
    # equal-L pair: cond_L = (1+k)/(1-k) exactly
    for k in (0.0, 0.9, 0.99):
        ckt = _coupled_circuit(k=k, l1=1e-3, l2=1e-3)
        topo = TopologyCompiler(ckt).compile({}, {})
        cond = topo.cond_report['cond_L']
        want = (1 + k) / (1 - k) if k > 0 else 1.0
        assert abs(cond - want) / want < 1e-12, (k, cond, want)
        # mutuals add no state: n_x stays nL + nC (2 + 0 caps here... the
        # circuit has no caps; n_x must equal the inductor count)
        assert topo.n_x == 2, topo.n_x
    # the diagnostic is present for ordinary compiles too
    ckt = _coupled_circuit(k=0.5)
    topo = TopologyCompiler(ckt).compile({}, {})
    assert 'cond_L' in topo.cond_report


if __name__ == '__main__':
    test_mutual_rejections()
    print('PASS test_mutual_rejections')
    test_mutual_matrix_symmetric_pd()
    print('PASS test_mutual_matrix_symmetric_pd')
    test_mutual_dot_polarity_exact()
    print('PASS test_mutual_dot_polarity_exact')
    test_mutual_k0_bitwise_degenerate()
    print('PASS test_mutual_k0_bitwise_degenerate')
    test_mutual_cond_diagnostic()
    print('PASS test_mutual_cond_diagnostic')
    print('all mutual-inductance tests passed')
