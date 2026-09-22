# -*- coding: utf-8 -*-
"""MNA topology compiler: stamp -> DAE -> explicit ODE reduction.

Per discrete topology (switch states + diode segments) we build the
descriptor system

    E zdot + G z = B u(t),      z = [v_nodes; i_V; i_E; i_H; i_L; i_C]

and reduce it to a natural state space with **capacitor voltages and
inductor currents as states** (no chain differentiation, no udot terms):

    xidot = A xi + D u
    z     = S xi + T u        (every MNA unknown reconstructed affinely)

Reduction (index-1 dynamic condensation):

1. State columns `sel` = pivot columns of E (QR column pivoting).  For this
   device set that is one node column per capacitor and one i_L column per
   inductor; E[:, r] is structurally zero for the rest.
2. Joint linear system for the dependent unknowns z_r and the state
   derivatives xidot:

       [ G_alg[:, r]      0        ] [ z_r  ]   [ -G_alg[:, sel]  B_alg ]
       [ G_dyn[:, r]  E_dyn[:, sel]] [ xidot] = [ -G_dyn[:, sel]  B_dyn ] [xi; u]

3. Invert once per topology -> A, D, S, T.

u = independent source values + digital Thevenin output values (digital
outputs are Norton branches whose value flips at scheduled event times, so
they behave as piecewise-constant input channels).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ir import Circuit, GND


@dataclass
class Topology:
    tid: str
    n_x: int
    a_mat: np.ndarray        # A (n_x x n_x): xidot = A xi + D u
    d_mat: np.ndarray        # D (n_x x n_u)
    colmap: Dict[int, Tuple[np.ndarray, np.ndarray]]
    # colmap[j] = (xvec, urow): z[j] = xvec @ xi + urow @ u
    state_names: List[str]
    alg_names: List[str]     # names for ALL z entries
    cond_report: Dict[str, float]


def effective_fidelity(fidelity: str, clamps: List[dict]) -> str:
    """Effective fidelity tier after solver-side resistance clamping
    (review §6): an ORIGINAL_EQUIVALENT model whose device values were
    modified by the solver runs CONDITIONED -- visible, never silent."""
    if clamps and fidelity == 'ORIGINAL_EQUIVALENT':
        return 'CONDITIONED'
    return fidelity


class TopologyCompiler:
    def __init__(self, circuit: Circuit, opt: Optional[dict] = None):
        opt = opt or {}
        self.ckt = circuit
        self.nodes: List[str] = circuit.nodes()
        self.nidx: Dict[str, int] = {n: i for i, n in enumerate(self.nodes)}
        self.n_n = len(self.nodes)

        self.vsrcs = circuit.by_kind('V')
        self.isrcs = circuit.by_kind('I')
        self.vcvss = circuit.by_kind('E')
        self.vccss = circuit.by_kind('G')
        self.ccvss = circuit.by_kind('H')
        self.inds = circuit.by_kind('L')
        self.mutuals = circuit.by_kind('K')
        self.ind_idx = {d.name: i for i, d in enumerate(self.inds)}
        self.caps = circuit.by_kind('C')
        self.sws = circuit.by_kind('SW')
        self.dios = circuit.by_kind('D')
        self.gates = circuit.by_kind('GATE')
        self.cmps = circuit.by_kind('CMP')
        self.srffs = circuit.by_kind('SRFF')

        self.dig_channels: List[Tuple[str, str]] = (
            [(g.name, g.out) for g in self.gates]
            + [(c.name, c.out) for c in self.cmps]
            + [(f.name + ':Q', f.q) for f in self.srffs]
            + [(f.name + ':NQ', f.nq) for f in self.srffs]
        )
        self.channels: List[Tuple[str, str]] = (
            [('V', d.name) for d in self.vsrcs]
            + [('I', d.name) for d in self.isrcs]
            + [('DIG', nm) for nm, _ in self.dig_channels]
        )
        for d in self.dios:
            self.channels.append(('DIG', '__DIO_' + d.name))
        self.chan_idx = {key: i for i, key in enumerate(self.channels)}
        self.n_u = len(self.channels)

        _b = self.n_n
        self.iv_cols = {d.name: _b + i for i, d in enumerate(self.vsrcs)}
        _b += len(self.vsrcs)
        self.ie_cols = {d.name: _b + i for i, d in enumerate(self.vcvss)}
        _b += len(self.vcvss)
        self.ih_cols = {d.name: _b + i for i, d in enumerate(self.ccvss)}
        _b += len(self.ccvss)
        self.il_cols = {d.name: _b + i for i, d in enumerate(self.inds)}
        _b += len(self.inds)
        self.ic_cols = {d.name: _b + i for i, d in enumerate(self.caps)}
        _b += len(self.caps)
        self.n_m = _b
        self.n_a = self.n_n + len(self.vsrcs) + len(self.vcvss) + len(self.ccvss)

        self.il_state = {d.name: i for i, d in enumerate(self.inds)}
        # capacitor STATE is a node voltage; map cap -> (n1, n2) for naming
        self.z_names = ([f"v:{n}" for n in self.nodes]
                        + [f"i:{d.name}" for d in self.vsrcs]
                        + [f"i:{d.name}" for d in self.vcvss]
                        + [f"i:{d.name}" for d in self.ccvss]
                        + [f"iL:{d.name}" for d in self.inds]
                        + [f"iC:{d.name}" for d in self.caps])

        self._cache: Dict[tuple, Topology] = {}
        # explicit resistance clamping (review §6): solver options, not
        # silent constants.  None disables a bound and honors the declared
        # value.  Every clamp is recorded with declared/effective values so
        # reports can downgrade the fidelity tier (effective_fidelity).
        self.min_ron = opt.get('min_ron', 1e-3)        # 1 m floor on ron
        self.max_roff = opt.get('max_roff', 1e8)       # 100 Meg ceiling (SW)
        self.max_roff_diode = opt.get('max_roff_diode', 1e9)   # 1 G (D)
        self.r_eff: Dict[str, Dict[str, float]] = {}
        self.clamps: List[dict] = []
        for d in self.sws:
            self.r_eff[d.name] = {
                'ron': self._clamp(d.name, 'ron', d.ron, self.min_ron,
                                   'min_ron'),
                'roff': self._clamp(d.name, 'roff', d.roff, self.max_roff,
                                    'max_roff')}
        for d in self.dios:
            self.r_eff[d.name] = {
                'ron': self._clamp(d.name, 'ron', d.ron, self.min_ron,
                                   'min_ron'),
                'roff': self._clamp(d.name, 'roff', d.roff,
                                    self.max_roff_diode, 'max_roff_diode')}
        self.regularizations: List[str] = []
        if circuit.gmin:
            self.regularizations.append(
                f"gmin={circuit.gmin:g} added to every node diagonal")
        for c in self.clamps:
            self.regularizations.append(
                f"{c['device']}.{c['param']} {c['declared']:g} -> "
                f"{c['effective']:g} ({c['reason']})")

    def _clamp(self, dev, param, declared, bound, reason) -> float:
        """Apply one explicit bound; record when it bites."""
        if bound is None:
            return declared
        eff = (max(declared, bound) if reason == 'min_ron'
               else min(declared, bound))
        if eff != declared:
            self.clamps.append({'device': dev, 'param': param,
                                'declared': declared, 'effective': eff,
                                'reason': reason})
        return eff

    def _n(self, node) -> int:
        return -1 if node == GND else self.nidx[node]

    def compile(self, sw_states: Dict[str, bool], d_segs: Dict[str, bool]) -> Topology:
        tid = ('SW:' + ','.join(f"{k}={int(v)}" for k, v in sorted(sw_states.items()))
               + '|D:' + ','.join(f"{k}={int(v)}" for k, v in sorted(d_segs.items())))
        if tid in self._cache:
            return self._cache[tid]

        ckt = self.ckt
        m = self.n_m
        Gm = np.zeros((m, m))
        E = np.zeros((m, m))
        B = np.zeros((m, self.n_u))

        def gadd(i, j, val):
            if i >= 0 and j >= 0:
                Gm[i, j] += val

        # extreme switch/diode resistances use the precomputed effective
        # values (explicit solver options, clamps recorded in __init__)
        for d in ckt.by_kind('R'):
            g = 1.0 / d.r
            a, b = self._n(d.n1), self._n(d.n2)
            gadd(a, a, g); gadd(b, b, g); gadd(a, b, -g); gadd(b, a, -g)

        for d in self.sws:
            eff = self.r_eff[d.name]
            g = 1.0 / (eff['ron'] if sw_states[d.name] else eff['roff'])
            a, b = self._n(d.d1), self._n(d.d2)
            gadd(a, a, g); gadd(b, b, g); gadd(a, b, -g); gadd(b, a, -g)
        for d in self.dios:
            a, b = self._n(d.a), self._n(d.k)
            if d_segs[d.name]:
                g = 1.0 / self.r_eff[d.name]['ron']
                j = d.vf * g
                gadd(a, a, g); gadd(b, b, g); gadd(a, b, -g); gadd(b, a, -g)
                ci = self.chan_idx[('DIG', '__DIO_' + d.name)]
                # i_dev(a->k) = g*(va-vk) - g*vf  ->  source pushes g*vf into a
                if a >= 0:
                    B[a, ci] += j
                if b >= 0:
                    B[b, ci] -= j
            else:
                g = 1.0 / self.r_eff[d.name]['roff']
                gadd(a, a, g); gadd(b, b, g); gadd(a, b, -g); gadd(b, a, -g)

        for i in range(self.n_n):
            Gm[i, i] += ckt.gmin

        for i, d in enumerate(self.vsrcs):
            r = self.n_n + i
            gadd(self._n(d.n1), self.iv_cols[d.name], 1.0)
            gadd(self._n(d.n2), self.iv_cols[d.name], -1.0)
            gadd(r, self._n(d.n1), 1.0)
            gadd(r, self._n(d.n2), -1.0)
            B[r, self.chan_idx[('V', d.name)]] += 1.0
        for d in self.isrcs:
            ci = self.chan_idx[('I', d.name)]
            if self._n(d.n1) >= 0:
                B[self._n(d.n1), ci] -= 1.0
            if self._n(d.n2) >= 0:
                B[self._n(d.n2), ci] += 1.0

        for i, d in enumerate(self.vcvss):
            r = self.n_n + len(self.vsrcs) + i
            gadd(self._n(d.n1), self.ie_cols[d.name], 1.0)
            gadd(self._n(d.n2), self.ie_cols[d.name], -1.0)
            gadd(r, self._n(d.n1), 1.0)
            gadd(r, self._n(d.n2), -1.0)
            gadd(r, self._n(d.na), -d.k)
            gadd(r, self._n(d.nb), d.k)
        for d in self.vccss:
            g = d.gm
            gadd(self._n(d.n1), self._n(d.na), g)
            gadd(self._n(d.n1), self._n(d.nb), -g)
            gadd(self._n(d.n2), self._n(d.na), -g)
            gadd(self._n(d.n2), self._n(d.nb), g)
        for i, d in enumerate(self.ccvss):
            r = self.n_n + len(self.vsrcs) + len(self.vcvss) + i
            sd = ckt.find(d.sense_i)
            scol = self.il_cols[sd.name] if sd.kind == 'L' else self.iv_cols[sd.name]
            gadd(self._n(d.n1), self.ih_cols[d.name], 1.0)
            gadd(self._n(d.n2), self.ih_cols[d.name], -1.0)
            gadd(r, self._n(d.n1), 1.0)
            gadd(r, self._n(d.n2), -1.0)
            gadd(r, scol, -d.k)

        # dynamic rows
        nL = len(self.inds)
        for i, d in enumerate(self.inds):
            r = self.n_a + i
            gadd(r, self._n(d.n1), 1.0)
            gadd(r, self._n(d.n2), -1.0)
            E[r, self.il_cols[d.name]] -= d.l
            gadd(self._n(d.n1), self.il_cols[d.name], 1.0)
            gadd(self._n(d.n2), self.il_cols[d.name], -1.0)
        # mutual inductance (plan T4): ONLY the E inductor block changes --
        # v1 = L1 di1/dt + M di2/dt is stamped by adding M to both off-
        # diagonal E entries (symmetric), no extra state is introduced
        # (rank of E is unchanged for |k| < 1, so the SVD reduction gives
        # the same state count).  Dot convention: n1 ends are the dots.
        for d in self.mutuals:
            i, j = self.ind_idx[d.l1], self.ind_idx[d.l2]
            m_ = d.k * np.sqrt(self.inds[i].l * self.inds[j].l)
            E[self.n_a + i, self.il_cols[self.inds[j].name]] -= m_
            E[self.n_a + j, self.il_cols[self.inds[i].name]] -= m_
        for i, d in enumerate(self.caps):
            r = self.n_a + nL + i
            if self._n(d.n1) >= 0:
                E[r, self._n(d.n1)] += d.c
            if self._n(d.n2) >= 0:
                E[r, self._n(d.n2)] -= d.c
            Gm[r, self.ic_cols[d.name]] -= 1.0
            gadd(self._n(d.n1), self.ic_cols[d.name], 1.0)
            gadd(self._n(d.n2), self.ic_cols[d.name], -1.0)

        # digital Nortons + RIN loads
        for ch_name, node in self.dig_channels:
            dev = ch_name.split(':')[0]
            a = self._n(node)
            rout = self._dig_rout(dev, ch_name)
            gadd(a, a, 1.0 / rout)
            if a >= 0:
                B[a, self.chan_idx[('DIG', ch_name)]] += 1.0 / rout
        for d in self.gates:
            for inp in d.inputs:
                a = self._n(inp)
                if a >= 0:
                    Gm[a, a] += 1.0 / d.rin
        for d in self.cmps:
            for inp in (d.inp, d.inn):
                a = self._n(inp)
                if a >= 0:
                    Gm[a, a] += 1e-10
        for d in self.srffs:
            for inp in (d.s, d.r):
                a = self._n(inp)
                if a >= 0:
                    Gm[a, a] += 1.0 / d.rin

        # ------------------------------------------------ reduction
        # SVD pencil reduction of the index-1 DAE  E zdot + G z = B u:
        #   E = U S Vᵀ, z = V y  ->  [S11 0; 0 0] ẏ + [H11 H12; H21 H22] y = UᵀBu
        #   index-1 <=> H22 invertible:
        #     y2 = H22⁻¹ (b2 − H21 y1)
        #     ẏ1 = S11⁻¹ (b1 − H11 y1 − H12 y2)
        #   z = (V1 + V2·P2) y1 + V2·Q2 u      (affine reconstruction)
        # Works for floating capacitors / arbitrary L-C placement (states are
        # linear combinations of node voltages and branch currents).
        n_dyn = nL + len(self.caps)
        if n_dyn == 0:
            raise RuntimeError("SINGULAR_TOPOLOGY: no dynamic elements")
        # row equilibration: scale each equation by the inverse of its largest
        # entry across [G | E | B] (row scaling preserves the solution exactly)
        rs = np.ones(m)
        for i in range(m):
            rmax = max(np.abs(Gm[i]).max(), np.abs(E[i]).max(),
                       np.abs(B[i]).max() if B.size else 0.0)
            if rmax > 0:
                rs[i] = 1.0 / rmax
        Gm = Gm * rs[:, None]
        E = E * rs[:, None]
        B = B * rs[:, None]
        U, Sv, Vt = np.linalg.svd(E)
        tol = (float(Sv[0]) if Sv.size else 0.0) * m * np.finfo(float).eps
        n_s = int(np.sum(Sv > tol)) if tol > 0 else 0
        if n_s < n_dyn:
            raise RuntimeError(
                "SINGULAR_TOPOLOGY: E rank deficient (parallel L or C?) "
                f"rank={n_s} needed={n_dyn} for {tid}")
        V = Vt.T
        H = U.T @ Gm @ V
        Bu = U.T @ B
        H11 = H[:n_s, :n_s]
        H12 = H[:n_s, n_s:]
        H21 = H[n_s:, :n_s]
        H22 = H[n_s:, n_s:]
        b1 = Bu[:n_s]
        b2 = Bu[n_s:]
        try:
            H22lu = np.linalg.solve(H22, np.eye(m - n_s))
        except np.linalg.LinAlgError:
            raise RuntimeError(
                f"SINGULAR_TOPOLOGY: H22 singular (index>1 DAE) for {tid}")
        P2 = H22lu @ (-H21)            # y2 = P2 y1 + Q2 u
        Q2 = H22lu @ b2
        Sinv = np.diag(1.0 / Sv[:n_s])
        A = -Sinv @ (H11 + H12 @ P2)
        D = Sinv @ (b1 - H12 @ Q2)
        Pfull = V[:, :n_s] + V[:, n_s:] @ P2
        Qfull = V[:, n_s:] @ Q2
        colmap = {j: (Pfull[j], Qfull[j]) for j in range(m)}

        state_names = [f"y{i}" for i in range(n_s)]
        Lm = self.inductance_matrix()
        topo = Topology(tid=tid, n_x=n_s, a_mat=A, d_mat=D, colmap=colmap,
                        state_names=state_names, alg_names=self.z_names,
                        cond_report={
                            'cond_H22': float(np.linalg.cond(H22)),
                            # 1.0 for inductor-free circuits (np.linalg.cond
                            # rejects 0x0)
                            'cond_L': float(np.linalg.cond(Lm))
                            if Lm.size else 1.0})
        self._cache[tid] = topo
        return topo

    def inductance_matrix(self) -> np.ndarray:
        """The nL x nL inductance matrix (diagonal L, symmetric M off the
        diagonal).  Exposed for verification (plan T4: symmetric, positive
        definite, k=0 degenerates, k->1 conditioning)."""
        nL = len(self.inds)
        Lm = np.zeros((nL, nL))
        for i, d in enumerate(self.inds):
            Lm[i, i] = d.l
        for d in self.mutuals:
            i, j = self.ind_idx[d.l1], self.ind_idx[d.l2]
            m_ = d.k * np.sqrt(self.inds[i].l * self.inds[j].l)
            Lm[i, j] += m_
            Lm[j, i] += m_
        return Lm

    def _dig_rout(self, dev, ch_name):
        for d in self.gates:
            if d.name == dev:
                return d.rout
        for d in self.cmps:
            if d.name == dev:
                return d.rout
        for d in self.srffs:
            if d.name + ':Q' == ch_name or d.name + ':NQ' == ch_name:
                return d.rout
        return 10.0


def _qr_pivot(E):
    from scipy.linalg import qr
    Q, Rk, piv = qr(E, pivoting=True)
    return Q, Rk, piv
