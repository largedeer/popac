# -*- coding: utf-8 -*-
"""Event-driven PWL transient kernel with exact (matrix-exponential)
propagation.

Between two events the circuit is a fixed linear system driven by
piecewise-affine + sinusoidal inputs, so the augmented state

    w = [x, one, sigma_1..sigma_r, osc_c_1, osc_s_1, ...]

propagates exactly as w(t+h) = expm(Aug*h) @ w.  For each distinct
(topology, segment-pattern, digital-values) combination the eigendecomposition
of Aug is cached, so a propagation is a diagonal scaling plus two matvecs.

Event kinds: SEG (source segment boundary), DELAY (scheduled digital output),
DEV (analog crossing: diode / comparator / trigger / source logic threshold).
"""
import heapq
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ir import Circuit, GND
from .mna import TopologyCompiler
from .waveforms import Segment

EV_SEG, EV_DELAY, EV_DEV = 'SEG', 'DELAY', 'DEV'


class SimError(Exception):
    def __init__(self, code, msg):
        super().__init__(f"[{code}] {msg}")
        self.code = code


@dataclass
class Monitor:
    tag: str
    kind: str           # 'D' | 'CMP' | 'TRIG' | 'SRCLOG'
    row: np.ndarray
    th_hi: float
    th_lo: float
    up: bool = True
    meta: dict = field(default_factory=dict)


class Engine:
    def __init__(self, circuit: Circuit, opt: Optional[dict] = None):
        opt = opt or {}
        self.ckt = circuit
        self.tc = TopologyCompiler(circuit, opt)

        def _numopt(key, default):
            """YAML analysis opts arrive unit-suffixed ('500n')."""
            v = opt.get(key, default)
            if isinstance(v, str):
                from .units import eng
                try:
                    return eng(v)
                except (ValueError, TypeError):
                    return default
            return v

        self.chunk = _numopt('chunk', 50e-9)
        self.max_events = int(_numopt('max_events', 20_000_000))
        self.max_settle = opt.get('max_settle', 500)
        # Unresolved event-search spans split into two classes (review P0):
        #   budget  -- TRIPWIRE: a bracketed (endpoint-sign-changing)
        #              interval dropped without resolution.  The hunt never
        #              does this by construction (brackets resolve
        #              regardless of budget), so a nonzero count means a
        #              regression: policy 'error' (default) fails the run.
        #   subband -- sign-stable spans below the declared dt/16 event
        #              bandwidth (proof-resolution limit); reported via
        #              event_report(), never fatal -- stiff circuits
        #              accumulate ~1e5 of these per POP from propagation-
        #              noise structure that cannot be resolved.
        self.event_unresolved_policy = opt.get('event_unresolved_policy', 'error')
        self.event_hunt_budget = opt.get('event_hunt_budget')   # None: formula
        self.warnings: List[str] = []
        self._cross_unresolved = 0
        self._cross_budget = 0
        self._cross_subband = 0
        self._cross_min_span: Optional[float] = None

        self.src_wave: Dict[str, object] = {}
        for d in circuit.by_kind('V') + circuit.by_kind('I'):
            self.src_wave[d.name] = d.wave
        self.dig_nodes: Dict[str, str] = {}
        for ch, node in self.tc.dig_channels:
            self.dig_nodes[node] = ch
        self.src_out_node: Dict[str, str] = {}
        for d in circuit.by_kind('V'):
            if d.n2 == GND and d.n1 != GND:
                self.src_out_node[d.n1] = d.name

        self.t = 0.0
        self.x = np.zeros(self.tc.n_m)     # placeholder sized on first compile
        self.sw = {d.name: (d.ic.upper() == 'CLOSE') for d in self.tc.sws}
        self.dio = {d.name: False for d in self.tc.dios}
        self.cmp_st = {d.name: d.ic for d in self.tc.cmps}
        self.srff_q = {d.name: d.ic for d in self.tc.srffs}
        self.dig_val: Dict[str, float] = {}
        self._init_dig_values()
        self.src_logic: Dict[str, int] = {}
        self._src_logic_specs: List[dict] = []
        self._heap: List[Tuple[float, str, float, int]] = []
        self._seq = 0
        self._last_flip: Dict[str, float] = {}
        self.seg: Dict[str, Segment] = {}
        for d in circuit.by_kind('V') + circuit.by_kind('I'):
            self.seg[d.name] = d.wave.segment_at(0.0)

        omegas = sorted({sn.omega for w in self.src_wave.values()
                         for sn in w.sines})
        self.omegas = omegas
        self.osc = np.zeros(2 * len(omegas))
        for j in range(len(omegas)):
            self.osc[2 * j] = 1.0

        self._aug_cache: Dict[tuple, tuple] = {}
        self._eig_cache: Dict[int, tuple] = {}
        # monitor-evaluation memos (R3 profile: the two n^2 products in
        # _h dominate; see _cvec/_grow)
        self._c_memo: tuple = (None, None)
        self._g_cache: Dict[tuple, np.ndarray] = {}
        self.event_log: List[Tuple[float, str, str]] = []
        self.nev = 0
        self.dbg = os.environ.get('POPAC_DBG')       # itrace window spec t0:t1
        self.itrace: List[str] = []

        self._wire_logic()
        self._init_state()
        self.trig_armed = False
        for tr in circuit.by_kind('TRIG'):
            v0 = self.probe_now('v', tr.node)
            if tr.edge == 'falling':
                # arm high, fire low
                self.trig_armed = v0 > tr.vref + 1e-3
            else:
                self.trig_armed = v0 < tr.vref - 1e-3

    # ------------------------------------------------------------ digital
    def _dig_levels(self, chan):
        for d in self.tc.gates:
            if d.name == chan:
                return d.voh, d.vol
        for d in self.tc.cmps:
            if d.name == chan:
                return d.voh, d.vol
        for d in self.tc.srffs:
            if d.name + ':Q' == chan:
                return d.voh, d.vol
            if d.name + ':NQ' == chan:
                return d.vol, d.voh
        raise KeyError(chan)

    def _init_dig_values(self):
        for ch, _ in self.tc.dig_channels:
            voh, vol = self._dig_levels(ch)
            if ch.endswith(':Q') and ch[:-2] in {f.name for f in self.tc.srffs}:
                st = self.srff_q.get(ch[:-2], 0)
                self.dig_val[ch] = voh if st else vol
            else:
                self.dig_val[ch] = vol

    def _logic_of_node(self, node: str, th: float) -> Optional[int]:
        if node in self.dig_nodes:
            return 1 if self.dig_val[self.dig_nodes[node]] > th else 0
        if node in self.src_out_node:
            sn = self.src_out_node[node]
            if sn in self.src_logic:
                return self.src_logic[sn]
            return 1 if self.src_wave[sn].eval(self.t) > th else 0
        return None

    def _wire_logic(self):
        need = []
        for g in self.tc.gates:
            for inp in g.inputs:
                if inp not in self.dig_nodes and inp in self.src_out_node:
                    need.append((self.src_out_node[inp], g.th, g.hystwd))
        for f in self.tc.srffs:
            for inp in (f.s, f.r):
                if inp not in self.dig_nodes and inp in self.src_out_node:
                    need.append((self.src_out_node[inp], f.th, f.hystwd))
        for s in self.tc.sws:
            if s.c1 not in self.dig_nodes and s.c1 in self.src_out_node:
                need.append((self.src_out_node[s.c1], s.threshold, s.hystwd))
        for sn, th, hyst in need:
            spec = next((s for s in self._src_logic_specs if s['src'] == sn), None)
            if spec is None:
                self._src_logic_specs.append({'src': sn, 'th': th, 'hyst': hyst})
            else:
                spec['hyst'] = max(spec['hyst'], hyst)
        for spec in self._src_logic_specs:
            self.src_logic[spec['src']] = \
                1 if self.src_wave[spec['src']].eval(0.0) > spec['th'] else 0

    def _update_switches_from_logic(self):
        for s in self.tc.sws:
            val = None
            if s.c1 in self.dig_nodes:
                if s.c2 == GND:
                    val = self.dig_val[self.dig_nodes[s.c1]]
                elif s.c2 in self.dig_nodes:
                    val = (self.dig_val[self.dig_nodes[s.c1]]
                           - self.dig_val[self.dig_nodes[s.c2]])
            elif s.c1 in self.src_out_node:
                sn = self.src_out_node[s.c1]
                val = 5.0 if self.src_logic.get(sn, 0) else 0.0
            if val is None:
                continue
            if val > s.threshold + s.hystwd / 2:
                self.sw[s.name] = True
            elif val < s.threshold - s.hystwd / 2:
                self.sw[s.name] = False

    def _settle(self):
        for _ in range(self.max_settle):
            changed = False
            for f in self.tc.srffs:
                s = self._logic_of_node(f.s, f.th)
                r = self._logic_of_node(f.r, f.th)
                if s is None or r is None:
                    continue
                if s:
                    q = 1
                elif r:
                    q = 0
                else:
                    q = self.srff_q[f.name]
                if q != self.srff_q[f.name]:
                    self.srff_q[f.name] = q
                    voh, vol = self._dig_levels(f.name + ':Q')
                    self._set_dig(f.name + ':Q', voh if q else vol)
                    self._set_dig(f.name + ':NQ', vol if q else voh)
                    changed = True
            for g in self.tc.gates:
                vals = [self._logic_of_node(inp, g.th) for inp in g.inputs]
                if any(v is None for v in vals):
                    continue
                if g.fn == 'BUF':
                    out = vals[0]
                elif g.fn == 'INV':
                    out = 1 - vals[0]
                elif g.fn == 'AND':
                    out = int(all(vals))
                elif g.fn == 'OR':
                    out = int(any(vals))
                else:
                    raise SimError('UNSUPPORTED_DEVICE', f"gate fn {g.fn}")
                target = g.voh if out else g.vol
                if g.delay <= 1e-12:
                    if self.dig_val[g.name] != target:
                        self._set_dig(g.name, target)
                        changed = True
                else:
                    pend_vals = [h[2] for h in self._heap if h[1] == g.name]
                    # only schedule an actual transition: target differs from
                    # the committed value, or an older value is still in flight
                    if (target != self.dig_val[g.name] or pend_vals) \
                            and target not in pend_vals:
                        self._seq += 1
                        heapq.heappush(self._heap,
                                       (self.t + g.delay, g.name, target, self._seq))
            if not changed:
                break
        else:
            raise SimError('EVENT_CHATTER', 'digital settle did not converge')
        self._update_switches_from_logic()

    def _set_dig(self, chan, value):
        self.dig_val[chan] = value

    # -------------------------------------------------------- initial state
    def _u_vector(self, topo) -> np.ndarray:
        u = np.zeros(self.tc.n_u)
        for (kind, name), idx in self.tc.chan_idx.items():
            if kind == 'DIG':
                if name.startswith('__DIO_'):
                    u[idx] = 1.0 if self.dio[name[6:]] else 0.0
                else:
                    u[idx] = self.dig_val[name]
            else:
                u[idx] = self.src_wave[name].eval(self.t)
        return u

    def _init_state(self):
        self._update_switches_from_logic()
        topo = None
        guard = 0
        changed = True
        x = np.zeros(1)
        while changed and guard < 10:
            guard += 1
            changed = False
            topo = self.tc.compile(self.sw, self.dio)
            u0 = self._u_vector(topo)
            # provisional x=0 -> node voltages
            z = self._z_from(topo, np.zeros(topo.n_x), u0)
            for d in self.tc.dios:
                va = z[self.tc._n(d.a)] if d.a != GND else 0.0
                vk = z[self.tc._n(d.k)] if d.k != GND else 0.0
                v = float(va - vk)
                if v > d.vf + 1e-6 and not self.dio[d.name]:
                    self.dio[d.name] = True
                    changed = True
                elif v < d.vf - 1e-6 and self.dio[d.name]:
                    self.dio[d.name] = False
                    changed = True
            for s in self.tc.sws:
                c1_digital = (s.c1 in self.dig_nodes
                              or s.c1 in self.src_out_node)
                if c1_digital and (s.c2 == GND or s.c2 in self.dig_nodes
                                   or s.c2 in self.src_out_node):
                    continue        # logic-driven: handled by _settle
                v1 = z[self.tc._n(s.c1)] if s.c1 != GND else 0.0
                v2 = z[self.tc._n(s.c2)] if s.c2 != GND else 0.0
                vc = float(v1 - v2)
                want = self.sw[s.name]
                if vc > s.threshold + s.hystwd / 2:
                    want = True
                elif vc < s.threshold - s.hystwd / 2:
                    want = False
                if want != self.sw[s.name]:
                    self.sw[s.name] = want
                    changed = True
        topo = self.tc.compile(self.sw, self.dio)
        n = topo.n_x
        x = np.zeros(n)
        u0 = self._u_vector(topo)
        # constraints: cap ICs (voltage across), inductor ICs (branch current)
        A_rows, b_vec = [], []
        for d in self.tc.caps:
            if d.ic_v is None:
                continue
            xa, ua = topo.colmap[self.tc._n(d.n1)] if d.n1 != GND else (None, None)
            xb, ub = topo.colmap[self.tc._n(d.n2)] if d.n2 != GND else (None, None)
            rvec = np.zeros(n)
            uvec = np.zeros(self.tc.n_u)
            if d.n1 != GND:
                rvec += topo.colmap[self.tc._n(d.n1)][0]
                uvec += topo.colmap[self.tc._n(d.n1)][1]
            if d.n2 != GND:
                rvec -= topo.colmap[self.tc._n(d.n2)][0]
                uvec -= topo.colmap[self.tc._n(d.n2)][1]
            A_rows.append(rvec)
            b_vec.append(d.ic_v - uvec @ u0)
        for d in self.tc.inds:
            if d.ic_i is None:
                continue
            xv, uv = topo.colmap[self.tc.il_cols[d.name]]
            A_rows.append(xv.copy())
            b_vec.append(d.ic_i - uv @ u0)
        if A_rows:
            A_m = np.vstack(A_rows)
            sol, *_ = np.linalg.lstsq(A_m, np.asarray(b_vec), rcond=None)
            x = sol
        self.x = x
        self._settle()

    def _z_from(self, topo, x, u):
        z = np.zeros(self.tc.n_m)
        for j, (xv, uv) in topo.colmap.items():
            z[j] = xv @ x + uv @ u
        return z

    # -------------------------------------------------- augmented matrices
    def _build_aug(self, topo):
        # the key must include EVERY source segment value (a, s), not just
        # the ramping ones: a pulse source's high and low plateaus have the
        # same zero slope, and keying on slopes alone made the cached Aug of
        # one plateau (wrong DC const column) serve the other
        segkey = tuple((nm, s.a, s.s) for nm, s in sorted(self.seg.items()))
        digkey = tuple(sorted(self.dig_val.items()))
        diokey = tuple(sorted(self.dio.items()))
        sinkey = tuple(len(self.src_wave[nm].sines) for nm in sorted(self.src_wave))
        key = (topo.tid, segkey, digkey, diokey, sinkey)
        hit = self._aug_cache.get(key)
        if hit is not None:
            return hit
        n_x = topo.n_x
        ramps = [nm for nm, s in sorted(self.seg.items()) if s.s != 0.0]
        n_r = len(ramps)
        n_o = len(self.omegas)
        n_w = n_x + 1 + n_r + 2 * n_o
        Aug = np.zeros((n_w, n_w))
        Aug[:n_x, :n_x] = topo.a_mat
        D = topo.d_mat
        const = np.zeros(n_x)
        for (kind, name), idx in self.tc.chan_idx.items():
            if kind in ('V', 'I'):
                seg = self.seg[name]
                col = D[:, idx]
                const = const + col * seg.a
                if seg.s != 0.0:
                    Aug[:n_x, n_x + 1 + ramps.index(name)] += col * seg.s
            elif kind == 'DIG':
                if name.startswith('__DIO_'):
                    const = const + D[:, idx] * (1.0 if self.dio[name[6:]] else 0.0)
                else:
                    const = const + D[:, idx] * self.dig_val[name]
        Aug[:n_x, n_x] = const
        for (kind, name), idx in self.tc.chan_idx.items():
            if kind not in ('V', 'I'):
                continue
            for sn in self.src_wave[name].sines:
                j = self.omegas.index(sn.omega)
                k_c = n_x + 1 + n_r + 2 * j
                Aug[:n_x, k_c] += D[:, idx] * sn.K * math.sin(sn.phi)
                Aug[:n_x, k_c + 1] += D[:, idx] * sn.K * math.cos(sn.phi)
        for k in range(n_r):
            Aug[n_x + 1 + k, n_x] = 1.0
        for j, om in enumerate(self.omegas):
            k_c = n_x + 1 + n_r + 2 * j
            Aug[k_c, k_c + 1] = -om
            Aug[k_c + 1, k_c] = om
        out = (Aug, {'ramps': ramps, 'n_w': n_w})
        self._aug_cache[key] = out
        return out

    def _eig_store(self, Aug):
        aid = id(Aug)
        if aid in self._eig_cache:
            return self._eig_cache[aid]
        lam, V = np.linalg.eig(Aug)
        condV = np.linalg.cond(V)
        if condV < 1e10:
            entry = ('eig', lam.astype(np.complex128), V, np.linalg.inv(V), condV)
        else:
            entry = ('expm', None, None, None, condV)
        self._eig_cache[aid] = entry
        return entry

    def _w_snapshot(self, layout) -> np.ndarray:
        n_r = len(layout['ramps'])
        w = np.zeros(layout['n_w'])
        n_x = len(self.x)
        w[:n_x] = self.x
        w[n_x] = 1.0
        for k, nm in enumerate(layout['ramps']):
            w[n_x + 1 + k] = self.t - self.seg[nm].t0
        w[n_x + 1 + n_r:] = self.osc
        return w

    def _propagate(self, Aug, w0, h) -> np.ndarray:
        kind, lam, V, Vinv, _ = self._eig_store(Aug)
        if kind == 'eig':
            c = Vinv @ w0.astype(np.complex128)
            return np.real(V @ (np.exp(lam * h) * c))
        from scipy.linalg import expm
        return expm(Aug * h) @ w0

    def _h(self, Aug, w0, row, tau) -> float:
        kind, lam, V, Vinv, _ = self._eig_store(Aug)
        if kind == 'eig':
            c = self._cvec(w0, Vinv)
            e = np.exp(lam * tau)
            g = self._grow(row, V)
            return float(np.real(np.dot(g, e * c)))
        from scipy.linalg import expm
        return float(row @ (expm(Aug * tau) @ w0))

    def _cvec(self, w0, Vinv):
        """V^-1 w0, memoized by array identity: identical for every
        monitor and every tau within one interval's search, so the R3
        profile's dominant n^2 product is computed once per interval
        instead of once per _h call.  Holding the reference keeps the id
        meaningful."""
        ref, c = self._c_memo
        if ref is not w0:
            c = Vinv @ w0.astype(np.complex128)
            self._c_memo = (w0, c)
        return c

    def _grow(self, row, V):
        """V^T row, memoized by (eig basis id, row bytes): identical for
        every tau of one monitor, and monitor rows are VALUE-stable
        across iterations (rebuilt from the same topology).  The eig
        cache never clears, so id(V) stays unique.  Size-capped; the
        cap only costs a recompute, never correctness."""
        key = (id(V), row.tobytes())
        g = self._g_cache.get(key)
        if g is None:
            if len(self._g_cache) >= 512:
                self._g_cache.clear()
            g = V.T @ row.astype(np.complex128)
            self._g_cache[key] = g
        return g

    # ------------------------------------------------------------ rows
    def _row_builder(self, topo, layout):
        n_w = layout['n_w']
        n_x = topo.n_x
        ramps = layout['ramps']
        n_r = len(ramps)
        ch = self.tc.chan_idx

        def r_of(spec):
            row = np.zeros(n_w)
            st = spec[0]
            # (xvec, urow) pair(s)
            if st == 'x':
                row[spec[1]] = 1.0
                return row
            if st == 'v':
                k = self.tc._n(spec[1])
                pair = topo.colmap[k]
                pairs = [pair]
            elif st == 'dv':
                pairs = []
                sgn = []
                if spec[1] != GND:
                    pairs.append(topo.colmap[self.tc._n(spec[1])])
                    sgn.append(1.0)
                if spec[2] != GND:
                    pairs.append(topo.colmap[self.tc._n(spec[2])])
                    sgn.append(-1.0)
            elif st == 'i':
                dev = self.ckt.find(spec[1])
                col = ({'V': self.tc.iv_cols, 'E': self.tc.ie_cols,
                        'H': self.tc.ih_cols, 'L': self.tc.il_cols,
                        'C': self.tc.ic_cols}[dev.kind][dev.name])
                pairs = [topo.colmap[col]]
                sgn = [1.0]
            else:
                raise ValueError(spec)
            if st in ('v', 'i'):
                sgn = [1.0]
            mrow = np.zeros(n_x)
            urow = np.zeros(self.tc.n_u)
            for (xv, uv), s in zip(pairs, sgn):
                mrow += s * xv
                urow += s * uv
            row[:n_x] = mrow
            for (kind, name), idx in ch.items():
                if kind in ('V', 'I'):
                    seg = self.seg[name]
                    row[n_x] += urow[idx] * seg.a
                    if seg.s != 0.0 and name in ramps:
                        row[n_x + 1 + ramps.index(name)] += urow[idx] * seg.s
                    for sn in self.src_wave[name].sines:
                        j = self.omegas.index(sn.omega)
                        k_c = n_x + 1 + n_r + 2 * j
                        row[k_c] += urow[idx] * sn.K * math.sin(sn.phi)
                        row[k_c + 1] += urow[idx] * sn.K * math.cos(sn.phi)
                else:
                    if name.startswith('__DIO_'):
                        row[n_x] += urow[idx] * (1.0 if self.dio[name[6:]] else 0.0)
                    else:
                        row[n_x] += urow[idx] * self.dig_val[name]
            return row
        return r_of

    def _probe_spec(self, pname):
        spec = self.ckt.probes.get(pname)
        return spec if spec is not None else ('v', pname)

    def _active_monitors(self, topo, layout, r_of) -> List[Monitor]:
        mons = []
        for d in self.tc.dios:
            row = r_of(('dv', d.a, d.k))
            on = self.dio[d.name]
            # segment rule: OFF -> ON when forward voltage exceeded;
            # ON -> OFF when the segment current reverses (i_dev < -1pA),
            # i.e. v_a - v_k < vf - 1pA/g_on.  The voltage-only rule would
            # keep the (mathematically consistent, physically invalid)
            # reverse-current ON-segment solution alive.
            if on:
                g_on = 1.0 / max(d.ron, 1e-3)
                mons.append(Monitor(f'D:{d.name}', 'D', row,
                                    1e18,
                                    d.vf - 1e-12 / g_on,
                                    up=False, meta={'dev': d.name}))
            else:
                mons.append(Monitor(f'D:{d.name}', 'D', row,
                                    d.vf + 1e-6,
                                    -1e18,
                                    up=True, meta={'dev': d.name}))
        for c in self.tc.cmps:
            row = r_of(('dv', c.inp, c.inn))
            h = max(c.hystwd, 1e-15)
            st = self.cmp_st[c.name]
            mons.append(Monitor(f'CMP:{c.name}', 'CMP', row,
                                h / 2 if st == 0 else 1e18,      # up threshold
                                -h / 2 if st == 1 else -1e18,    # down threshold
                                up=(st == 0), meta={'dev': c.name}))
        for s in self.tc.sws:
            # analog-controlled switches: digital/source-driven controls are
            # handled by _update_switches_from_logic (fast path); everything
            # else gets a hysteretic threshold monitor (review issue 2)
            c1_digital = (s.c1 in self.dig_nodes
                          or s.c1 in self.src_out_node)
            if c1_digital and (s.c2 == GND or s.c2 in self.dig_nodes
                               or s.c2 in self.src_out_node):
                continue
            row = r_of(('dv', s.c1, s.c2))
            on = self.sw[s.name]
            mons.append(Monitor(f'SW:{s.name}', 'SW', row,
                                s.threshold + s.hystwd / 2 if not on else 1e18,
                                -1e18 if not on else s.threshold - s.hystwd / 2,
                                up=not on, meta={'dev': s.name}))
        for tr in self.ckt.by_kind('TRIG'):
            row = r_of(('v', tr.node))
            band = 1e-3
            mons.append(Monitor('TRIG', 'TRIG', row, tr.vref + band,
                                tr.vref - band,
                                up=self.trig_armed, meta={'edge': tr.edge}))
        return mons

    # --------------------------------------------------------- event search
    def _bisect(self, f, a, fa, b):
        for _ in range(80):
            m = 0.5 * (a + b)
            fm = f(m)
            if fm == 0.0 or (b - a) < 1e-18:
                return m
            if (fa < 0) != (fm < 0):
                b = m
            else:
                a, fa = m, fm
        return 0.5 * (a + b)

    def _hunt_budget(self, n: int, force: bool = False) -> int:
        """Eval budget for one crossing search over n grid intervals.

        Sized empirically (review round-7 §13.2 P0): a full-depth pair
        hunt of one sign-stable grid interval costs ~15 midpoint evals
        (4 halvings down to dt/16), while resolving a bracket costs
        ~log2(n*2^24) evals of subdivision plus bisection.  The old
        16n+64 starved whenever brackets and deep hunts coexisted, which
        left WIDE sign-stable spans unexamined (root-pair risk); the
        factor below leaves headroom for both on the reference models
        (PS15 both fidelities: zero wide skips).  force=True returns the
        formula even under an event_hunt_budget override: the retro-hunt
        backstop must never be starvable (a zero refill would loop
        forever re-skipping the same unexamined spans).
        """
        return 48 * n + 128 if (self.event_hunt_budget is None
                                or force) else self.event_hunt_budget

    def _find_crossing(self, f, span, h0, t0=0.0, fp=None, curv=None):
        """Leftmost zero of f on [t0, span].

        Plain sign-change detection can miss an even number of zeros inside
        one grid step (review issue 1), so sign-stable grid intervals are
        recursively subdivided.  Callers may pass fp (f') and curv (a bound
        on |f''| over the interval, constant): when the smallest sampled
        |f| exceeds curv*(b-a)^2/8, a hidden pair of roots is impossible and
        the subinterval is provably root-free (early exit).
        """
        if h0 == 0.0:
            h0 = -1e-300
        s0 = 1 if h0 > 0 else -1
        n = max(1, min(4096, int(math.ceil((span - t0) / self.chunk))))
        dt = (span - t0) / n
        # coarse grid: explicit samples, locate the first sign change
        pts = [(t0, h0)]
        first_change = None
        for i in range(1, n + 1):
            # grid runs over [t0, span]; a plain i*dt would sample below t0,
            # searching inside the dwell-forbidden region (phantom flips)
            tt = t0 + i * dt
            hh = f(tt)
            pts.append((tt, hh))
            if hh == 0.0 or (1 if hh > 0 else -1) != s0:
                first_change = i
                break
        # The first sign change does NOT bound the earliest event: earlier
        # grid intervals can hide root pairs (an even number of zeros) that
        # precede the bracketed root.  Hunt every interval up to and
        # including the first bracket (its own subdivision yields the
        # leftmost root inside it); intervals after it only hold later
        # roots and are skipped.
        # The envelope test (curv) proves root-freeness when it applies; a
        # bracketed sign change is itself subdivided down to min_step before
        # bisection, so the LEFTMOST root of the span is returned even when
        # a bracket holds several.  The eval budget bounds worst-case cost
        # on stiff systems where the bound is too loose — unresolved spans
        # are counted (engine._cross_unresolved), never silently dropped.
        min_step = max(span * 2.0 ** -24, 1e-18)
        budget = self._hunt_budget(n)
        unresolved = getattr(self, '_cross_unresolved', 0)
        nbud = getattr(self, '_cross_budget', 0)
        nsub = getattr(self, '_cross_subband', 0)
        wmin = getattr(self, '_cross_min_span', None)

        def push(a, fa, b, fb):
            stack.insert(0, (a, fa, b, fb))

        limit = first_change if first_change is not None else len(pts) - 1
        stack = [(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                 for i in range(limit)]
        # Review round-7 §13.2 P0: a sign-stable span skipped at budget
        # exhaustion is only "below bandwidth" when its width is already
        # <= dt/16.  A WIDE skipped span is unexamined and can hide a
        # resolvable root pair; when a root is later found to its RIGHT the
        # leftmost-root guarantee is void, so those spans are retro-hunted
        # with a refilled budget (bounded; each hunt ends at dt/16 or at a
        # root).  Spans that stay unexamined in a root-free search count as
        # budget class (fatal under the default policy).
        pend: List[Tuple[float, float, float, float]] = []
        result = None
        while True:
            while stack:
                a, fa, b, fb = stack.pop(0)      # FIFO: leftmost first
                if budget <= 0:
                    # The budget guards the expensive root-FREE verification
                    # hunts (sign-stable intervals that cannot be proven via
                    # the curvature envelope).  A bracketed pop is a definite
                    # crossing: resolving it self-terminates (subdivision to
                    # min_step, then bisection returns the root and ends the
                    # whole search), so brackets are NEVER dropped for
                    # budget.  This keeps the fired-event set independent of
                    # the budget and of interval ordering.
                    if not (fb == 0.0 or (fa > 0) != (fb > 0)):
                        if (b - a) <= dt / 16.0:
                            unresolved += 1
                            nsub += 1
                            wmin = (b - a if wmin is None
                                    else min(wmin, b - a))
                            continue
                        # wide unexamined span: provisionally budget class
                        pend.append((a, fa, b, fb))
                        unresolved += 1
                        nbud += 1
                        wmin = b - a if wmin is None else min(wmin, b - a)
                        continue
                    # bracketed: fall through and resolve regardless of budget
                if fb == 0.0:                    # grid-point zero (sign test
                    result = b                   # below can't bracket it)
                    break
                bracketed = (fa > 0) != (fb > 0)
                if bracketed and (b - a) <= min_step:
                    result = self._bisect(f, a, fa, b)
                    break
                if not bracketed:
                    # cheap pruning BEFORE any midpoint eval: with |f''|<=curv,
                    # f >= min(fa,fb) - curv*(b-a)^2/8 on [a,b], so endpoints
                    # alone can prove root-freeness
                    if curv is not None and min(abs(fa), abs(fb)) > \
                            curv * (b - a) ** 2 / 8.0:
                        continue                  # provably root-free
                    if (b - a) <= dt / 16.0:
                        # pair-hunting depth exhausted without evidence of a
                        # root; narrower structure is below the engine's
                        # event bandwidth (stiff-mode noise).  Counted,
                        # never silently dropped.
                        unresolved += 1
                        nsub += 1
                        wmin = (b - a if wmin is None
                                else min(wmin, b - a))
                        continue
                m = 0.5 * (a + b)
                fm = f(m)
                budget -= 1
                if fm == 0.0:
                    result = m
                    break
                left_change = (fm > 0) != (fa > 0)
                right_change = (fb > 0) != (fm > 0)
                if left_change or right_change or bracketed:
                    # keep narrowing; left half first so the leftmost wins
                    stack.insert(0, (m, fm, b, fb))
                    stack.insert(0, (a, fa, m, fm))
                    continue
                stack.insert(0, (m, fm, b, fb))
                stack.insert(0, (a, fa, m, fm))
            if not pend:
                break
            # Wide unexamined spans remain (when a root was found they all
            # PRECEDE it, voiding the leftmost guarantee; in a root-free
            # search they could still hide a pair).  Re-hunt them with the
            # formula budget -- the correctness backstop is never starvable
            # by the event_hunt_budget override (a zero refill would loop
            # forever re-skipping the same spans).  Undo their provisional
            # budget-class counts; the re-hunt recounts naturally at its
            # own depth floor.
            stack = list(pend)
            pend = []
            unresolved -= len(stack)
            nbud -= len(stack)
            budget = self._hunt_budget(n, force=True)
            # any root found in the re-hunt precedes `result`; if none is,
            # the original result stands.
            # any root found in the re-hunt precedes `result`; if none is,
            # the original result stands.
        # single exit: counters are committed on EVERY return path
        self._cross_unresolved = unresolved
        self._cross_budget = nbud
        self._cross_subband = nsub
        if wmin is not None:
            self._cross_min_span = (wmin if self._cross_min_span is None
                                    else min(self._cross_min_span, wmin))
        return result

    def _hdot(self, Aug, w0, row, tau) -> float:
        """d/dt of the monitored quantity along the flow (eig path)."""
        kind, lam, V, Vinv, _ = self._eig_store(Aug)
        if kind == 'eig':
            c = Vinv @ w0.astype(np.complex128)
            e = np.exp(lam * tau)
            g = V.T @ row.astype(np.complex128)
            return float(np.real(np.dot(g, lam * e * c)))
        from scipy.linalg import expm
        return float(row @ (Aug @ (expm(Aug * tau) @ w0)))

    def _hdd_bound(self, Aug, w0, row, span) -> Optional[float]:
        """Bound on |d2h/dt2| over [0, span] (None if unavailable).

        Stiff modes make the rigorous bound astronomically large (useless);
        in that case return None so the caller falls back to budgeted
        subdivision instead of a false all-clear."""
        kind, lam, V, Vinv, _ = self._eig_store(Aug)
        if kind == 'eig':
            c = np.abs(self._cvec(w0, Vinv))
            g = np.abs(self._grow(row, V))
            grow = np.exp(np.maximum(np.real(lam), 0.0) * span)
            h_scale = float(np.sum(g * c * grow)) + 1e-300
            B = float(np.sum(g * c * np.abs(lam) ** 2 * grow))
            return B if B < 1e12 * h_scale / max(span, 1e-18) ** 2 else None
        return None

    def _next_seg_boundary(self) -> float:
        nxt = float('inf')
        for seg in self.seg.values():
            if seg.t1 > self.t + 1e-15 and seg.t1 < nxt:
                nxt = seg.t1
        return nxt

    def _advance_segments(self, t):
        # advance with a 1 fs nudge so period boundaries computed on different
        # float paths (k*T + t_end vs (k+1)*T) cannot leave a stale cursor
        for name, seg in self.seg.items():
            guard = 0
            while seg.t1 <= t + 1e-15 and guard < 100000:
                seg = self.src_wave[name].segment_at(max(t, seg.t1) + 1e-15)
                guard += 1
            self.seg[name] = seg

    def _commit_state(self, w1, layout, t_new):
        n_x = len(self.x)
        n_r = len(layout['ramps'])
        self.x = np.real(np.asarray(w1[:n_x]))
        self.osc = np.asarray(w1[n_x + 1 + n_r:], dtype=float)
        self.t = t_new
        self.nev += 1
        if self.nev > self.max_events:
            raise SimError('EVENT_CHATTER', f'exceeded max_events={self.max_events}')

    def _fire_known(self, t_known):
        fired = None
        while self._heap and self._heap[0][0] <= t_known + 1e-18:
            tf, chan, val, _ = heapq.heappop(self._heap)
            self._set_dig(chan, val)
            fired = (tf, EV_DELAY, chan)
        for spec in self._src_logic_specs:
            sn = spec['src']
            v = self.src_wave[sn].eval(self.t)
            st = self.src_logic[sn]
            if st == 0 and v > spec['th'] + spec['hyst'] / 2:
                self.src_logic[sn] = 1
                fired = (self.t, EV_DEV, f'SRCLOG:{sn}')
            elif st == 1 and v < spec['th'] - spec['hyst'] / 2:
                self.src_logic[sn] = 0
                fired = (self.t, EV_DEV, f'SRCLOG:{sn}')
        self._settle()
        if fired is None:
            fired = (t_known, EV_SEG, 'seg')
        self.event_log.append(fired)
        return fired

    def _apply_dev(self, m: Monitor):
        if m.kind == 'D':
            self.dio[m.meta['dev']] = not self.dio[m.meta['dev']]
        elif m.kind == 'CMP':
            c = self.ckt.find(m.meta['dev'])
            new = 1 - self.cmp_st[c.name]
            self.cmp_st[c.name] = new
            val = c.voh if new else c.vol
            if c.delay <= 1e-12:
                self._set_dig(c.name, val)
            else:
                # cancel-and-repush: the output must follow the state within
                # one delay; stale in-flight transitions are dropped
                self._heap = [h for h in self._heap if h[1] != c.name]
                heapq.heapify(self._heap)
                self._seq += 1
                heapq.heappush(self._heap, (self.t + c.delay, c.name, val, self._seq))
        elif m.kind == 'SW':
            # set the TARGET state (armed direction), never blind toggling
            self.sw[m.meta['dev']] = bool(m.up)
        if m.kind in ('D', 'CMP', 'SW'):
            self._last_flip[m.tag] = self.t
        elif m.kind == 'SRCLOG':
            sn = m.meta['src']
            self.src_logic[sn] = 1 - self.src_logic[sn]
        self._settle()
        self.event_log.append((self.t, EV_DEV, m.tag))

    # ------------------------------------------------------------- run
    def event_report(self) -> dict:
        """Unresolved event-search span counts for result reporting.

        budget spans may hide a real missed crossing; subband spans are
        below the declared dt/16 event bandwidth (stiff-noise structure).
        """
        return {'unresolved': self._cross_unresolved,
                'budget': self._cross_budget,
                'subband': self._cross_subband,
                'min_span': self._cross_min_span}

    def _gate_unresolved(self, u0: int, b0: int, s0: int):
        """Apply event_unresolved_policy to spans accrued during a run().

        Budget-class spans (a real crossing may have been missed) fail the
        run under the default 'error' policy; sub-bandwidth spans are
        within the engine's declared resolution and only count."""
        dbud = self._cross_budget - b0
        dsub = self._cross_subband - s0
        if dbud <= 0 and dsub <= 0:
            return
        w = self._cross_min_span
        ws = f', narrowest {w:.3e} s' if w is not None else ''
        msg = (f'{dbud} budget-exhausted + {dsub} sub-bandwidth unresolved '
               f'event spans{ws}')
        if self.event_unresolved_policy == 'error' and dbud > 0:
            raise SimError('EVENTS_UNRESOLVED', msg)
        if self.event_unresolved_policy == 'warn':
            self.warnings.append(f'EVENTS_UNRESOLVED: {msg}')

    def run(self, t_stop: float, sample_dt: Optional[float] = None,
            probes: Optional[List[str]] = None, stop_on_trigger: bool = False,
            on_trigger=None, rec=None):
        probes = probes or list(self.ckt.probes)
        samples: List[Tuple[float, Dict[str, float]]] = []
        next_s = self.t
        trig_t = None
        # unresolved-span gate counts this run's delta only: reruns after a
        # snapshot restore (POP iterations, AC windows) must not re-fail on
        # spans that were already accounted (or warned about)
        u0, b0, s0 = (self._cross_unresolved, self._cross_budget,
                      self._cross_subband)
        while self.t < t_stop - 1e-15:
            t_start = self.t
            self._advance_segments(self.t)   # heal any stale segment cursor
            topo = self.tc.compile(self.sw, self.dio)
            Aug, layout = self._build_aug(topo)
            r_of = self._row_builder(topo, layout)
            probe_rows = {p: r_of(self._probe_spec(p)) for p in probes}
            w0 = self._w_snapshot(layout)
            t_end = min(self._next_seg_boundary(),
                        self._heap[0][0] if self._heap else float('inf'),
                        t_stop)
            span = t_end - self.t
            if span <= 1e-18:
                self._advance_segments(t_end)
                hb = list(self._heap)
                fired = self._fire_known(t_end)
                if rec is not None:
                    rec({'t0': t_start, 't1': t_end, 'Aug': Aug,
                         'topo': topo, 'layout': layout, 'w0': w0,
                         'ev_kind': fired[1], 'ev_tag': fired[2],
                         'ev_t': fired[0], 'row': None,
                         'heap_before': hb, 'heap_after': list(self._heap)})
                if self.t >= t_stop - 1e-15:
                    break
                continue
            best = None
            dbg_on = bool(self.dbg)
            if dbg_on:
                db0, db1 = (float(x) for x in self.dbg.split(':'))
            for m in self._active_monitors(topo, layout, r_of):
                h0 = float(np.dot(m.row, w0))
                # anti-chatter dwell: a D/CMP/SW device cannot re-fire within
                # its dwell window; searching resumes when the dwell expires
                # (inside this interval if needed) so crossings are never lost
                tau0 = 0.0
                if m.kind in ('D', 'CMP', 'SW'):
                    dwell = 1e-12
                    if m.kind == 'CMP':
                        c = self.ckt.find(m.meta['dev'])
                        dwell = max(1e-12, c.delay)
                    rem = dwell - (self.t - self._last_flip.get(m.tag, -1e18))
                    if rem >= span:
                        if dbg_on and db0 <= self.t <= db1:
                            self.itrace.append(
                                f'  itr [{self.t:.12e},{t_end:.12e}] {m.tag}'
                                f' DWELL-SKIP rem={rem:.6e} span={span:.6e}')
                        continue
                    tau0 = max(rem, 0.0)
                h_at = h0 if tau0 == 0.0 else self._h(Aug, w0, m.row, tau0)
                if dbg_on and db0 <= self.t <= db1 and abs(h0) > 1e3:
                    bad = np.where(np.abs(m.row) > 1e2)[0]
                    self.itrace.append(
                        f'  GARBAGE {m.tag} h0={h0:.6e}: row cols>1e2: '
                        + ', '.join(f'{k}:{m.row[k]:.3e}' for k in bad[:8])
                        + f' | w0 max={np.max(np.abs(w0)):.3e}'
                        f' at {int(np.argmax(np.abs(w0)))}')
                if dbg_on and db0 <= self.t <= db1:
                    self.itrace.append(
                        f'  itr [{self.t:.12e},{t_end:.12e}] {m.tag}'
                        f' up={int(m.up)} h0={h0:+.9e} h@tau0={h_at:+.9e}'
                        f' tau0={tau0:.6e} th_hi={m.th_hi:.3e} th_lo={m.th_lo:.3e}')
                # event closure: a stateful device found past its armed
                # threshold fires immediately. TRIG is a hysteretic edge
                # detector supporting both edge directions; smooth (analog)
                # trigger nodes also get a real crossing search below.
                if m.kind == 'TRIG':
                    rising = m.meta.get('edge', 'rising') != 'falling'
                    if not self.trig_armed:
                        # rising: arm low, fire high; falling: arm high, low
                        if (h_at > m.th_hi) if not rising else (h_at < m.th_lo):
                            self.trig_armed = True
                        continue
                    fire_th = m.th_hi if rising else m.th_lo
                    past = h_at > m.th_hi if rising else h_at < m.th_lo
                    if past:
                        if best is None or tau0 < best[0]:
                            best = (tau0, m)
                        continue
                    f = (lambda tau, mm=m, thv=fire_th:
                         self._h(Aug, w0, mm.row, tau) - thv)
                    curv = self._hdd_bound(Aug, w0, m.row, span)
                    hit = self._find_crossing(f, span, h_at - fire_th,
                                              t0=tau0, curv=curv)
                    if hit is not None and (best is None or hit < best[0]):
                        best = (hit, m)
                    continue
                if m.up and h_at > m.th_hi:
                    if best is None or tau0 < best[0]:
                        best = (tau0, m)
                    continue
                if (not m.up) and h_at < m.th_lo:
                    if best is None or tau0 < best[0]:
                        best = (tau0, m)
                    continue
                th = m.th_hi if m.up else m.th_lo
                f = (lambda tau, mm=m, thv=th:
                     self._h(Aug, w0, mm.row, tau) - thv)
                curv = self._hdd_bound(Aug, w0, m.row, span)
                hit = self._find_crossing(f, span, h_at - th, t0=tau0,
                                          curv=curv)
                if dbg_on and db0 <= self.t <= db1:
                    self.itrace.append(
                        f'  itr [{self.t:.12e},{t_end:.12e}] {m.tag}'
                        f' search th={th:.6e} hit={hit}')
                if hit is not None and (best is None or hit < best[0]):
                    best = (hit, m)
            for spec in self._src_logic_specs:
                seg = self.seg[spec['src']]
                st = self.src_logic[spec['src']]
                th = spec['th'] + (spec['hyst'] / 2 if st == 0 else -spec['hyst'] / 2)
                if seg.s != 0.0 and seg.t0 <= self.t < seg.t1:
                    tcross = seg.t0 + (th - seg.a) / seg.s
                    if self.t < tcross <= t_end:
                        dt = tcross - self.t
                        if best is None or dt < best[0]:
                            best = (dt, Monitor(f'SRCLOG:{spec["src"]}', 'SRCLOG',
                                                np.zeros(layout['n_w']), 0, 0, True,
                                                meta={'src': spec['src']}))
            if best is not None and best[0] < span - 1e-18:
                t_event = self.t + best[0]
                if sample_dt:
                    while next_s <= t_event + 1e-15:
                        if next_s >= self.t - 1e-15:
                            tau = max(next_s - self.t, 0.0)
                            wv = self._propagate(Aug, w0, tau)
                            samples.append((next_s, {p: float(np.dot(probe_rows[p], wv))
                                                     for p in probes}))
                        next_s += sample_dt
                w1 = self._propagate(Aug, w0, best[0])
                self._commit_state(w1, layout, self.t + best[0])
                self._advance_segments(self.t)
                if best[1].kind == 'TRIG':
                    trig_t = self.t
                    self.trig_armed = False
                    if rec is not None:
                        rec({'t0': t_start, 't1': self.t, 'Aug': Aug,
                             'topo': topo, 'layout': layout, 'w0': w0,
                             'ev_kind': 'TRIG', 'ev_tag': best[1].tag,
                             'ev_t': self.t, 'row': None,
                             'heap_before': list(self._heap),
                             'heap_after': list(self._heap)})
                    if on_trigger is not None:
                        on_trigger(self)
                    if stop_on_trigger:
                        self._gate_unresolved(u0, b0, s0)
                        return samples, trig_t
                hb = list(self._heap)
                self._apply_dev(best[1])
                if rec is not None:
                    rec({'t0': t_start, 't1': self.t, 'Aug': Aug,
                         'topo': topo, 'layout': layout, 'w0': w0,
                         'ev_kind': best[1].kind, 'ev_tag': best[1].tag,
                         'ev_t': t_event, 'row': best[1].row,
                         'heap_before': hb, 'heap_after': list(self._heap)})
            else:
                if sample_dt:
                    while next_s <= t_end + 1e-15:
                        if next_s >= self.t - 1e-15:
                            tau = max(next_s - self.t, 0.0)
                            wv = self._propagate(Aug, w0, tau)
                            samples.append((next_s, {p: float(np.dot(probe_rows[p], wv))
                                                     for p in probes}))
                        next_s += sample_dt
                w1 = self._propagate(Aug, w0, span)
                self._commit_state(w1, layout, t_end)
                self._advance_segments(t_end)
                hb = list(self._heap)
                fired = self._fire_known(t_end)
                if rec is not None:
                    rec({'t0': t_start, 't1': t_end, 'Aug': Aug,
                         'topo': topo, 'layout': layout, 'w0': w0,
                         'ev_kind': fired[1], 'ev_tag': fired[2],
                         'ev_t': fired[0], 'row': None,
                         'heap_before': hb, 'heap_after': list(self._heap)})
        self._gate_unresolved(u0, b0, s0)
        return samples, trig_t

    def probe_now(self, ptype, ref):
        topo = self.tc.compile(self.sw, self.dio)
        Aug, layout = self._build_aug(topo)
        r_of = self._row_builder(topo, layout)
        w = self._w_snapshot(layout)
        return float(np.dot(r_of((ptype, ref)), w))

    # ---------------------------------------------------------- snapshots
    def snapshot(self) -> dict:
        return {
            't': self.t, 'x': self.x.copy(),
            'sw': dict(self.sw), 'dio': dict(self.dio),
            'cmp': dict(self.cmp_st), 'srff_q': dict(self.srff_q),
            'dig': dict(self.dig_val), 'src_logic': dict(self.src_logic),
            'seg': {k: (s.t0, s.t1, s.a, s.s) for k, s in self.seg.items()},
            'heap': list(self._heap),
            'osc': self.osc.copy(),
            'nev': self.nev,
            'last_flip': dict(self._last_flip),
            'trig_armed': self.trig_armed,
        }

    def restore(self, s: dict):
        self.t = s['t']
        self.x = s['x'].copy()
        self.sw = dict(s['sw'])
        self.dio = dict(s['dio'])
        self.cmp_st = dict(s['cmp'])
        self.srff_q = dict(s['srff_q'])
        self.dig_val = dict(s['dig'])
        self.src_logic = dict(s['src_logic'])
        self.seg = {k: Segment(*v) for k, v in s['seg'].items()}
        self._heap = list(s['heap'])
        heapq.heapify(self._heap)
        self.osc = s['osc'].copy()
        self._last_flip = dict(s.get('last_flip', {}))
        self.trig_armed = s.get('trig_armed', False)
