# -*- coding: utf-8 -*-
"""POP (Periodic Operating Point) solver — plan §9.

Poincare map P: full engine state at one trigger event -> state at the next
trigger. Newton on R(x) = P(x) - x with a finite-difference Jacobian and
damped updates; plain fixed-point iteration as a fallback. Because every
propagation is exact (matrix exponential), the map itself has no integration
error, so the fixed point can be located to near machine precision.

Discrete states (switches, diodes, comparator, latches, delay queue) must
repeat at the boundary before convergence is declared.
"""
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional

import numpy as np

from .engine import Engine, SimError


@dataclass
class PopResult:
    ok: bool
    iterations: int
    period: float
    x: np.ndarray
    residual: float
    discrete_match: bool
    floquet: Optional[np.ndarray] = None
    max_multiplier: Optional[float] = None
    floquet_isolated: Optional[np.ndarray] = None   # bool mask on `floquet`
    history: List[dict] = field(default_factory=list)
    diag: str = ""


def _discrete_signature(s: dict, T_nom: Optional[float] = None) -> tuple:
    t0 = s['t']
    # pending delay events (review §7.1): channel, target value and the
    # remaining time RELATIVE to the boundary.  Absolute event times shift
    # by exactly one period on the true orbit, so only the relative
    # encoding is periodic; the heap seq counter is bookkeeping, not state.
    # Time fields round to 1e-10 s (100 ps), NOT 1e-12: on orbits whose
    # trigger crossing is an analog ramp (negative inductor valley: SW
    # rides the CSW ramp during dead time) the crossing time carries a
    # state-dependent jitter of ~ps at the atol-calibrated convergence
    # level; a 1 ps grid then never matches and POP rejects a legitimate
    # converged orbit (fsw=300k repro, tenth session R1b).  100 ps is
    # still two orders below the fastest modelled digital timing (1 ns
    # CMP delay / 20 ns dead time) and cannot conflate distinct modes.
    heap = tuple(sorted((ch, round(val, 9), round(t - t0, 10))
                        for (t, ch, val, _seq) in s.get('heap', ())))
    # source segment pattern and phase: affine content always, plus the
    # segment start relative to the boundary (modulo the nominal switching
    # period when known) for finite segments.  Periodic sources roll their
    # t0 by exactly T per period, so the relative time is already periodic;
    # ONE-SHOT sources (e.g. a load-step step at 500 us) sit in a long
    # finite segment whose t0 never advances -- the modulo makes their
    # residue periodic too.  Infinite segments (DC, sine envelope) carry
    # content only: their relative time would drift by -T per period.
    segs = []
    for k, (s_t0, s_t1, s_a, s_s) in sorted(s.get('seg', {}).items()):
        item = (k, round(s_a, 9), round(s_s, 9))
        if not math.isinf(s_t1):
            rel = s_t0 - t0
            if T_nom:
                rel %= T_nom
            item = item + (round(rel, 10),)
        segs.append(item)
    return (tuple(sorted(s['sw'].items())), tuple(sorted(s['dio'].items())),
            tuple(sorted(s['cmp'].items())), tuple(sorted(s['srff_q'].items())),
            tuple(sorted((k, round(v, 9)) for k, v in s['dig'].items())),
            tuple(sorted(s['src_logic'].items())),
            tuple(segs), heap)


class PopSolver:
    def __init__(self, engine: Engine, opt: Optional[dict] = None):
        opt = opt or {}
        self.e = engine
        self.pre_cycles = opt.get('pre_cycles', 150)
        # 100 (not 20): harder basins floor-straddle the residual gate --
        # C6x2 N=1 lands at iter 22, fsw=300k at 25, L=2.35u wanders the
        # noise floor until ~iter 40-85 (tenth session R1).  Successful
        # solves exit on the first gate pass, so the budget only costs
        # wall time on genuinely failing cases.
        self.max_iter = opt.get('max_iter', 100)
        self.tol = opt.get('tol', 1e-6)
        # atol is calibrated against the eig-propagator roundoff floor of the
        # one-cycle map (~2.4e-11 abs on this model): the scaled residual
        # |dx|/(atol+rtol*|x|) must be able to fall below tol at the map's own
        # reproducibility limit, else POP never declares convergence even on
        # the exact fixed point (nano-scale boundary states such as the shorted
        # ramp cap sit at 2 nV and would otherwise dominate the residual).
        self.atol = opt.get('atol', 5e-5)
        self.rtol = opt.get('rtol', 1e-7)
        self.max_period_cycles = opt.get('max_period_cycles', 8)
        self.n_mult = opt.get('n_mult', 2)      # 2 = try N=1 then N=2 (period-2)
        # declared nominal period for oscillator-clocked models (may be a
        # unit string like '10u' -- parsed lazily in solve)
        self.period_guess = opt.get('period_guess')

    # ------------------------------------------------------------------
    def _run_to_trigger(self, t_limit):
        """Run until the next trigger crossing; returns trigger time."""
        _, trig = self.e.run(t_limit, stop_on_trigger=True)
        if trig is None:
            raise SimError('TRIGGER_NOT_FOUND',
                           f'no trigger crossing before {t_limit}')
        return trig

    def _map(self, snap: dict, x: np.ndarray, cycles: float, t_limit):
        """Evaluate P(x): restore snapshot, perturb x, run to next trigger(s)."""
        self.e.restore(snap)
        self.e.x = x.copy()
        t0 = snap['t']
        # run `cycles` trigger crossings (period-N support: cycles>1)
        trig = None
        for _ in range(int(cycles)):
            trig = self._run_to_trigger(t_limit)
        s1 = self.e.snapshot()
        T = trig - t0
        return s1, T

    @staticmethod
    def _scaled_res(x0, x1, atol, rtol):
        sc = np.maximum(np.maximum(np.abs(x0), np.abs(x1)), 1e-3)
        return np.abs(x1 - x0) / (atol + rtol * sc)

    # ------------------------------------------------------------------
    def solve(self) -> PopResult:
        e = self.e
        # 1. pre-POP transient so the orbit basin / topology sequence settles
        from .waveforms import Waveform
        # nominal switching period from any periodic pulse source
        T_nom = None
        for d in e.ckt.by_kind('V'):
            if d.wave.period:
                T_nom = d.wave.period
                break
        if T_nom is None:
            # Oscillator-clocked models (P2 VCO) carry no periodic pulse
            # source: fall back to the declared pop.period_guess before
            # the hard 1 us default -- both the trigger-search window
            # (max_period_cycles * T_nom) and the discrete-signature
            # period derive from T_nom.  Models with a pulse source are
            # unaffected (the first branch wins), so existing goldens
            # cannot move.
            pg = self.period_guess
            if isinstance(pg, str):
                from .units import eng
                pg = eng(pg)
            T_nom = pg if pg else 1e-6
        e.run(self.pre_cycles * T_nom)
        # 2. first trigger
        self._run_to_trigger(e.t + self.max_period_cycles * T_nom + 1e-6)
        snap0 = e.snapshot()
        self.snap0 = snap0        # exposed for AC (start exactly on x*)
        self._sig_T = T_nom       # signature period (one-shot sources)
        sig0 = _discrete_signature(snap0, T_nom)
        t_lim = snap0['t'] + self.max_period_cycles * T_nom + 1e-6

        # period-N candidates: N=1 first; N=2 catches subharmonic (period-2)
        # orbits where P itself has no fixed point but P^2 does.
        last = None
        for N in ([1, 2] if self.n_mult == 2 else [self.n_mult]):
            last = self._solve_N(snap0, sig0, t_lim, N)
            last.history = [{'N': N, **h} for h in last.history]
            if last.ok:
                return last
        return last

    def _solve_N(self, snap0, sig0, t_lim, N) -> PopResult:
        x = snap0['x'].copy()
        hist = []
        last_diag = ""
        stall = 0
        rmax = float('inf')
        T = 0.0
        dmatch = False
        for it in range(1, self.max_iter + 1):
            s1, T = self._map(snap0, x, N, t_lim)
            r = self._scaled_res(x, s1['x'], self.atol, self.rtol)
            rmax = float(r.max()) if r.size else 0.0
            sig1 = _discrete_signature(s1, getattr(self, '_sig_T', None))
            dmatch = sig1 == sig0
            hist.append({'iter': it, 'res': rmax, 'T': T, 'discrete': dmatch})
            if rmax < self.tol and dmatch:
                # verification run from the candidate (plan §9.3 phase 4)
                s2, T2 = self._map(snap0, s1['x'], N, t_lim)
                r2 = self._scaled_res(s1['x'], s2['x'], self.atol, self.rtol)
                if float(r2.max()) < max(self.tol * 10, 1e-6):
                    flo, maxmul, fiso = self._floquet(snap0, s1['x'], t_lim, N)
                    return PopResult(ok=True, iterations=it, period=T2,
                                     x=s2['x'], residual=float(r2.max()),
                                     discrete_match=True, floquet=flo,
                                     max_multiplier=maxmul,
                                     floquet_isolated=fiso, history=hist,
                                     diag=last_diag)
                last_diag = f"verification residual bounced: {float(r2.max()):.3g}"
            # --- step selection: fixed point while it contracts fast; Newton
            # (monotone-accepted) once the fixed point stalls
            prev_res = hist[-2]['res'] if len(hist) > 1 else float('inf')
            contracting = rmax < 0.9 * prev_res
            if contracting or stall < 2:
                stall = 0 if contracting else stall + 1
                x = s1['x'].copy()
                continue
            # stalled: one damped Newton attempt (accept only if it improves).
            # R(x) = P(x) - x  =>  dR/dx = DP - I  (review issue 4: solving
            # with plain DP gives a wrong direction on slow/unstable maps)
            stall = 0
            DP = self._fd_jacobian(snap0, x, s1['x'], t_lim, hrel=3e-6, N=N)
            R = s1['x'] - x
            JR = DP - np.eye(len(x))
            try:
                dx = np.linalg.solve(JR, -R)
            except np.linalg.LinAlgError:
                dx, *_ = np.linalg.lstsq(JR, -R, rcond=None)
            lam = 1.0
            xt_best, rt_best = None, None
            for _ in range(6):
                xt = x + lam * dx
                try:
                    st, _ = self._map(snap0, xt, N, t_lim)
                except SimError:
                    lam *= 0.5
                    continue
                rt = float(self._scaled_res(xt, st['x'], self.atol,
                                            self.rtol).max())
                if rt_best is None or rt < rt_best:
                    xt_best, rt_best = xt, rt
                if rt < 0.5 * rmax:
                    break
                lam *= 0.5
            if xt_best is not None and rt_best < rmax:
                x = xt_best
            else:
                x = s1['x'].copy()
            if not np.all(np.isfinite(x)):
                raise SimError('POP_STAGNATION', 'state diverged')

        return PopResult(ok=False, iterations=self.max_iter, period=T, x=x,
                         residual=rmax, discrete_match=dmatch,
                         floquet=None, max_multiplier=None,
                         floquet_isolated=None, history=hist,
                         diag=last_diag or f'POP did not converge (N={N})')

    def _fd_jacobian(self, snap, x, x1_ref, t_lim, hrel=1e-7, N=1):
        n = len(x)
        J = np.zeros((n, n))
        sc = np.maximum(np.abs(x), 1e-3)
        for j in range(n):
            h = hrel * sc[j]
            xp = x.copy()
            xp[j] += h
            sp, _ = self._map(snap, xp, N, t_lim)
            J[:, j] = (sp['x'] - x1_ref) / h
        return J

    def _floquet(self, snap, x_star, t_lim, N=1):
        n = len(x_star)
        s1, _ = self._map(snap, x_star, N, t_lim)
        # central differences for a cleaner monodromy matrix
        sc = np.maximum(np.abs(x_star), 1e-3)
        J = np.zeros((n, n))
        for j in range(n):
            h = 1e-6 * sc[j]
            xp = x_star.copy(); xp[j] += h
            xm = x_star.copy(); xm[j] -= h
            sp, _ = self._map(snap, xp, N, t_lim)
            sm, _ = self._map(snap, xm, N, t_lim)
            J[:, j] = (sp['x'] - sm['x']) / (2 * h)
        ev, vecs = np.linalg.eig(J)
        # Structural isolation replaces the old |lambda|~1 filter (review
        # issue 5): only modes whose perturbation mass provably lies on
        # storage elements outside the trigger's galvanic component are
        # separated; a genuine near-unity MAIN mode is always kept.
        iso = self._isolated_modes(snap, vecs)
        main = np.abs(ev)[~iso] if iso is not None else np.abs(ev)
        return ev, (float(np.max(main)) if main.size else 0.0), iso

    # ------------------------------------------------------------ isolation
    def _iso_setup(self, snap):
        """Rows reconstructing isolated vs main storage values in the y
        basis.  Returns (P_iso, P_main) or None when the circuit has no
        galvanically separate storage (nothing to separate)."""
        if getattr(self, '_iso_rows', None) is not None:
            return self._iso_rows
        e = getattr(self, 'e', None)
        if e is None:
            return None                     # synthetic maps in tests
        ckt = e.ckt
        GND = '0'
        parent = {}

        def find(a):
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            if a == GND or b == GND or a is None or b is None:
                return
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # union all device terminal pairs EXCLUDING ground (ground is the
        # shared reference, not a signal path).  'inputs' makes the
        # gate-input -> gate-output dependency edge explicit (review
        # round-7 §13.2): without it the list branch below was dead code.
        for d in ckt.devices:
            terms = []
            for attr in ('n1', 'n2', 'd1', 'd2', 'c1', 'c2', 'a', 'k',
                         'na', 'nb', 'inp', 'inn', 'out', 'q', 'nq',
                         's', 'r', 'node', 'inputs'):
                v = getattr(d, attr, None)
                if isinstance(v, str):
                    terms.append(v)
                if attr in ('inputs',) and isinstance(v, list):
                    terms += [t for t in v if isinstance(t, str)]
            for i in range(1, len(terms)):
                union(terms[0], terms[i])
        trigs = ckt.by_kind('TRIG')
        main_comp = find(trigs[0].node) if trigs else None
        if main_comp is None:
            self._iso_rows = None
            return None
        topo = self.e.tc.compile(snap['sw'], snap['dio'])
        tc = self.e.tc

        def rows_for(devs):
            rows = []
            for d in devs:
                if d.kind == 'C':
                    r = np.zeros(topo.n_x)
                    if d.n1 != GND:
                        r += topo.colmap[tc._n(d.n1)][0]
                    if d.n2 != GND:
                        r -= topo.colmap[tc._n(d.n2)][0]
                    rows.append(r)
                elif d.kind == 'L':
                    rows.append(topo.colmap[tc.il_cols[d.name]][0])
            return np.array(rows) if rows else np.zeros((0, topo.n_x))

        def comp_of(d):
            ns = [x for x in (getattr(d, 'n1', None), getattr(d, 'n2', None))
                  if isinstance(x, str) and x != GND]
            return find(ns[0]) if ns else None

        caps = ckt.by_kind('C')
        inds = ckt.by_kind('L')
        iso_devs = [d for d in caps + inds if comp_of(d) != main_comp]
        main_devs = [d for d in caps + inds if comp_of(d) == main_comp]
        if not iso_devs:
            self._iso_rows = None
            return None
        self._iso_rows = (rows_for(iso_devs), rows_for(main_devs))
        return self._iso_rows

    def _isolated_modes(self, snap, vecs):
        """Boolean mask over eigenvectors: perturbation mass >= 90% on
        storage elements outside the trigger component."""
        setup = self._iso_setup(snap)
        if setup is None:
            return None
        P_iso, P_main = setup
        mask = np.zeros(vecs.shape[1], dtype=bool)
        for k in range(vecs.shape[1]):
            v = vecs[:, k]
            mi = float(np.linalg.norm(P_iso @ np.real(v))) ** 2 + \
                float(np.linalg.norm(P_iso @ np.imag(v))) ** 2
            mm = float(np.linalg.norm(P_main @ np.real(v))) ** 2 + \
                float(np.linalg.norm(P_main @ np.imag(v))) ** 2
            tot = mi + mm
            if tot > 0:
                mask[k] = mi > 0.9 * tot
        return mask
