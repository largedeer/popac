# -*- coding: utf-8 -*-
"""AC loop-gain analysis by injection fitting (plan Phase 7).

Starting from the POP periodic operating point, a small sine is injected in
series with the loop (Middlebrook): the PS15 probe source V16 sits between
the loop upstream node (vout) and the downstream divider node (fbt).  For
each frequency the engine is rebuilt (the augmented oscillator bank is
frozen per engine instance), the POP snapshot is restored and patched with
the injection oscillator state, the transient is run past a discard window,
and v(probe_a)/v(probe_b) are least-squares fitted to cos/sin at the
injection frequency.  Loop gain

    T(f) = - phasor(probe_a) / phasor(probe_b)

(sign follows the standard Bode-probe convention IN=upstream / OUT=downstream;
if the phase margin comes out negative the model uses the opposite sign).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .engine import Engine, SimError
from .units import eng
from .waveforms import Waveform


def _num(v, default):
    """YAML analysis options arrive unit-suffixed ('20m', '2.5u')."""
    try:
        return eng(v)
    except (ValueError, TypeError):
        return float(default)


@dataclass
class AcResult:
    freqs: np.ndarray                 # Hz
    T: np.ndarray                     # complex loop gain at each freq
    ok: List[bool]                    # per-point run success
    status: List[str] = field(default_factory=list)    # PASS / NOT_SETTLED / FAILED
    amp_linearity: Optional[Dict] = None   # per-band 5m-vs-20m comparison
    diag: str = ""
    metrics: Dict = field(default_factory=dict)


class AcSolver:
    def __init__(self, circuit, opt: Optional[dict] = None,
                 pop_snapshot: Optional[dict] = None,
                 pop_x: Optional[np.ndarray] = None):
        opt = opt or {}
        self.ckt = circuit
        self.snap = pop_snapshot
        self.x_star = pop_x
        self.fstart = _num(opt.get('fstart', 100.0), 100.0)
        self.fstop = _num(opt.get('fstop', 1e6), 1e6)
        self.per_decade = int(opt.get('per_decade', 25))
        self.inject_src = opt.get('inject_src', 'V16')
        self.inject_amp = _num(opt.get('inject_amp', 20e-3), 20e-3)
        self.probe_a = opt.get('probe_a', 'vout')   # upstream side
        self.probe_b = opt.get('probe_b', 'fbt')    # downstream side
        # 300 cycles (750 us) settles the near-crossover closed-loop
        # transient: at |T|~1 the response is largest and a 60-cycle
        # discard left half-window disagreements of ~2x the tolerance
        # (points at fc gated NOT_SETTLED, PM uncomputable)
        self.discard_cycles = opt.get('discard_cycles', 300)
        self.min_cycles = opt.get('min_cycles', 100)
        self.max_cycles = opt.get('max_cycles', 4000)
        self.window_sec = opt.get('window_sec', 1.2)   # observe >= 1.2 periods
        self.samples_per_cycle = opt.get('samples_per_cycle', 16)
        self.linearity_amp2 = opt.get('linearity_amp2', 5e-3)
        # quality gates (review issue 6)
        self.harm_k = int(opt.get('harm_k', 2))        # sideband pairs in fit
        self.win_tol_db = opt.get('win_tol_db', 0.25)  # half-window agreement
        self.win_tol_deg = opt.get('win_tol_deg', 1.5)
        # per-point engine event cap: a legitimate point needs <10k events
        # (n_win cycles x ~25 events); a grazing/chatter frequency would
        # otherwise burn hours walking to the default 20M cap.
        self.max_events = int(opt.get('max_events', 2_000_000))
        # optional per-point progress hook: fn(i, freq, status); run.py
        # uses it for incremental ac_raw.json flushes
        self.on_point = opt.get('on_point')

    # ------------------------------------------------------------------
    def _freqs(self) -> np.ndarray:
        n_dec = math.log10(self.fstop / self.fstart)
        return self.fstart * 10 ** (
            np.arange(0, round(n_dec * self.per_decade) + 1) / self.per_decade)

    def _engine_for(self, freq: float) -> Engine:
        dev = self.ckt.find(self.inject_src)
        if dev is None or dev.kind != 'V':
            raise SimError('AC_CONFIG',
                           f'inject source {self.inject_src} not found / not a V')
        dev.wave = Waveform.sine(0.0, self.inject_amp, freq)
        e = Engine(self.ckt, {'max_events': self.max_events})
        if self.snap is not None:
            e.restore(self.snap)
            e.x = (self.x_star if self.x_star is not None
                   else self.snap['x']).copy()
            # patch the injection source into the restored state: segment
            # cursor + oscillator phase at the restore instant.  The snap's
            # oscillator bank (no sines at POP time) is rebuilt for this run.
            e.seg[self.inject_src] = dev.wave.segment_at(e.t)
            e.osc = np.zeros(2 * len(e.omegas))
            w = 2 * math.pi * freq
            e.osc[0] = math.cos(w * e.t)
            e.osc[1] = math.sin(w * e.t)
        return e

    def _fit(self, samples, freq, T_sw):
        """LSQ phasors at the injection frequency with switching-harmonic
        sidebands absorbed: the basis carries DC, the k=0 pair at omega and
        the pairs at omega +/- k*omega_s (k=1..harm_k), so ripple leakage
        does not bias the extracted phasor (review issue 6).  Columns whose
        frequency folds onto DC or duplicates another column (omega near a
        switching harmonic) are dropped."""
        ts = np.array([t for t, _ in samples])
        w = 2 * math.pi * freq
        ws = 2 * math.pi / T_sw
        self._last_fold = False        # set True when k0 folds onto a column
        cols = [np.ones_like(ts)]
        i0 = None                       # column index of the k=0 cos coeff
        seen = [0.0]                     # DC already present
        seen_idx = [0]

        def add(wk, tag=None):
            nonlocal i0
            for j, u in enumerate(seen):
                if abs(abs(wk) - u) < 1e-9 * ws:
                    if tag == 'k0':
                        # injection frequency folded onto an existing
                        # column (f at/near a switching harmonic): the
                        # alias is physically inseparable — reuse it
                        i0 = seen_idx[j]
                        self._last_fold = True
                    return
            seen.append(abs(wk))
            cols.append(np.cos(wk * ts))
            cols.append(np.sin(wk * ts))
            seen_idx.append(len(cols) - 2)
            if tag == 'k0':
                i0 = len(cols) - 2

        for k in range(-self.harm_k, self.harm_k + 1):
            add(w + k * ws, 'k0' if k == 0 else None)
        # the switching ripple itself (k*ws) also leaks into a non-integer
        # excitation window; absorb it explicitly
        for k in range(1, self.harm_k + 1):
            add(k * ws)
        M = np.column_stack(cols)
        out = {}
        for p in (self.probe_a, self.probe_b):
            y = np.array([v[p] for _, v in samples])
            coef, *_ = np.linalg.lstsq(M, y, rcond=None)
            out[p] = complex(coef[i0], -coef[i0 + 1])
        return out

    def run_point(self, freq: float, amp: Optional[float] = None):
        """Run one frequency; returns {'T', 'halves'} for quality gating.

        _engine_for rewrites the injection source's waveform to a sine;
        the original is restored on exit so the shared circuit object is
        not silently mutated for later solvers (a fresh Engine over the
        circuit would otherwise see a sine the POP snapshot never had).
        """
        if amp is not None:
            self.inject_amp = amp
        T_sw = 2.5e-6
        for d in self.ckt.by_kind('V'):
            if d.wave.period:
                T_sw = d.wave.period
                break
        inj = self.ckt.find(self.inject_src)
        orig_wave = inj.wave
        try:
            e = self._engine_for(freq)
            n_win = int(min(self.max_cycles,
                            max(self.min_cycles,
                                math.ceil(self.window_sec / (freq * T_sw)))))
            e.run(self.discard_cycles * T_sw)
            t0 = e.t
            samples, _ = e.run(t0 + n_win * T_sw,
                               sample_dt=T_sw / self.samples_per_cycle,
                               probes=[self.probe_a, self.probe_b])
        finally:
            inj.wave = orig_wave
        if len(samples) < 8:
            raise SimError('AC_WINDOW', f'too few samples at {freq:g} Hz')
        ph = self._fit(samples, freq, T_sw)
        folded = self._last_fold          # k0 folded onto a sideband column
        mid = len(samples) // 2
        h1 = self._fit(samples[:mid], freq, T_sw)
        h2 = self._fit(samples[mid:], freq, T_sw)
        Tb = ph[self.probe_b]
        if abs(Tb) == 0:
            raise SimError('AC_DEGENERATE',
                           f'probe {self.probe_b} flat at {freq:g} Hz')
        return {'T': -ph[self.probe_a] / Tb,
                'h1': h1, 'h2': h2, 'n_cycles': n_win,
                'truncated': n_win >= self.max_cycles, 'folded': folded}

    def _point_status(self, pt):
        """PASS only when both window halves agree (steady state reached).
        A folded point (injection at/near a switching harmonic) carries a
        physically aliased phasor and never participates in PM/GM."""
        if pt.get('folded'):
            return 'FOLDED'
        dT = []
        for p in (self.probe_a, self.probe_b):
            a, b = pt['h1'][p], pt['h2'][p]
            if abs(a) == 0 or abs(b) == 0:
                return 'NOT_SETTLED'
            dm = abs(20 * math.log10(abs(b) / abs(a)))
            dp = abs(np.degrees(np.angle(b / a)))
            dT.append(max(dm / max(self.win_tol_db, 1e-9),
                          dp / max(self.win_tol_deg, 1e-9)))
        return 'PASS' if max(dT) < 1.0 else 'NOT_SETTLED'

    def solve_all(self) -> AcResult:
        freqs = self._freqs()
        T = np.zeros(len(freqs), dtype=complex)
        ok, status = [], []
        # in-progress refs for the on_point incremental flush
        self._part = (freqs, T, ok, status)
        T_sw = 2.5e-6
        for d in self.ckt.by_kind('V'):
            if d.wave.period:
                T_sw = d.wave.period
                break
        for i, f in enumerate(freqs):
            try:
                pt = self.run_point(f)
                T[i] = pt['T']
                ok.append(True)
                status.append(self._point_status(pt))
            except SimError as exc:
                ok.append(False)
                status.append('FAILED')
                T[i] = complex(np.nan, np.nan)
                print(f'  AC[{i}] {f:g} Hz FAILED: {exc}', flush=True)
            if self.on_point is not None:
                self.on_point(i, float(f), status[-1])
        res = AcResult(freqs=freqs, T=T, ok=ok, status=status)
        # amplitude linearity across the band (low / mid / fs-over-2 / high):
        # |T| must not depend on the injection amplitude in the small-signal
        # regime; each entry is one band.
        res.amp_linearity = {}
        amp_nom = self.inject_amp
        T_fsw = T_sw
        bands = {'low': float(freqs[1]),
                 'mid': float(freqs[len(freqs) // 2]),
                 'fs_half': 0.5 / T_fsw,
                 'high': float(freqs[-2])}
        for name, f_lin in bands.items():
            try:
                t1 = self.run_point(f_lin, amp_nom)['T']
                t2 = self.run_point(f_lin, self.linearity_amp2)['T']
                res.amp_linearity[name] = {
                    'freq': f_lin,
                    'mag_ratio': float(abs(t2) / abs(t1)) if abs(t1) else np.nan,
                    'phase_diff_deg': float(np.degrees(np.angle(t2 / t1))
                                            if abs(t1) else np.nan),
                }
            except SimError:
                res.amp_linearity[name] = None
        # PM/GM participate from PASS points only, split into contiguous
        # segments (review §7.4: no interpolation across invalid points)
        res.metrics = segmented_metrics(freqs, T, status)
        res.metrics['n_pass'] = int(sum(1 for s in status if s == 'PASS'))
        res.metrics['n_total'] = len(freqs)
        return res


def _principal_deg(d: float) -> float:
    """Map degrees to (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def segmented_metrics(freqs, T, status) -> Dict:
    """PM/GM from PASS points split into contiguous segments (review §7.4).

    Interpolation is legal only between ORIGINAL-adjacent points, so each
    run of PASS points is measured independently.  Where an invalid gap
    separates flanking PASS points whose magnitude sits on opposite sides
    of 0 dB (or whose phase on opposite sides of -180 deg), a crossing is
    CERTAIN to hide in the gap: flagged in margin_unresolved
    (MARGIN_UNRESOLVED) instead of being silently bridged by masked
    interpolation over the gap."""
    freqs = np.asarray(freqs, dtype=float)
    T = np.asarray(T)
    ok = np.isfinite(T.real) & np.isfinite(T.imag)
    mask = np.array([s == 'PASS' for s in status]) & ok
    out: Dict = {'crossovers': [], 'gain_margin_db': None,
                 'margin_unresolved': [], 'segments': []}
    idx = np.where(mask)[0]
    segs: List = []
    if idx.size:
        start = prev = int(idx[0])
        for i in idx[1:]:
            i = int(i)
            if i != prev + 1:
                segs.append((start, prev))
                start = i
            prev = i
        segs.append((start, prev))
    for i0, i1 in segs:
        sm = loop_metrics(freqs[i0:i1 + 1], T[i0:i1 + 1])
        for c in sm['crossovers']:
            c['seg'] = [i0, i1]
        out['crossovers'] += sm['crossovers']
        out['segments'].append([i0, i1])
        if out['gain_margin_db'] is None and sm.get('gm_at_freq') is not None:
            out['gain_margin_db'] = sm['gain_margin_db']
            out['gm_at_freq'] = sm['gm_at_freq']
    # gaps between consecutive segments: opposing flank signs prove that a
    # crossing hides inside the invalid region
    for k in range(len(segs) - 1):
        ia, ib = segs[k][1], segs[k + 1][0]
        if (abs(T[ia]) > 1.0) != (abs(T[ib]) > 1.0):
            out['margin_unresolved'].append(
                {'kind': 'crossover', 'f_lo': float(freqs[ia]),
                 'f_hi': float(freqs[ib])})
        pa = float(np.degrees(np.angle(T[ia])))
        pb = pa + _principal_deg(float(np.degrees(np.angle(T[ib]))) - pa)
        if (pa + 180.0 > 0.0) != (pb + 180.0 > 0.0):
            out['margin_unresolved'].append(
                {'kind': 'phase_180', 'f_lo': float(freqs[ia]),
                 'f_hi': float(freqs[ib])})
    return out


def loop_metrics(freqs: np.ndarray, T: np.ndarray) -> Dict:
    """0 dB crossings with phase margin, and gain margin (first -180 crossing).

    Non-finite points (PAC at a folded switching harmonic yields NaN) are
    dropped first: np.sign(NaN-1) is NaN and would fabricate a spurious
    crossover with an NaN frequency."""
    ok = np.isfinite(np.asarray(T).real) & np.isfinite(np.asarray(T).imag)
    freqs = np.asarray(freqs, dtype=float)[ok]
    T = np.asarray(T)[ok]
    mag = np.abs(T)
    ph = np.unwrap(np.angle(T))
    out = {'crossovers': []}
    s = np.sign(mag - 1.0)
    for i in np.where(np.diff(s) != 0)[0]:
        f1, f2 = freqs[i], freqs[i + 1]
        m1, m2 = mag[i], mag[i + 1]
        if m1 == m2:
            continue
        fc = f1 * (1.0 / m1) ** (np.log(f2 / f1) / np.log(m2 / m1))
        p1, p2 = ph[i], ph[i + 1]
        pfc = p1 + (p2 - p1) * np.log(fc / f1) / np.log(f2 / f1)
        out['crossovers'].append(
            {'freq': float(fc), 'phase_deg': float(np.degrees(pfc)),
             'pm_deg': float(180.0 + np.degrees(pfc))})
    gm = None
    sp = np.sign(np.degrees(ph) + 180.0)
    for i in np.where(np.diff(sp) != 0)[0]:
        if mag[i] > 0:
            gm = -20.0 * np.log10(mag[i])
            out['gm_at_freq'] = float(freqs[i])
            break
    out['gain_margin_db'] = gm
    return out
