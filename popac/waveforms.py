# -*- coding: utf-8 -*-
"""Affine-segment waveforms with optional sinusoid components.

A waveform is piecewise-affine in time plus optional sinusoids. Between any
two events every independent source is exactly representable as

    u(t) = a + s*(t - t_seg_start) + sum_k K_k * sin(w_k*t + phi_k)

which the propagation kernel integrates exactly via an augmented matrix
exponential (oscillator states carry absolute phase).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .units import eng


@dataclass
class Segment:
    """Half-open absolute window [t0, t1); value a + s*(t-t0)."""
    t0: float
    t1: float           # inf for the final segment of a non-periodic source
    a: float            # value at t0
    s: float            # slope V/s (A/s)

    def value(self, t: float) -> float:
        return self.a + self.s * (t - self.t0)


@dataclass
class SinePart:
    omega: float        # rad/s (absolute time base t=0)
    K: float            # amplitude
    phi: float = 0.0    # radians


class Waveform:
    """Periodic or single-shot piecewise-affine waveform (+ sines).

    period is None -> segments list is absolute & finite (usually one DC seg).
    """

    def __init__(self, period: Optional[float], segs: List[Segment],
                 sines: Optional[List[SinePart]] = None):
        self.period = period
        self.segs = segs          # relative [0, period) if periodic
        self.sines = sines or []
        self._cursor_k = 0        # current cycle index (int), maintained by engine
        self._cursor_i = 0

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def dc(v):
        return Waveform(None, [Segment(0.0, float('inf'), float(v), 0.0)])

    @staticmethod
    def pulse(v1, v2, period, width, rise, fall, delay=0.0):
        """SPICE-style PULSE. width counted from end of rise. Periodic."""
        v1, v2, period, width, rise, fall = map(float, (v1, v2, period, width, rise, fall))
        segs = []
        t = delay
        if delay > 0:
            segs.append(Segment(0.0, t, v1, 0.0))
        if rise > 0:
            segs.append(Segment(t, t + rise, v1, (v2 - v1) / rise))
        t += rise
        if width > 0:
            segs.append(Segment(t, t + width, v2, 0.0))
        t += width
        if fall > 0:
            segs.append(Segment(t, t + fall, v2, (v1 - v2) / fall))
        t += fall
        # low plateau to period end
        if t < period:
            segs.append(Segment(t, period, v1, 0.0))
        segs = [s for s in segs if s.t1 > s.t0] or [Segment(0.0, period, v1, 0.0)]
        return Waveform(period, segs)

    @staticmethod
    def sine(dc, amp, freq_hz, phase_deg=0.0):
        import math
        w = 2 * math.pi * freq_hz
        return Waveform(None, [Segment(0.0, float('inf'), float(dc), 0.0)],
                        [SinePart(w, float(amp), math.radians(phase_deg))])

    # -- engine interface ----------------------------------------------------
    def segment_at(self, t: float) -> Segment:
        """Absolute Segment containing t (advancing into later cycles)."""
        if self.period is None:
            for s in self.segs:
                if s.t0 <= t < s.t1 or s.t1 == float('inf'):
                    return s
            return self.segs[-1]
        k = int(t // self.period)
        tr = t - k * self.period
        for s in self.segs:
            if s.t0 <= tr < s.t1:
                return Segment(k * self.period + s.t0, k * self.period + s.t1,
                               s.a, s.s)
        # t exactly at a segment/period end -> first segment of next cycle
        s0 = self.segs[0]
        k2 = k + 1
        return Segment(k2 * self.period + s0.t0, k2 * self.period + s0.t1,
                       s0.a, s0.s)

    def next_boundary(self, t: float) -> float:
        """First segment boundary strictly after t."""
        if self.period is None:
            nxt = float('inf')
            for s in self.segs:
                if s.t1 > t and s.t1 < nxt:
                    nxt = s.t1
            return nxt
        k = int(t // self.period)
        tr = t - k * self.period
        for s in self.segs:
            if s.t0 > tr:
                return k * self.period + s.t0
        return (k + 1) * self.period + self.segs[0].t0

    def eval(self, t: float) -> float:
        v = self.segment_at(t).value(t)
        for sn in self.sines:
            v += sn.K * __import__('math').sin(sn.omega * t + sn.phi)
        return v


def waveform_from_spec(spec) -> Waveform:
    """Build a Waveform from YAML: number | {dc:} | {pulse:} | {sine:}."""
    if isinstance(spec, (int, float, str)):
        try:
            return Waveform.dc(eng(spec))
        except ValueError:
            raise ValueError(f"bad source value {spec!r}")
    if isinstance(spec, dict):
        if 'dc' in spec:
            return Waveform.dc(eng(spec['dc']))
        if 'pulse' in spec:
            p = spec['pulse']
            return Waveform.pulse(eng(p['v1']), eng(p['v2']), eng(p['period']),
                                  eng(p.get('width', 0)), eng(p.get('rise', 0)),
                                  eng(p.get('fall', 0)), eng(p.get('delay', 0)))
        if 'sine' in spec:
            s = spec['sine']
            return Waveform.sine(eng(s.get('dc', 0)), eng(s['amp']), eng(s['freq']),
                                 float(s.get('phase_deg', 0)))
    raise ValueError(f"unsupported waveform spec {spec!r}")
