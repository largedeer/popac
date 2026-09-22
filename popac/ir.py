# -*- coding: utf-8 -*-
"""Circuit intermediate representation.

Device set covers the PS15 benchmark (plan §7.1 first priority):
R, C, L, V/I sources (dc/pulse/sine), VCVS/VCCS/CCVS, voltage-controlled
switch, PWL diode, comparator, BUF/INV/AND/OR gates, set-dominant SR latch,
POP trigger, probes.

Conventions
-----------
* Ground node is the string "0".
* Branch currents (V source, VCVS output, CCVS output, inductor) are positive
  flowing from the first-listed terminal into the device.
* Controlled-source sense: 'sense_v' = (n+, n-) node pair, or
  'sense_i' = name of an inductor / 0V-ammeter whose branch current is sensed.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .units import eng
from .waveforms import Waveform, waveform_from_spec

GND = "0"


# ---------------------------------------------------------------- passive
@dataclass
class Res:
    name: str; n1: str; n2: str; r: float
    kind = "R"


@dataclass
class Cap:
    name: str; n1: str; n2: str; c: float; ic_v: float = None
    kind = "C"


@dataclass
class Ind:
    name: str; n1: str; n2: str; l: float; ic_i: float = 0.0
    kind = "L"


@dataclass
class Mutual:
    """Linear mutual inductance between two named inductors (plan T4).

    M = k*sqrt(L1*L2) with |k| < 1 (keeps the inductance matrix positive
    definite).  DOT POLARITY: the dot of each winding is its n1 end --
    currents defined n1->n2 on both inductors give the +M sign; k < 0 is
    the reversed-dot solution (equivalently, swap one winding's pins).
    An inductor may appear in at most ONE Mutual: repeated coupling of
    the same winding would need a general multi-winding matrix and is
    rejected at validation.
    """
    name: str; l1: str; l2: str; k: float
    kind = "K"


# ---------------------------------------------------------------- sources
@dataclass
class VSrc:
    name: str; n1: str; n2: str; wave: Waveform
    kind = "V"


@dataclass
class ISrc:
    name: str; n1: str; n2: str; wave: Waveform
    kind = "I"


# --------------------------------------------------------- controlled srcs
@dataclass
class Vcvs:
    name: str; n1: str; n2: str            # output +/-
    na: str; nb: str                       # control +/-
    k: float
    kind = "E"


@dataclass
class Vccs:
    name: str; n1: str; n2: str            # current injected into n2 (from n1)
    na: str; nb: str
    gm: float
    kind = "G"


@dataclass
class Ccvs:
    name: str; n1: str; n2: str            # output voltage = k * i(sense)
    sense_i: str                           # device name (Ind or VSrc ammeter)
    k: float
    kind = "H"


# ---------------------------------------------------------------- switching
@dataclass
class VcSwitch:
    """Voltage-controlled switch: on when v(c1)-v(c2) crosses thresholds.

    Off -> on at THRESHOLD + HYSTWD/2, on -> off at THRESHOLD - HYSTWD/2.
    In this engine the control node is almost always driven by a digital
    output (VOL/VOH), so the state is derived from the driving logic value
    whenever the control source is digital; the analog threshold path exists
    for genuinely analog control.
    """
    name: str; d1: str; d2: str; c1: str; c2: str
    ron: float; roff: float; threshold: float = 2.0; hystwd: float = 0.1
    ic: str = "OPEN"
    kind = "SW"


@dataclass
class Diode:
    """Two-segment PWL diode: OFF below vf (resistance roff), ON above
    (resistance ron in series with vf)."""
    name: str; a: str; k: str
    vf: float = 0.001; ron: float = 0.01; roff: float = 1e9
    kind = "D"


# ---------------------------------------------------------------- digital
@dataclass
class Comparator:
    """Analog comparator with input hysteresis and output propagation delay.
    Output is a Thevenin source VOL/VOH with ROUT driving single-ended `out`
    (referenced to ground)."""
    name: str; out: str; inp: str; inn: str
    vol: float = 0.0; voh: float = 5.0; rout: float = 10.0
    hystwd: float = 0.0; delay: float = 0.0; ic: int = 0
    kind = "CMP"


@dataclass
class Gate:
    """1-2 input gate: 'BUF' | 'INV' | 'AND' | 'OR'. Single-ended output."""
    name: str; out: str; inputs: List[str]
    fn: str = "BUF"; vol: float = 0.0; voh: float = 5.0
    rout: float = 41.67; rin: float = 1e7; th: float = 2.5
    hystwd: float = 0.0; delay: float = 0.0; ic: int = 0
    kind = "GATE"


@dataclass
class SrLatch:
    """Set-dominant level-sensitive SR flip-flop with Q and /Q outputs."""
    name: str; q: str; nq: str; s: str; r: str
    vol: float = 0.0; voh: float = 5.0; rout: float = 10.0
    rin: float = 1e7; th: float = 2.5; hystwd: float = 0.0
    delay: float = 0.0; ic: int = 0
    kind = "SRFF"


@dataclass
class PopTrigger:
    """POP trigger: rising/falling crossing of VREF on node `node`."""
    name: str; node: str; vref: float = 2.5
    edge: str = "rising"      # 'rising' | 'falling'
    kind = "TRIG"


# ---------------------------------------------------------------- circuit
@dataclass
class Circuit:
    devices: List[object] = field(default_factory=list)
    probes: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # name -> ('v', node) | ('i', dev)
    gmin: float = 1e-12
    meta: Dict[str, str] = field(default_factory=dict)

    def by_kind(self, k):
        return [d for d in self.devices if d.kind == k]

    def find(self, name):
        for d in self.devices:
            if d.name == name:
                return d
        raise KeyError(name)

    def nodes(self) -> List[str]:
        """All non-ground nodes (sorted for determinism)."""
        ns = set()
        pin_attrs = {
            'R': ('n1', 'n2'), 'C': ('n1', 'n2'), 'L': ('n1', 'n2'),
            'V': ('n1', 'n2'), 'I': ('n1', 'n2'),
            'E': ('n1', 'n2', 'na', 'nb'), 'G': ('n1', 'n2', 'na', 'nb'),
            'H': ('n1', 'n2'),
            'SW': ('d1', 'd2', 'c1', 'c2'), 'D': ('a', 'k'),
            'CMP': ('out', 'inp', 'inn'),
            'GATE': ('out',), 'SRFF': ('q', 'nq'), 'TRIG': ('node',),
        }
        for d in self.devices:
            for a in pin_attrs.get(d.kind, ()):
                v = getattr(d, a)
                if isinstance(v, list):
                    ns.update(x for x in v if x != GND)
                elif v != GND:
                    ns.add(v)
        return sorted(ns)

    # ---------------- validation ----------------
    def validate(self) -> List[str]:
        errs = []
        names = [d.name for d in self.devices]
        if len(names) != len(set(names)):
            dup = {n for n in names if names.count(n) > 1}
            errs.append(f"duplicate device names: {sorted(dup)}")
        for d in self.by_kind('H'):
            try:
                self.find(d.sense_i)
            except KeyError:
                errs.append(f"{d.name}: sense device '{d.sense_i}' not found")
                continue
            sd = self.find(d.sense_i)
            if sd.kind not in ('L', 'V'):
                errs.append(f"{d.name}: sense must be an inductor or ammeter, got {sd.kind}")
        coupled = {}
        for d in self.by_kind('K'):
            if d.l1 == d.l2:
                errs.append(f"{d.name}: cannot couple an inductor to itself")
            for ln in (d.l1, d.l2):
                try:
                    sd = self.find(ln)
                except KeyError:
                    errs.append(f"{d.name}: inductor '{ln}' not found")
                    continue
                if sd.kind != 'L':
                    errs.append(f"{d.name}: '{ln}' is a {sd.kind}, not an inductor")
                if ln in coupled:
                    errs.append(
                        f"{d.name}: inductor '{ln}' already coupled by "
                        f"{coupled[ln]} (repeated coupling -> illegal matrix)")
                coupled[ln] = d.name
            if not abs(d.k) < 1.0:
                errs.append(
                    f"{d.name}: |k|={abs(d.k):g} must be < 1 "
                    f"(inductance matrix positive definite)")
        node_set = set(self.nodes()) | {GND}
        for pname, (ptype, ref) in self.probes.items():
            if ptype == 'v' and ref not in node_set:
                errs.append(f"probe {pname}: unknown node {ref}")
            if ptype == 'i':
                try:
                    self.find(ref)
                except KeyError:
                    errs.append(f"probe {pname}: unknown device {ref}")
        return errs
