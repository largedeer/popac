# -*- coding: utf-8 -*-
"""YAML model loader: normalized DSL -> Circuit + analyses config.

Schema (models/buck5v.yaml is the reference):
  nodes: []                  # optional documentation
  devices:
    - {kind: R,  name: R1, n1: a, n2: b, r: 100k}
    - {kind: C,  name: C1, n1: a, n2: "0", c: 88u, ic_v: 5}
    - {kind: L,  name: L1, n1: a, n2: b, l: 4.7u, ic_i: 0}
    - {kind: V,  name: V1, n1: a, n2: "0", value: {dc: 12} | {pulse: {...}} | {sine: {...}} | 12}
    - {kind: I,  name: I1, n1: a, n2: "0", value: {dc: 1}}
    - {kind: E,  name: E1, n1: o1, n2: "0", na: i1, nb: "0", k: 1}
    - {kind: G,  name: G1, n1: "0", n2: o, na: i1, nb: i2, gm: 24u}
    - {kind: H,  name: H1, n1: o1, n2: "0", sense_i: L1, k: 107m}
    - {kind: SW, name: S1, d1: a, d2: b, c1: g, c2: "0",
       ron: 1m, roff: 10Meg, threshold: 2, hystwd: 100m, ic: OPEN}
    - {kind: D,  name: D1, a: an, k: ka, vf: 1m, ron: 10m, roff: 1G}
    - {kind: CMP, name: CMP1, out: y, inp: p, inn: n, vol: 0, voh: 5,
       rout: 10, hystwd: 1p, delay: 1n, ic: 0}
    - {kind: GATE, name: U1, out: y, inputs: [a, b], fn: AND,
       vol: 0, voh: 5, rout: 41.67, rin: 10Meg, th: 2.5, hystwd: 0.1,
       delay: 2p, ic: 0}
    - {kind: SRFF, name: U1, q: qn, nq: nqn, s: s, r: r, th: 2.5, ic: 0}
    - {kind: TRIG, name: X1, node: sw, vref: 2.5, edge: rising}
  probes: {VOUT: v(VOUT), IL: i(L1), GH: v(GH)}
  analyses: {...}
"""
from typing import Dict, Any

import yaml

from .units import eng
from .waveforms import waveform_from_spec
from . import ir


def _req(d, key, ctx):
    if key not in d:
        raise ValueError(f"device {ctx}: missing '{key}'")
    return d[key]


def build_device(d: Dict[str, Any]):
    k = d.get('kind')
    n = d.get('name', '?')
    if k == 'R':
        return ir.Res(n, _req(d, 'n1', n), _req(d, 'n2', n), eng(_req(d, 'r', n)))
    if k == 'C':
        ic = d.get('ic_v')
        return ir.Cap(n, _req(d, 'n1', n), _req(d, 'n2', n), eng(_req(d, 'c', n)),
                      eng(ic) if ic is not None else None)
    if k == 'L':
        ic = d.get('ic_i')
        return ir.Ind(n, _req(d, 'n1', n), _req(d, 'n2', n), eng(_req(d, 'l', n)),
                      eng(ic) if ic is not None else 0.0)
    if k in ('V', 'I'):
        cls = ir.VSrc if k == 'V' else ir.ISrc
        wave = waveform_from_spec(d.get('value', 0))
        return cls(n, _req(d, 'n1', n), _req(d, 'n2', n), wave)
    if k == 'E':
        return ir.Vcvs(n, d['n1'], d['n2'], d['na'], d['nb'], eng(d['k']))
    if k == 'G':
        return ir.Vccs(n, d['n1'], d['n2'], d['na'], d['nb'], eng(d['gm']))
    if k == 'H':
        return ir.Ccvs(n, d['n1'], d['n2'], d['sense_i'], eng(d['k']))
    if k == 'K':
        return ir.Mutual(n, _req(d, 'l1', n), _req(d, 'l2', n),
                         eng(_req(d, 'k', n)))
    if k == 'SW':
        return ir.VcSwitch(n, d['d1'], d['d2'], d['c1'], d['c2'],
                           eng(d['ron']), eng(d['roff']),
                           eng(d.get('threshold', 2)), eng(d.get('hystwd', 0.1)),
                           d.get('ic', 'OPEN'))
    if k == 'D':
        return ir.Diode(n, d['a'], d['k'], eng(d.get('vf', '1m')),
                        eng(d.get('ron', '10m')), eng(d.get('roff', '1G')))
    if k == 'CMP':
        return ir.Comparator(n, d['out'], d['inp'], d['inn'],
                             eng(d.get('vol', 0)), eng(d.get('voh', 5)),
                             eng(d.get('rout', 10)), eng(d.get('hystwd', 0)),
                             eng(d.get('delay', 0)), int(d.get('ic', 0)))
    if k == 'GATE':
        return ir.Gate(n, d['out'], list(d['inputs']), d.get('fn', 'BUF'),
                       eng(d.get('vol', 0)), eng(d.get('voh', 5)),
                       eng(d.get('rout', 41.67)), eng(d.get('rin', '10Meg')),
                       eng(d.get('th', 2.5)), eng(d.get('hystwd', 0)),
                       eng(d.get('delay', 0)), int(d.get('ic', 0)))
    if k == 'SRFF':
        return ir.SrLatch(n, d['q'], d['nq'], d['s'], d['r'],
                          eng(d.get('vol', 0)), eng(d.get('voh', 5)),
                          eng(d.get('rout', 10)), eng(d.get('rin', '10Meg')),
                          eng(d.get('th', 2.5)), eng(d.get('hystwd', 0)),
                          eng(d.get('delay', 0)), int(d.get('ic', 0)))
    if k == 'TRIG':
        return ir.PopTrigger(n, d['node'], eng(d.get('vref', 2.5)),
                             d.get('edge', 'rising'))
    raise ValueError(f"unknown device kind {k!r} ({n})")


def _parse_probe(spec: str):
    s = spec.strip()
    if s.startswith('v(') and s.endswith(')'):
        return ('v', s[2:-1])
    if s.startswith('i(') and s.endswith(')'):
        return ('i', s[2:-1])
    return ('v', s)


def load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        doc = yaml.safe_load(f)
    return load_dict(doc)


def load_dict(doc: Dict[str, Any]):
    ckt = ir.Circuit(meta=doc.get('meta', {}))
    for d in doc.get('devices', []):
        ckt.devices.append(build_device(d))
    ckt.probes = {name: _parse_probe(spec)
                  for name, spec in doc.get('probes', {}).items()}
    errs = ckt.validate()
    if errs:
        raise ValueError("model validation errors:\n  " + "\n  ".join(errs))
    return ckt, doc.get('analyses', {})
