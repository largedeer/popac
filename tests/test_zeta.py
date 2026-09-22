# -*- coding: utf-8 -*-
"""Eighth-topology smoke (Tier 2): Zeta, 12 V -> 5 V / 1 A at 200 kHz.
Tier-2 bar: regulation law + i2 = IOUT + POP/PAC VALIDATED.  No golden
collection (plan P1)."""
import os
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.model_loader import load_yaml
from popac.engine import Engine
from popac.pop import PopSolver
from popac.pac import PacSolver

_M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'models', 'zeta24.yaml')


@lru_cache(maxsize=None)
def _orbit():
    ckt, ana = load_yaml(_M)
    e = Engine(ckt, {'max_events': 5_000_000, 'chunk': 500e-9})
    solver = PopSolver(e, dict(ana.get('pop', {})))
    pop = solver.solve()
    assert pop.ok, f'POP failed: {pop.diag}'
    e.restore(solver.snap0)
    e.x = pop.x.copy()
    t0 = e.t
    s, _ = e.run(t0 + 2 * pop.period, sample_dt=pop.period / 200,
                 probes=['VOUT', 'IL1', 'IL2'])
    per = [v for t, v in s if t >= t0 + pop.period - 1e-15]
    return {'ckt': ckt, 'ana': ana, 'T': pop.period, 'res': pop.residual,
            'maxlam': pop.max_multiplier, 'snap': solver.snap0,
            'x': pop.x.copy(),
            'vout': float(np.mean([v['VOUT'] for v in per])),
            'il1': np.array([v['IL1'] for v in per]),
            'il2': np.array([v['IL2'] for v in per])}


def test_zeta_dc_law():
    o = _orbit()
    ckt = o['ckt']
    vref = next(d for d in ckt.devices if d.name == 'VREF').wave.eval(0.0)
    r17 = next(d for d in ckt.devices if d.name == 'R17').r
    r18 = next(d for d in ckt.devices if d.name == 'R18').r
    r1 = next(d for d in ckt.devices if d.name == 'R1').r
    assert abs(o['vout'] - vref * (r17 + r18) / r18) < 0.15, o['vout']
    i2 = float(np.mean(o['il2']))
    assert abs(i2 - o['vout'] / r1) / i2 < 0.05, i2    # = IOUT


def test_zeta_pop_pac():
    o = _orbit()
    assert abs(o['T'] - 5e-6) < 1e-12, o['T']
    assert o['res'] < 2e-5, o['res']
    pac = PacSolver(o['ckt'], o['snap'], o['x'], dict(o['ana'].get('pac', {})))
    val = pac.solve()['info']['validation']
    assert val['status'] == 'VALIDATED', val


if __name__ == '__main__':
    test_zeta_dc_law()
    print('PASS test_zeta_dc_law')
    test_zeta_pop_pac()
    print('PASS test_zeta_pop_pac')
    print('all zeta tests passed')
