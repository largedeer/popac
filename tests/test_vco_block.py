# -*- coding: utf-8 -*-
"""P2 precondition (TOPOLOGY_CAMPAIGN_PLAN section 3): the VCO block.
f = gm*vctrl/(2*C*dV) built from existing devices; the triangle state
is continuous and periodic -> POP signature semantics untouched.  The
PAC VALIDATED check on a standalone oscillator exercises the saltation
machinery on genuine threshold crossings (the r1_term_probe class)."""
import os
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.model_loader import load_yaml
from popac.waveforms import Waveform
from popac.engine import Engine
from popac.pop import PopSolver
from popac.pac import PacSolver

_M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'models', 'vco_block.yaml')

GM = 80e-6
CC = 100e-12
DV = 4.0


def _f_formula(v):
    return GM * v / (2 * CC * DV)          # gm*vctrl/(2*C*dV)


def _orbit(vctl):
    ckt, ana = load_yaml(_M)
    for d in ckt.devices:
        if getattr(d, 'name', '') == 'VCTL':
            d.wave = Waveform.dc(vctl)      # flyback K1-style device patch
    e = Engine(ckt, {'max_events': 5_000_000, 'chunk': 100e-9})
    solver = PopSolver(e, dict(ana.get('pop', {})))
    pop = solver.solve()
    assert pop.ok, f'POP failed at vctl={vctl}: {pop.diag}'
    e.restore(solver.snap0)
    e.x = pop.x.copy()
    t0 = e.t
    s, _ = e.run(t0 + 2 * pop.period, sample_dt=pop.period / 500,
                 probes=['PH', 'CLK', 'CTL'])
    per = [v for t, v in s if t >= t0 + pop.period - 1e-15]
    ph = np.array([v['PH'] for v in per])
    return {'ckt': ckt, 'ana': ana, 'T': pop.period,
            'res': pop.residual, 'maxlam': pop.max_multiplier,
            'snap': solver.snap0, 'x': pop.x.copy(), 'ph': ph,
            'n': pop.iterations}


@lru_cache(maxsize=None)
def _at(vctl):
    return _orbit(vctl)


def test_vco_frequency_linearity():
    for v in (0.5, 1.0, 1.5):
        o = _at(v)
        f_meas = 1.0 / o['T']
        f_pred = _f_formula(v)
        assert abs(f_meas - f_pred) / f_pred < 0.005, (v, f_meas, f_pred)
        # triangle amplitude spans the hysteresis band
        assert abs(o['ph'].max() - 4.5) < 0.05, o['ph'].max()
        assert abs(o['ph'].min() - 0.5) < 0.05, o['ph'].min()


def test_vco_pop_period():
    o = _at(1.0)
    assert abs(o['T'] - 1.0 / _f_formula(1.0)) / (1.0 / _f_formula(1.0)) \
        < 0.002, o['T']
    assert o['res'] < 1e-8, o['res']


def test_vco_pac_validated():
    o = _at(1.0)
    pac = PacSolver(o['ckt'], o['snap'], o['x'], dict(o['ana'].get('pac', {})))
    val = pac.solve()['info']['validation']
    assert val['status'] == 'VALIDATED', val


if __name__ == '__main__':
    test_vco_frequency_linearity()
    print('PASS test_vco_frequency_linearity')
    test_vco_pop_period()
    print('PASS test_vco_pop_period')
    test_vco_pac_validated()
    print('PASS test_vco_pac_validated')
    print('all vco block tests passed')
