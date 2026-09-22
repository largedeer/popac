# -*- coding: utf-8 -*-
"""Second-topology generality regression (HANDOFF §9.4-2 / §10).

boost24: peak-current-mode asynchronous boost (12 V -> ~24.5 V / 2 A at
200 kHz), peak-CMC control on a boost power stage.  Pins
the DSL/POP/PAC pipeline on a topology the engine was not tuned around:

- POP converges with the exact switching period and a physical Floquet
  spectrum (dominant slow mode = the ~0.99 output-pole mode, stable)
- PAC's Psi-vs-FD validity gate passes (max col rel ~2.5e-5, 20x tighter
  than PS15) and the loop metrics land where the design puts them
  (fc ~0.7 kHz: gentle EA, PM ~28 deg)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from popac.model_loader import load_yaml
from popac.engine import Engine
from popac.pop import PopSolver
from popac.pac import PacSolver
from popac.ac import loop_metrics


def _boost():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return load_yaml(os.path.join(here, 'models', 'boost24.yaml'))


def test_boost_pop_and_pac():
    ckt, ana = _boost()
    e = Engine(ckt, {'max_events': 5_000_000, 'chunk': 500e-9})
    pop = PopSolver(e, dict(ana.get('pop', {}))).solve()
    assert pop.ok, f'boost POP failed: {pop.diag}'
    assert abs(pop.period - 5e-6) < 1e-12, pop.period   # exact T_sw
    # steady-state boundary values at the design point
    e.restore(getattr(pop, 'snap0', None) or e.snapshot())
    e.x = pop.x.copy()
    vout = e.probe_now('v', 'vout')
    assert abs(vout - 24.5) < 0.3, f'VOUT boundary {vout}'
    # physical Floquet: slow output-pole mode near unity but < 1, stable
    assert 0.985 < pop.max_multiplier < 0.999, pop.max_multiplier

    pac = PacSolver(ckt, e.snapshot(), pop.x,
                    {'fstart': 30.0, 'fstop': 180e3, 'per_decade': 10,
                     'inject_src': 'V16', 'probe_a': 'vout',
                     'probe_b': 'fbt'})
    res = pac.solve()
    val = res['info']['validation']
    assert val['status'] == 'VALIDATED', val
    m = loop_metrics(res['freqs'], res['T'])
    assert m['crossovers'], 'no 0 dB crossing found'
    fc = m['crossovers'][0]
    # gentle-EA design: low-bandwidth loop (0.3..1.5 kHz), modest PM
    assert 300 < fc['freq'] < 1500, fc
    assert 15 < fc['pm_deg'] < 45, fc


if __name__ == '__main__':
    test_boost_pop_and_pac()
    print('PASS test_boost_pop_and_pac')
    print('all second-topology tests passed')
