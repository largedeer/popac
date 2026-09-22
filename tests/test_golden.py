# -*- coding: utf-8 -*-
"""Stage R4: Tier A golden regression (plan §5.2/§6/R4).

The goldens under tests/golden/ are the EXECUTABLE SPEC of the Python
reference: every engine/PAC change must reproduce them within the
tolerances below.  If a change legitimately moves a golden, regenerate
with `python debug/make_golden.py <name>` and say why in the commit --
never edit the JSON by hand, never delete an assertion to make it pass.

Tolerances (same code/machine reproduces bitwise; these absorb
cross-BLAS/OS drift): x* 1e-9 abs+rel, Floquet/PAC spectra 1e-6 rel,
cluster times 1e-12 s + 1e-9*T, residual within 3x (convergence metric,
not a state).  The validation max_col_rel is noise-sensitive, so only
its STATUS must match and both sides must sit below the gate tolerance.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'debug'))

from golden_util import GOLDEN_ROOT, SCHEMA_VERSION, streams_match
from make_golden import MODELS, build

X_ATOL = X_RTOL = 1e-9
SPEC_RTOL = 1e-6
SPEC_ATOL = 1e-9
RESID_FACTOR = 3.0

# Golden tiering (TOPOLOGY_CAMPAIGN_PLAN section 4, triggered when the
# full gate crossed 60 min at 90 items): Tier A runs in every gate
# (the four models that anchor the engine semantics -- PS15 both
# fidelities, boost24, pacmini); Tier B is the topology-collection
# models (buck2ph/sepic24/flyback24/buckboost24/cuk24), excluded from
# the fast lane via `pytest -m "not tierb"` and covered by the full
# gate / pre-release runs.  NO tolerance or assertion changes -- only
# scheduling.  The __main__ direct-run mode still runs everything.
import pytest

_TIER_B = {'buck2ph', 'sepic24', 'flyback24', 'buckboost24', 'cuk24'}


def _load(name):
    path = os.path.join(GOLDEN_ROOT, name, 'golden.json')
    assert os.path.exists(path), (
        f'golden for {name} missing: regenerate with '
        f'"python debug/make_golden.py {name}" and document why')
    g = json.loads(open(path, encoding='utf-8').read())
    assert g['schema_version'] == SCHEMA_VERSION, g['schema_version']
    return g


def _compare(name, fresh, gold):
    tag = f'[{name}] '
    # provenance: the model file itself must not have drifted silently
    assert fresh['model_sha256'] == gold['model_sha256'], \
        tag + 'model YAML changed: regenerate + document'
    for k in ('fidelity_declared', 'fidelity_effective',
              'event_unresolved_policy'):
        assert fresh['policy'][k] == gold['policy'][k], tag + k
    assert fresh['policy']['pop_opts'] == gold['policy']['pop_opts'], \
        tag + 'pop opts'
    # pop
    fp, gp = fresh['pop'], gold['pop']
    assert fp['ok'] and gp['ok']
    assert fp['iterations'] == gp['iterations'], \
        tag + f"iterations {fp['iterations']} vs {gp['iterations']}"
    assert abs(fp['period'] - gp['period']) <= 1e-12, tag + 'period'
    assert max(fp['residual'], gp['residual']) \
        <= RESID_FACTOR * max(min(fp['residual'], gp['residual']), 1e-300), \
        tag + f"residual {fp['residual']:g} vs {gp['residual']:g}"
    assert fp['discrete_match'] and gp['discrete_match'], tag + 'dmatch'
    assert abs(fp['max_multiplier'] - gp['max_multiplier']) \
        <= SPEC_ATOL + SPEC_RTOL * gp['max_multiplier'], tag + 'maxlam'
    assert np.allclose(fp['x_star'], gp['x_star'],
                       atol=X_ATOL, rtol=X_RTOL), tag + 'x*'
    assert np.allclose(fp['floquet_abs'], gp['floquet_abs'],
                       atol=SPEC_ATOL, rtol=SPEC_RTOL), tag + 'floquet'
    assert fp['floquet_isolated'] == gp['floquet_isolated'], tag + 'iso mask'
    for k in ('sw', 'dio', 'cmp', 'srff_q'):
        assert fp['boundary_discrete'][k] == gp['boundary_discrete'][k], \
            tag + 'boundary ' + k
    assert fp['boundary_discrete']['trig_armed'] \
        == gp['boundary_discrete']['trig_armed'], tag + 'trig_armed'
    # event stream (plan §5.2 tolerances)
    ok, msg = streams_match(fresh['event_stream'], gold['event_stream'],
                            gp['period'])
    assert ok, tag + 'event stream: ' + msg
    # tripwire: budget-class unresolved spans are a construction defect
    assert fresh['event_report']['budget'] == 0, tag + 'budget spans'
    assert gold['event_report']['budget'] == 0, tag + 'golden budget spans'
    # pac
    fq, gq = fresh['pac'], gold['pac']
    assert np.array_equal(np.array(fq['freqs']), np.array(gq['freqs'])), \
        tag + 'freq grid'
    for side in ('T_re', 'T_im'):
        a, b = np.array(fq[side]), np.array(gq[side])
        assert np.array_equal(np.isnan(a), np.isnan(b)), tag + side + ' NaNs'
        m = ~np.isnan(a)
        assert np.allclose(a[m], b[m], atol=SPEC_ATOL, rtol=SPEC_RTOL), \
            tag + side
    assert fq['validation']['status'] == gq['validation']['status'], \
        tag + 'PAC validation status'
    assert fq['validation']['max_col_rel'] < fq['validation']['tol'] and \
        gq['validation']['max_col_rel'] < gq['validation']['tol'], \
        tag + 'PAC validation rel above tol'
    assert fq['trigger_saltation'] == gq['trigger_saltation'], \
        tag + 'trigger_saltation'
    assert np.allclose(fq['floquet_abs'], gq['floquet_abs'],
                       atol=SPEC_ATOL, rtol=SPEC_RTOL), tag + 'PAC floquet'
    assert np.allclose(fq['floquet_frozen_abs'], gq['floquet_frozen_abs'],
                       atol=SPEC_ATOL, rtol=SPEC_RTOL), tag + 'PAC frozen'


def _case(name):
    g = _load(name)
    m = next(mm for mm in MODELS if mm[1] == name)
    fresh = build(m[0], m[2], m[3])
    _compare(name, fresh, g)


def test_golden_pacmini():
    _case('pacmini')


def test_golden_buck5v():
    _case('buck5v')


def test_golden_boost24():
    _case('boost24')


@pytest.mark.tierb
def test_golden_buck2ph():
    _case('buck2ph')


@pytest.mark.tierb
def test_golden_sepic24():
    _case('sepic24')


@pytest.mark.tierb
def test_golden_flyback24():
    _case('flyback24')


@pytest.mark.tierb
def test_golden_buckboost24():
    _case('buckboost24')


@pytest.mark.tierb
def test_golden_cuk24():
    _case('cuk24')


def test_golden_manifest_current():
    """Every golden dir is in the manifest and every manifest entry exists
    (no orphaned/stale goldens)."""
    mpath = os.path.join(GOLDEN_ROOT, 'manifest.json')
    man = json.loads(open(mpath, encoding='utf-8').read())
    dirs = set(d for d in os.listdir(GOLDEN_ROOT)
               if os.path.isdir(os.path.join(GOLDEN_ROOT, d)))
    assert dirs == set(man['models']), (dirs, set(man['models']))
    for name, entry in man['models'].items():
        g = _load(name)
        assert entry['model_sha256'] == g['model_sha256'], name


if __name__ == '__main__':
    for n in ('pacmini', 'buck5v',
              'boost24', 'buck2ph', 'sepic24', 'flyback24', 'buckboost24',
              'cuk24'):
        _case(n)
        print(f'PASS test_golden_{n}')
    test_golden_manifest_current()
    print('PASS test_golden_manifest_current')
    print('all golden tests passed')
