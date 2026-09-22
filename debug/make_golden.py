"""Stage R4 (plan §6/R4 + §5.2): freeze the Tier A goldens.

Per model: POP + one-cycle canonical event stream + PAC sweep, serialized
at full float precision into tests/golden/<name>/golden.json.  The golden
is the EXECUTABLE SPEC of the Python reference implementation: engine
changes must not move it, and regenerating requires a documented reason
in the commit that does it.

  python debug/make_golden.py [name ...]     # names = golden dir names
"""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'tests'))

import numpy as np
import scipy

from popac.model_loader import load_yaml
from popac.engine import Engine
from popac.pop import PopSolver
from popac.pac import PacSolver
from popac.mna import effective_fidelity
from golden_util import canonical_stream, GOLDEN_ROOT, SCHEMA_VERSION

# (model yaml, golden name, engine opts, PAC sweep override)
# Sweep configs are FIXED policy: lean sweeps so the golden test stays
# minutes-scale; inject/probe nodes come from the model's own YAML.
MODELS = [
    ('buck5v.yaml', 'buck5v', {}, None),
    ('boost24.yaml', 'boost24', {'chunk': 500e-9},
     {'fstart': 30.0, 'fstop': 180e3, 'per_decade': 10}),
    ('pacmini.yaml', 'pacmini', {}, None),   # keep the model's own sweep
    # Tier B (plan §275: join after verification): buck2ph verified in
    # HANDOFF §15.2 (61-test gate); sepic24 in §16 (66-test gate);
    # flyback24 in §18 (76-test gate); YAML carries its own PAC sweep
    ('buck2ph.yaml', 'buck2ph', {}, None),
    ('sepic24.yaml', 'sepic24', {'chunk': 500e-9}, None),
    ('flyback24.yaml', 'flyback24', {'chunk': 1e-6}, None),
    ('buckboost24.yaml', 'buckboost24', {'chunk': 500e-9}, None),
    ('cuk24.yaml', 'cuk24', {'chunk': 500e-9}, None),
]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(model_yaml, eopt, pac_sweep):
    """All golden content for one model, as a plain-json dict."""
    ckt, ana = load_yaml(str(HERE / 'models' / model_yaml))
    pop_opts = dict(ana.get('pop', {}))
    pac_opts = dict(ana.get('pac', {}))
    if pac_sweep:
        pac_opts.update(pac_sweep)

    e = Engine(ckt, {'max_events': 20_000_000, **eopt})
    solver = PopSolver(e, pop_opts)
    pop = solver.solve()
    if not pop.ok:
        raise SystemExit(f'{model_yaml}: POP failed: {pop.diag}')

    # one canonical cycle from the boundary: event stream + tripwires
    e2 = Engine(ckt, {'max_events': 20_000_000, **eopt})
    e2.restore(solver.snap0)
    e2.x = pop.x.copy()
    t0 = e2.t
    recs = []
    e2.run(t0 + pop.period, rec=recs.append, stop_on_trigger=True)

    snap = solver.snap0
    g = {
        'schema_version': SCHEMA_VERSION,
        'model': model_yaml,
        'model_sha256': _sha256(HERE / 'models' / model_yaml),
        'versions': {'python': platform.python_version(),
                     'numpy': np.__version__, 'scipy': scipy.__version__},
        'policy': {
            'engine_opts': {k: v for k, v in eopt.items()},
            'pop_opts': pop_opts, 'pac_opts': pac_opts,
            'fidelity_declared': ckt.meta.get('fidelity', 'UNDECLARED'),
            'fidelity_effective': effective_fidelity(
                ckt.meta.get('fidelity', 'UNDECLARED'), e.tc.clamps),
            'clamps': e.tc.clamps,
            'event_unresolved_policy': e.event_unresolved_policy,
        },
        'pop': {
            'ok': True, 'iterations': pop.iterations,
            'period': float(pop.period),
            'residual': float(pop.residual),
            'discrete_match': bool(pop.discrete_match),
            'max_multiplier': float(pop.max_multiplier),
            'n_states': len(pop.x),
            'x_star': [float(v) for v in pop.x],
            # pop.floquet holds the COMPLEX eigenvalues; the golden
            # stores magnitudes (same quantity the reports use)
            'floquet_abs': [float(abs(v)) for v in pop.floquet],
            'floquet_isolated': ([bool(v) for v in pop.floquet_isolated]
                                 if pop.floquet_isolated is not None else []),
            'boundary_discrete': {
                'sw': {k: int(v) for k, v in snap['sw'].items()},
                'dio': {k: int(v) for k, v in snap['dio'].items()},
                'cmp': {k: int(v) for k, v in snap['cmp'].items()},
                'srff_q': {k: int(v) for k, v in snap['srff_q'].items()},
                'trig_armed': bool(snap['trig_armed']),
            },
        },
        'event_stream': canonical_stream(recs, t0),
        'event_report': e2.event_report(),
    }

    # PAC on exactly the same boundary (validation gate included)
    e3 = Engine(ckt, {'max_events': 20_000_000, **eopt})
    e3.restore(snap)
    e3.x = pop.x.copy()
    pres = PacSolver(ckt, e3.snapshot(), pop.x, pac_opts).solve()
    val = pres['info']['validation']
    g['pac'] = {
        'freqs': [float(f) for f in pres['freqs']],
        'T_re': [float(v) for v in np.real(pres['T'])],
        'T_im': [float(v) for v in np.imag(pres['T'])],
        'validation': {'status': val['status'],
                       'max_col_rel': float(val['max_col_rel']),
                       'tol': float(val['tol'])},
        'trigger_saltation': bool(pres['info']['trigger_saltation']),
        'floquet_abs': [float(v) for v in pres['info']['floquet_abs']],
        'floquet_frozen_abs': [float(v)
                               for v in pres['info']['floquet_frozen_abs']],
    }
    return g


def main():
    names = [a for a in sys.argv[1:]]
    sel = [m for m in MODELS if not names or m[1] in names]
    if not sel:
        raise SystemExit(f'no model matches {names}; '
                         f'known: {[m[1] for m in MODELS]}')
    manifest = {}
    for model_yaml, name, eopt, sweep in sel:
        print(f'=== golden {name} ({model_yaml})', flush=True)
        g = build(model_yaml, eopt, sweep)
        d = Path(GOLDEN_ROOT) / name
        d.mkdir(parents=True, exist_ok=True)
        (d / 'golden.json').write_text(
            json.dumps(g, indent=1, sort_keys=False), encoding='utf-8')
        manifest[name] = {
            'model': model_yaml, 'model_sha256': g['model_sha256'],
            'period': g['pop']['period'],
            'residual': g['pop']['residual'],
            'pac_status': g['pac']['validation']['status'],
            'events': len(g['event_stream']),
        }
        print(f'    T={g["pop"]["period"]*1e6:.6f}us '
              f'res={g["pop"]["residual"]:.3e} '
              f'events={len(g["event_stream"])} '
              f'PAC={g["pac"]["validation"]["status"]} '
              f'({g["pac"]["validation"]["max_col_rel"]:.2e})', flush=True)
    # merge into the manifest (keep other entries)
    mpath = Path(GOLDEN_ROOT) / 'manifest.json'
    allm = json.loads(mpath.read_text(encoding='utf-8')) \
        if mpath.exists() else {'schema_version': SCHEMA_VERSION}
    allm.setdefault('models', {}).update(manifest)
    allm['schema_version'] = SCHEMA_VERSION
    mpath.write_text(json.dumps(allm, indent=1), encoding='utf-8')
    print('manifest updated:', mpath, flush=True)


if __name__ == '__main__':
    main()
