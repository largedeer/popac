"""Stage R3 (CPP_MIGRATION_AND_TOPOLOGY_PLAN §6/R3): repeatable
performance baseline.

Rules (plan §6/R3):
  - fixed models, fixed solver policy, fixed output sampling -- nothing
    about the workload may drift between runs;
  - warm-up 1 + REPS timed reps per stage, report median / p90 / min
    wall (never a single wall-clock as migration evidence);
  - peak memory from a SEPARATE tracemalloc rep (tracemalloc skews
    timing ~2x, so it never runs inside a timed rep);
  - cold start = fresh subprocess, import + first POP, one shot;
  - cProfile attribution of one extra rep into the five hot buckets
    (MNA / propagate / monitors / event search / event apply);
  - manifest: platform, cores, BLAS, python/numpy/scipy versions.

Usage:
  python debug/benchmark.py [--quick] [model ...]
  --quick : 2 reps, no profile/cold-start (smoke only, NOT evidence)
Output: out/benchmark/bench_<model>.json + a printed summary table.

Timing metric: PRIMARY = process CPU time (time.process_time, sums all
threads) -- wall clock on this machine swings +/-30-75% between runs
under background load waves (iCloud sync, thermals; measured 2026-09-21,
out/benchmark/run1.log.bak vs run2.log), which would make any migration
Go/No-Go meaningless.  Wall is recorded as secondary.  BLAS threads are
PINNED to 1 (set before numpy import; recorded in the manifest) so the
workload is fixed.  The repeatability gate (<10%) is evaluated on CPU
medians.
"""
import cProfile
import io
import json
import os
import pstats
import subprocess
import sys
import time
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import platform
import scipy

from popac.model_loader import load_yaml
from popac.engine import Engine
from popac.pop import PopSolver
from popac.pac import PacSolver

OUT = Path(__file__).resolve().parent.parent / 'out' / 'benchmark'

# fixed workloads: (model, transient horizon, transient sample_dt,
#                   engine opts, PAC sweep)
MODELS = [
    ('buck5v.yaml', 300e-6, 50e-9, {},
     dict(fstart=3e3, fstop=180e3, per_decade=10, inject_src='V16',
          probe_a='vout', probe_b='fbt')),
    ('boost24.yaml', 150e-6, 100e-9, {'chunk': 500e-9},
     dict(fstart=30.0, fstop=180e3, per_decade=10, inject_src='V16',
          probe_a='vout', probe_b='fbt')),
    ('pacmini.yaml', 50e-6, 20e-9, {}, None),
]

# pstats bucket rules: (bucket, file-substring, func-substring)
BUCKETS = [
    ('mna_compile', 'mna.py', ''),
    ('propagate', 'engine.py', '_propagate'),
    ('propagate', 'engine.py', '_eig'),
    ('propagate', 'engine.py', '_expm'),
    ('propagate', 'expm', ''),
    ('monitors', 'engine.py', '_h'),
    ('monitors', 'engine.py', '_active_monitors'),
    ('monitors', 'engine.py', '_hdd_bound'),
    ('event_search', 'engine.py', '_find_crossing'),
    ('event_search', 'engine.py', '_bisect'),
    ('event_apply', 'engine.py', '_apply_dev'),
    ('event_apply', 'engine.py', '_fire_known'),
    ('event_apply', 'engine.py', '_wire_logic'),
    ('event_apply', 'engine.py', '_commit_state'),
]


def manifest():
    cfg = {}
    try:                      # numpy >= 2: structured config dict
        cfg = np.__config__.CONFIG
    except AttributeError:
        pass
    blas = 'unknown'
    try:
        kv = json.dumps(cfg)
        for name in ('openblas', 'mkl', 'blis'):
            if name in kv.lower():
                blas = name
                break
    except TypeError:
        pass
    return {
        'platform': platform.platform(),
        'processor': platform.processor() or 'n/a',
        'cores': os.cpu_count(),
        'blas_threads_pinned': os.environ.get('OPENBLAS_NUM_THREADS'),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'blas': blas,
    }


def _load(model, eopt):
    ckt, ana = load_yaml(str(Path(__file__).resolve().parent.parent
                             / 'models' / model))
    return ckt, ana, {'max_events': 20_000_000, **eopt}


def stage_transient(ckt, ana, eopt, tstop, dt):
    e = Engine(ckt, dict(eopt))
    e.run(tstop, sample_dt=dt, probes=list(ckt.probes))


def stage_pop(ckt, ana, eopt, holder=None):
    e = Engine(ckt, dict(eopt))
    solver = PopSolver(e, dict(ana.get('pop', {})))
    pop = solver.solve()
    if holder is not None:
        holder['snap0'] = solver.snap0   # PopResult has no snap0 field
    return pop


def stage_pac(ckt, eopt, pop, snap0, pacopt):
    e = Engine(ckt, dict(eopt))
    e.restore(snap0)
    e.x = pop.x.copy()
    PacSolver(ckt, e.snapshot(), pop.x, dict(pacopt)).solve()


def _stats(ts):
    a = np.array(ts)
    return {'median': float(np.median(a)), 'p90': float(np.percentile(a, 90)),
            'min': float(a.min()), 'max': float(a.max()), 'n': len(ts)}


def _timed(fn, *a):
    """(cpu_seconds, wall_seconds) -- cpu is the primary metric."""
    c0, w0 = time.process_time(), time.perf_counter()
    fn(*a)
    return (time.process_time() - c0, time.perf_counter() - w0)


def _stage_stats(pairs):
    return {'cpu': _stats([p[0] for p in pairs]),
            'wall': _stats([p[1] for p in pairs])}


def bench_model(model, tstop, dt, eopt, pacopt, reps, quick=False):
    ckt, ana, eopt = _load(model, eopt)
    res = {'model': model, 'stages': {}}

    # --- transient + POP: timed reps (warm-up first, untimed)
    stage_transient(ckt, ana, eopt, tstop, dt)
    pairs = [_timed(stage_transient, ckt, ana, eopt, tstop, dt)
             for _ in range(reps)]
    res['stages']['transient'] = _stage_stats(pairs)
    print(f'  transient  cpu={res["stages"]["transient"]["cpu"]["median"]:8.2f}s'
          f' wall={res["stages"]["transient"]["wall"]["median"]:8.2f}s',
          flush=True)

    stage_pop(ckt, ana, eopt)
    pairs = []
    for _ in range(reps):
        c0, w0 = time.process_time(), time.perf_counter()
        stage_pop(ckt, ana, eopt)
        pairs.append((time.process_time() - c0, time.perf_counter() - w0))
    res['stages']['pop'] = _stage_stats(pairs)
    print(f'  pop        cpu={res["stages"]["pop"]["cpu"]["median"]:8.2f}s'
          f' wall={res["stages"]["pop"]["wall"]["median"]:8.2f}s', flush=True)

    # --- PAC on a fresh converged orbit (validation included, x3 eps)
    if pacopt is not None:
        holder = {}
        pop = stage_pop(ckt, ana, eopt, holder)
        stage_pac(ckt, eopt, pop, holder['snap0'], pacopt)
        pairs = []
        for _ in range(reps):
            pop = stage_pop(ckt, ana, eopt, holder)
            c0, w0 = time.process_time(), time.perf_counter()
            stage_pac(ckt, eopt, pop, holder['snap0'], pacopt)
            pairs.append((time.process_time() - c0,
                          time.perf_counter() - w0))
        res['stages']['pac'] = _stage_stats(pairs)
        print(f'  pac        cpu={res["stages"]["pac"]["cpu"]["median"]:8.2f}s'
              f' wall={res["stages"]["pac"]["wall"]["median"]:8.2f}s',
              flush=True)

    if not quick:
        # --- peak memory (separate rep, untimed)
        import tracemalloc
        tracemalloc.start()
        stage_pop(ckt, ana, eopt)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        res['pop_peak_mem_mb'] = peak / 1e6

        # --- cProfile bucket attribution of one POP rep.  Buckets use
        # SELF time (tt) so they are exclusive: _h's time inside
        # _find_crossing lands in monitors, not double-counted in the
        # search bucket (cumtime would overlap).
        pr = cProfile.Profile()
        pr.enable()
        stage_pop(ckt, ana, eopt)
        pr.disable()
        s = io.StringIO()
        pst = pstats.Stats(pr, stream=s)
        total = pst.total_tt
        bucket = {}
        for (filename, _, funcname), (cc, nc, tt, ct, _) \
                in pst.stats.items():
            for bname, fsub, funsub in BUCKETS:
                if fsub in filename and (not funsub or funsub in funcname):
                    bucket[bname] = bucket.get(bname, 0.0) + tt
                    break
        res['pop_profile'] = {
            'total_s': total,
            'buckets': {k: round(v, 3) for k, v in
                        sorted(bucket.items(), key=lambda kv: -kv[1])},
            'unattributed_s': round(
                total - sum(bucket.values()), 3),
        }

        # --- cold start: fresh subprocess, import + first POP
        code = ('import sys, time; sys.path.insert(0, ".");'
                't0=time.perf_counter();'
                'from popac.model_loader import load_yaml;'
                'from popac.engine import Engine;'
                'from popac.pop import PopSolver;'
                'ckt, ana = load_yaml("models/%s");'
                'PopSolver(Engine(ckt, {"max_events": 20000000%s}),'
                ' dict(ana.get("pop", {}))).solve();'
                'print(time.perf_counter() - t0)' % (
                    model, ', "chunk": %r' % eopt['chunk']
                    if 'chunk' in eopt else ''))
        t0 = time.perf_counter()
        subprocess.run([sys.executable, '-c', code], capture_output=True,
                       cwd=str(Path(__file__).resolve().parent.parent))
        res['cold_start_s'] = time.perf_counter() - t0
    return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    quick = '--quick' in sys.argv
    reps = 2 if quick else 5
    sel = [m for m in MODELS if not args or m[0] in args]
    OUT.mkdir(parents=True, exist_ok=True)
    man = manifest()
    print('manifest:', json.dumps(man), flush=True)
    all_res = {'manifest': man, 'quick': quick, 'reps': reps, 'models': []}
    for model, tstop, dt, eopt, pacopt in sel:
        print(f'=== {model} (reps={reps})', flush=True)
        r = bench_model(model, tstop, dt, eopt, pacopt, reps, quick)
        all_res['models'].append(r)
        for st, s in r['stages'].items():
            print(f'  {st:10s} cpu median={s["cpu"]["median"]:8.2f}s '
                  f'p90={s["cpu"]["p90"]:8.2f}s min={s["cpu"]["min"]:8.2f}s'
                  f'  (wall {s["wall"]["median"]:8.2f}s)', flush=True)
        if 'pop_peak_mem_mb' in r:
            print(f'  pop peak mem = {r["pop_peak_mem_mb"]:.1f} MB, '
                  f'cold start = {r["cold_start_s"]:.1f}s', flush=True)
            print(f'  pop buckets: ' + ', '.join(
                f'{k}={v:.1f}s' for k, v in
                r['pop_profile']['buckets'].items()), flush=True)
        (OUT / f'bench_{Path(model).stem}.json'
         ).write_text(json.dumps(r, indent=1), encoding='utf-8')
    (OUT / 'bench_summary.json').write_text(
        json.dumps(all_res, indent=1), encoding='utf-8')
    print('written to', OUT, flush=True)


if __name__ == '__main__':
    main()
