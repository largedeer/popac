# -*- coding: utf-8 -*-
"""CLI runner.

    python -m popac.run models/buck5v.yaml --analysis all [--out out/buck5v]

Runs transient / POP / AC (AC starts from the POP operating point), writes
results.json, report.md, transient.png, bode.png into --out.
"""
import argparse
import os
import sys
import time

import numpy as np

from .model_loader import load_yaml
from .engine import Engine
from .pop import PopSolver
from .ac import AcSolver, loop_metrics
from .pac import PacSolver
from . import report


def _avg_model_ref():
    """Optional overlay from the Loop_Analyzer average model (plan route 6)."""
    try:
        sys.path.insert(0, os.path.abspath('../Loop_Analyzer'))
        from buck_loop_analyzer import (BuckLoopAnalyzer, PowerStageParams,
                                        CurrentSenseParams, ErrorAmpParams,
                                        FeedbackParams, CompensatorParams)
        p = os.path.join(os.path.abspath('../Loop_Analyzer'),
                         'buck_loop_params.json')
        import json
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
        az = BuckLoopAnalyzer(PowerStageParams(**d['power']),
                              CurrentSenseParams(**d['sense']),
                              ErrorAmpParams(**d['ea']),
                              FeedbackParams(**d['fb']),
                              CompensatorParams(**d['comp']))
        f = np.logspace(1, 6.3, 400)
        return f, az.loop_gain_tf(f)
    except Exception:
        return None


def _event_search_md(md, e):
    """Surface the engine's unresolved event-search report (review P0):
    sub-bandwidth spans are within the declared dt/16 resolution and only
    count; dropped brackets are a regression tripwire (would have raised
    under the default policy)."""
    rep = e.event_report()
    if rep['unresolved']:
        w = (f", narrowest {rep['min_span']:.2e} s"
             if rep['min_span'] is not None else '')
        md.append(f'- event search: {rep["subband"]} sub-bandwidth unresolved '
                  f'spans (below dt/16 resolution{w}), '
                  f'{rep["budget"]} dropped brackets\n')
    for wmsg in e.warnings:
        md.append(f'- warning: {wmsg}\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--analysis', default='all',
                    choices=['transient', 'pop', 'ac', 'pac', 'all'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--cycles', type=int, default=300,
                    help='transient cycles for --analysis transient|all')
    ap.add_argument('--fstart', type=float, default=None)
    ap.add_argument('--fstop', type=float, default=None)
    args = ap.parse_args(argv)

    ckt, ana = load_yaml(args.model)
    name = ckt.meta.get('model', os.path.splitext(
        os.path.basename(args.model))[0]) if hasattr(ckt, 'meta') else 'model'
    outdir = args.out or os.path.join('out', name)
    os.makedirs(outdir, exist_ok=True)
    figures, data, md = {}, {}, []

    T_sw = 2.5e-6
    for d in ckt.by_kind('V'):
        if d.wave.period:
            T_sw = d.wave.period
            break

    do = args.analysis
    solver_opt = dict(ana.get('solver', {}))    # YAML solver: min_ron etc.
    if do in ('transient', 'all'):
        t0 = time.time()
        e = Engine(ckt, {'max_events': 50_000_000, **solver_opt})
        n = args.cycles
        # probes come from the model declaration (generality: no PS15
        # node names assumed); PS15-style grouping only when present
        plist = list(ckt.probes) or ['VOUT', 'IL', 'FB', 'EA', 'SW']
        s, _ = e.run(n * T_sw, sample_dt=T_sw / 50, probes=plist)
        tail = [v for t, v in s if t > 0.9 * n * T_sw]
        stats = {p: float(np.mean([v[p] for v in tail])) for p in plist}
        print('transient tail means:', {k: f'{v:.5f}' for k, v in stats.items()})
        fig = os.path.join(outdir, 'transient.png')
        win = [sv for sv in s if sv[0] > (n - 6) * T_sw]   # last 6 cycles
        if all(p in plist for p in ('VOUT', 'IL', 'FB', 'EA', 'SW',
                                    'GH', 'GL')):
            groups = {'V': ['VOUT', 'FB', 'EA'], 'I(L2) [A]': ['IL'],
                      'digital': ['SW', 'GH', 'GL']}
            win_p = ['VOUT', 'IL', 'FB', 'EA', 'SW', 'GH', 'GL']
        else:
            groups = None
            win_p = plist
        report.plot_transient(win, win_p, fig, groups=groups)
        figures['transient'] = fig
        data['transient'] = {'cycles': n, 'tail_means': stats,
                             'elapsed_s': round(time.time() - t0, 1),
                             'events': e.event_report()}
        md.append('## Transient\n\n- steady-state (last 10%): '
                  + ', '.join(f'{k}={v:.5f}' for k, v in stats.items())
                  + f'\n- window shown: last 6 cycles ({6 * T_sw * 1e6:.1f} us)\n')
        _event_search_md(md, e)

    pop = None
    psolver = None
    if do in ('pop', 'ac', 'pac', 'all'):
        t0 = time.time()
        e = Engine(ckt, {'max_events': 50_000_000, **solver_opt})
        psolver = PopSolver(e, dict(ana.get('pop', {})))
        pop = psolver.solve()
        ml = ('n/a' if pop.max_multiplier is None
              else f'{pop.max_multiplier:.4f}')
        print(f'POP ok={pop.ok} res={pop.residual:.3e} T={pop.period:.6e}s '
              f'max|lam|={ml}')
        data['pop'] = {'ok': pop.ok, 'residual': pop.residual,
                       'period_s': pop.period,
                       'discrete_match': pop.discrete_match,
                       'iterations': pop.iterations,
                       'floquet_abs': np.sort(np.abs(pop.floquet))[::-1]
                       if pop.floquet is not None else None,
                       'max_multiplier_physical': pop.max_multiplier,
                       'events': e.event_report()}
        md.append('## POP\n\n'
                  f'- converged: {pop.ok} (residual {pop.residual:.2e} scaled, '
                  f'{pop.iterations} iter)\n'
                  f'- period: {pop.period * 1e6:.4f} us '
                  f'({pop.period / T_sw:.3f} x T_sw)\n'
                  f'- discrete states repeat: {pop.discrete_match}\n'
                  f'- Floquet max |lambda| (physical modes): '
                  f'{ml}\n'
                  + ('- full |lambda| spectrum: '
                     + ', '.join(f'{v:.4g}' for v in
                                 np.sort(np.abs(pop.floquet))[::-1]) + '\n'
                     if pop.floquet is not None else ''))
        _event_search_md(md, e)

    if do in ('ac', 'all'):
        assert pop is not None and pop.ok, 'AC requires a converged POP'
        t0 = time.time()
        # start AC exactly on the fixed point: the POP boundary snapshot
        # (discrete states / heap / segments) with the converged x*
        e.restore(getattr(psolver, 'snap0', None) or e.snapshot())
        e.x = pop.x.copy()
        snap = e.snapshot()
        acopt = dict(ana.get('ac', {}))
        if args.fstart:
            acopt['fstart'] = args.fstart
            acopt['fstop'] = args.fstop
        raw_path = os.path.join(outdir, 'ac_raw.json')

        def _flush(i, f, st):
            """Per-point progress + incremental ac_raw.json persistence.
            (ac is bound at call time — solve_all runs after construction.)"""
            import json
            print(f'  AC {i + 1}: {f:g} Hz {st}', flush=True)
            fr, T, ok, stl = ac._part
            with open(raw_path, 'w', encoding='utf-8') as fh:
                json.dump(report._jsonable(
                    {'freqs': fr, 'T_re': np.real(T), 'T_im': np.imag(T),
                     'ok': ok, 'status': stl, 'partial': True}), fh)

        acopt['on_point'] = _flush
        ac = AcSolver(ckt, acopt, pop_snapshot=snap, pop_x=pop.x)
        amp_sweep = ac.inject_amp        # solve_all's linearity check mutates it
        res = ac.solve_all()
        dt = time.time() - t0
        n_pass = sum(1 for s in res.status if s == 'PASS')
        print(f'AC: {sum(res.ok)}/{len(res.freqs)} ran, {n_pass} PASS '
              f'in {dt:.0f}s')
        # persist the raw sweep immediately: it is the expensive artifact
        with open(os.path.join(outdir, 'ac_raw.json'), 'w',
                  encoding='utf-8') as f:
            import json
            json.dump(report._jsonable(
                {'freqs': res.freqs,
                 'T_re': np.real(res.T), 'T_im': np.imag(res.T),
                 'ok': res.ok, 'status': res.status,
                 'metrics': res.metrics,
                 'linearity': res.amp_linearity}), f, indent=2)
        fig = os.path.join(outdir, 'bode.png')
        report.plot_bode(res.freqs, res.T, fig, ref=_avg_model_ref(),
                         status=res.status)
        figures['bode'] = fig
        data['ac'] = {'freqs': res.freqs, 'T': res.T, 'ok': res.ok,
                      'status': res.status,
                      'metrics': res.metrics,
                      'linearity': res.amp_linearity, 'elapsed_s': dt}
        m = res.metrics
        gm_txt = (f"{m['gain_margin_db']:.1f} dB at "
                  f"{m['gm_at_freq'] / 1e3:.1f} kHz"
                  if m.get('gm_at_freq') else 'n/a')
        md.append('## AC loop gain (Middlebrook injection fit)\n\n'
                  f'- points: {sum(res.ok)}/{len(res.freqs)} ran, '
                  f'**{n_pass} PASS** (PM/GM from PASS points only) '
                  f'({ac.fstart:g}..{ac.fstop:g} Hz, '
                  f'{ac.per_decade}/decade, amp {amp_sweep:g} V)\n'
                  '- 0 dB crossings: '
                  + (', '.join(f"{c['freq'] / 1e3:.2f} kHz (PM {c['pm_deg']:.1f} deg)"
                               for c in m['crossovers']) or 'none')
                  + f'\n- gain margin: {gm_txt}\n')
        if m.get('margin_unresolved'):
            md.append('- **MARGIN_UNRESOLVED** (crossing hidden inside an '
                      'invalid gap): '
                      + '; '.join(f"{u['kind']} in {u['f_lo']:g}..{u['f_hi']:g} Hz"
                                  for u in m['margin_unresolved']) + '\n')
        if m.get('segments') and len(m['segments']) > 1:
            md.append(f"- PASS points form {len(m['segments'])} contiguous "
                      f"segments (margins per segment, no cross-gap "
                      f"interpolation)\n")
        for name, lin in (res.amp_linearity or {}).items():
            if lin:
                md.append(
                    f"- linearity[{name}] @ {lin['freq'] / 1e3:.1f} kHz: "
                    f"|T| ratio 5m/{amp_sweep * 1e3:.0f}m = "
                    f"{lin['mag_ratio']:.4f}, "
                    f"phase diff {lin['phase_diff_deg']:.3f} deg\n")

    if do in ('pac', 'all'):
        assert pop is not None and pop.ok, 'PAC requires a converged POP'
        t0 = time.time()
        # fresh boundary snapshot from the POP solution (AC above may have
        # mutated the engine/sources)
        e.restore(getattr(psolver, 'snap0', None) or e.snapshot())
        e.x = pop.x.copy()
        popt = dict(ana.get('pac', {}))
        # fall back to the AC sweep definition for frequencies/probes
        acdef = dict(ana.get('ac', {}))
        for k in ('inject_src', 'probe_a', 'probe_b'):
            popt.setdefault(k, acdef.get(k))
        popt.setdefault('fstart', acdef.get('fstart', 100))
        popt.setdefault('fstop', acdef.get('fstop', 1e6))
        popt.setdefault('per_decade', acdef.get('per_decade', 25))
        if args.fstart:
            popt['fstart'] = args.fstart
            popt['fstop'] = args.fstop
        pac = PacSolver(ckt, e.snapshot(), pop.x, popt)
        pres = pac.solve()
        dt = time.time() - t0
        val = pres['info'].get('validation', {})
        pac_ok = val.get('status', 'VALIDATED') == 'VALIDATED'
        if pac_ok:
            m = loop_metrics(pres['freqs'], pres['T'])
        else:
            m = {'crossovers': [], 'gain_margin_db': None,
                 'invalid_reason': (
                     f"Psi vs FD map mismatch (max col rel "
                     f"{val.get('max_col_rel', float('nan')):.2e} >= "
                     f"{val.get('tol', float('nan')):.0e})")}
            print(f"PAC INVALID: {m['invalid_reason']} - no PM/GM computed")
        print(f'PAC: {len(pres['freqs'])} points in {dt:.1f}s, '
              + (', '.join(f"fc {c['freq'] / 1e3:.2f} kHz PM {c['pm_deg']:.1f} deg"
                           for c in m['crossovers']) or 'no crossover'))
        import json
        with open(os.path.join(outdir, 'pac_raw.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(report._jsonable(
                {'freqs': pres['freqs'],
                 'T_re': np.real(pres['T']), 'T_im': np.imag(pres['T']),
                 'metrics': m,
                 'validation': {k: val.get(k) for k in
                               ('status', 'max_col_rel', 'tol')},
                 'floquet_abs': pres['info']['floquet_abs'],
                 'floquet_frozen_abs': pres['info']['floquet_frozen_abs'],
                 'elapsed_s': round(dt, 2)}), fh, indent=2)
        figp = os.path.join(outdir, 'bode_pac.png')
        report.plot_bode(pres['freqs'], pres['T'], figp, ref=_avg_model_ref())
        figures['bode_pac'] = figp
        data['pac'] = {'freqs': pres['freqs'], 'T': pres['T'],
                       'metrics': m, 'info': pres['info'],
                       'elapsed_s': dt}
        gm_txt = (f"{m['gain_margin_db']:.1f} dB at "
                  f"{m['gm_at_freq'] / 1e3:.1f} kHz"
                  if m.get('gm_at_freq') else 'n/a')
        md.append('## PAC loop gain (variational, plan Phase 8)\n\n'
                  f'- points: {len(pres['freqs'])} in {dt:.1f} s '
                  '(walk once + demodulated expm per interval per point)\n'
                  f"- validity: **{val.get('status', 'VALIDATED')}** "
                  f"(Psi-vs-FD max column rel "
                  f"{val.get('max_col_rel', float('nan')):.2e}, tol "
                  f"{val.get('tol', float('nan')):.0e})\n"
                  + ('' if pac_ok else
                     '**PAC INVALID: margins suppressed** '
                     f"({m['invalid_reason']})\n")
                  + '- analytic Floquet |lambda| (main circuit): '
                  + ', '.join(f'{v:.4g}' for v in
                              pres['info']['floquet_abs'][:8])
                  + (f" (+{len(pres['info']['floquet_frozen_abs'])} "
                     "frozen-island ~1)" if len(
                         pres['info']['floquet_frozen_abs']) else '')
                  + '\n'
                  '- 0 dB crossings: '
                  + (', '.join(f"{c['freq'] / 1e3:.2f} kHz "
                               f"(PM {c['pm_deg']:.1f} deg)"
                               for c in m['crossovers']) or 'none')
                  + f'\n- gain margin: {gm_txt}\n')

    fid_declared = getattr(ckt, 'meta', {}).get('fidelity', 'UNDECLARED') \
        if hasattr(ckt, 'meta') else 'UNDECLARED'
    # solver-side resistance clamps downgrade the tier (review §6): the
    # declared devices did not run as declared
    from .mna import effective_fidelity
    try:
        tc_clamps = e.tc.clamps
    except NameError:
        tc_clamps = []
    fid = effective_fidelity(fid_declared, tc_clamps)
    md.append(f'## Model fidelity tier: {fid}\n\n')
    if fid != fid_declared:
        md.append('- declared: ' + fid_declared
                  + ', downgraded: the solver clamped '
                    f'{len(tc_clamps)} device value(s) -- '
                  + '; '.join(f"{c['device']}.{c['param']} "
                              f"{c['declared']:g}->{c['effective']:g} "
                              f"({c['reason']})" for c in tc_clamps) + '\n')
    md.append('- ORIGINAL_EQUIVALENT: only provably equivalent conversions\n'
              '- CONDITIONED: numerical-conditioning parasitics added '
              '(declared below)\n'
              '- SIMPLIFIED: control network or devices replaced '
              '(declared below)\n')

    md.append('## Model notes (declared simplifications)\n')
    if fid == 'SIMPLIFIED':
        md.append(
            '\n- slope compensation: original sampled-echo network '
            '(S2/S3/C3/C5/E1/E2/E5) replaced by textbook EA - Se*t '
            '(Se = 200 kV/s from the original ramp generator)\n'
            '- ideal-diode ron = 1 mOhm (assumed; PARAM_VALUES truncated)\n'
            '- U4 dead-time delay 20n (the .sxsch value is 5n)\n'
            '- nodes 56/59 merged into SW (H1 sense is a zero-ohm path)\n'
            '- floating load-step island retained (isolated); 1 Meg bleeder '
            'RB1 added for numerical conditioning of the open S6 case\n'
            '- CSW = 100 pF switch-node snubber added (real switch-node '
            'capacitance; keeps the dead-time algebraic reconstruction '
            'well-conditioned)\n')
    elif fid == 'CONDITIONED':
        md.append(
            '\n- sampled-echo slope compensation transcribed verbatim '
            '(v_cmd = EA + v(C3) - 2*v(C16))\n'
            '- diode USER_DIODE params from .sxsch PARAM_VALUES; '
            'U4 delay 5n, gate params from in-file PARAMETERS\n'
            '- nodes 56/59 merged into SW (H1 sense is a zero-ohm path)\n'
            '- conditioning parasitics: CSW = 100 pF on the switch node, '
            'RB1 = 1 Meg bleeder on isl4 (declared; sensitivity scan shows '
            'max|lambda| varies <0.1% for CSW 10p..1n / absent)\n')
    elif fid == 'ORIGINAL_EQUIVALENT':
        md.append(
            '\n- full .sxsch transcription (64 devices verified '
            'item-by-item; see debug/sxsch_dump.py output)\n'
            '- provably inert omissions: dangling V5, probes/terminals\n'
            '- nodes 56/59 merged into SW (H1 sense is a zero-ohm path)\n')
    else:
        md.append('\n- (no notes declared)\n')

    report.write_outputs(outdir, data, figures, md)
    print('report written to', outdir)


if __name__ == '__main__':
    main()
