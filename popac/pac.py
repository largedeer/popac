# -*- coding: utf-8 -*-
"""Fast periodic small-signal AC (plan Phase 8, backend B).

Along the POP cycle the perturbation obeys, per interval,

    d(delta x)/dt = A_i delta x + b_i * eps e^{j omega t}

with saltation jumps at events.  Event-time sensitivity is exact:
state-triggered events (comparator / diode) carry the analytic row
g = -n^T / hdot; delayed digital events inherit their cause's timing through
augmented "pending timing" slots (one per in-flight delay-queue entry with
state-dependent timing).  All perturbation components advance across the
cycle boundary by e^{j omega T}, giving the uniform quasi-periodic system

    (Psi - e^{j omega T} I) delta z(0) = -G(omega)

with Psi the augmented monodromy (built once, real arithmetic).  Per
frequency only the complex input/output pass is redone: one small
demodulated augmented expm per interval supplies both the state transfer
and the k=0 output integral, so a full sweep costs ~seconds.

T(omega) = -phasor(probe_a)/phasor(probe_b), matching ac.py (backend A).
"""
import math
from typing import Dict, List, Optional

import numpy as np
from scipy.linalg import expm

from .engine import Engine, SimError
from .pop import PopSolver
from .units import eng


def _num(v, default):
    try:
        return eng(v)
    except (ValueError, TypeError):
        return float(default)


class PacSolver:
    def __init__(self, circuit, snap: dict, x_star: np.ndarray,
                 opt: Optional[dict] = None):
        opt = opt or {}
        self.ckt = circuit
        self.snap = snap
        self.x_star = np.asarray(x_star, float).copy()
        self.inject_src = opt.get('inject_src', 'V16')
        self.probe_a = opt.get('probe_a', 'vout')
        self.probe_b = opt.get('probe_b', 'fbt')
        self.fstart = _num(opt.get('fstart', 100.0), 100.0)
        self.fstop = _num(opt.get('fstop', 1e6), 1e6)
        self.per_decade = int(opt.get('per_decade', 25))
        self.svd_rcond = opt.get('svd_rcond', 1e-10)
        # Oscillator-clocked models (P2 VCO) have no periodic pulse source:
        # declared nominal period, same fallback role as pop's T_nom fix.
        self.period_guess = _num(opt.get('period_guess'), 0.0) or None

    # ------------------------------------------------------------------
    def _T_nom(self):
        for d in self.ckt.by_kind('V'):
            if d.wave.period:
                return d.wave.period
        # see __init__: oscillator-clocked models declare the nominal
        # period (the one-cycle walk window is 8 * T_nom -- a 1 us
        # default truncated the VCO walk mid-ramp and broke both the
        # terminal saltation and Psi, bring-up round 4).  Models with a
        # pulse source are unaffected (first branch wins).
        return self.period_guess if self.period_guess else 1e-6

    def _freqs(self):
        nd = math.log10(self.fstop / self.fstart)
        return self.fstart * 10 ** (
            np.arange(0, round(nd * self.per_decade) + 1) / self.per_decade)

    # -------------------------------------------------------------- walk
    def _walk(self):
        e = Engine(self.ckt, {'max_events': 20_000_000})
        e.restore(self.snap)
        e.x = self.x_star.copy()
        records: List[dict] = []
        t0 = e.t
        e.run(e.t + 8 * self._T_nom(), stop_on_trigger=True,
              rec=records.append)
        return records, e, e.t - t0

    # ----------------------------------------------------------- events
    def _build_events(self, records, e):
        """Classify events; return (events, nslots, comp_of, comp_coef)."""
        n = len(self.x_star)
        ch = e.tc.chan_idx[('V', self.inject_src)]
        events = []
        comp_of = {entry[3]: None for entry in self.snap['heap']}
        # initial heap entries get slots 0.. of the pending block
        base_slots = {}
        nslots = n
        for entry in self.snap['heap']:
            base_slots[entry[3]] = nslots
            comp_of[entry[3]] = nslots
            nslots += 1
        comp_coef = {}          # slot -> (t_cause, coef): dt_in = coef e^{jwt}
        slot_seq = {}           # slot -> heap seq that owns it

        nrec = len(records)
        for i, r in enumerate(records):
            kind, tag = r['ev_kind'], r['ev_tag']
            hb = {en[3]: en for en in r['heap_before']}
            ha = {en[3]: en for en in r['heap_after']}
            fired = [s for s in hb if s not in ha]
            pushed = [ha[s] for s in ha if s not in hb]
            ev = {'kind': kind, 'tag': tag, 'i': i}
            if kind in ('CMP', 'D', 'SW', 'DELAY') and i + 1 < nrec:
                w_ev = e._propagate(r['Aug'], r['w0'], r['t1'] - r['t0'])
                f_pre = (r['Aug'] @ w_ev)[:n]
                f_post = (records[i + 1]['Aug'] @ w_ev)[:n]
                ev['df'] = f_post - f_pre
                ev['t_ev'] = r['t1']
            if kind in ('CMP', 'D', 'SW'):
                row = r['row']
                nx = np.asarray(row[:n])
                hdot = float(np.dot(row, r['Aug'] @ w_ev))
                if abs(hdot) < 1e-30:
                    raise SimError('PAC_SINGULAR',
                                   f'grazing event ({tag}): hdot~0')
                ev['g'] = -nx / hdot
                # the direct input channel into the monitor must come from
                # the topology of THIS interval, not the cycle's first one
                ev['urow'] = self._mon_urow(e, r['topo'], tag, ch)
                ev['dt_coef'] = -ev['urow'] / hdot
            elif kind == 'DELAY':
                # simultaneous firings are common by construction (dead-time
                # pairs like BUF(pwm,d) + BUF(INV(pwm),d) transition at the
                # same instants).  Each push owns a distinct slot INDEX, so
                # independence is judged by timing coefficient identity:
                # pushes from one cause share the coef object and inherited
                # slots propagate the same (t_cause, coef) tuple.
                def timing_id(seq):
                    slot = comp_of.get(seq)
                    if slot is None:
                        return None                      # fixed (clock) timing
                    cc = comp_coef.get(slot)
                    return (cc[0], id(cc[1])) if cc else 'free'

                slots = {timing_id(s) for s in fired}
                if len(slots) > 1:
                    raise SimError(
                        'PAC_HEAP',
                        f'{len(fired)} heap entries with independent '
                        f'timing fired together')
                ev['fire_slot'] = comp_of.get(fired[0]) if fired else None
            # attribute pushes to the current event's timing source
            for entry in pushed:
                seq = entry[3]
                if kind in ('CMP', 'D'):
                    comp_of[seq] = nslots
                    comp_coef[nslots] = (r['t1'], ev['dt_coef'])
                    slot_seq[nslots] = seq
                    ev.setdefault('push', []).append(('g', nslots))
                    nslots += 1
                elif kind == 'DELAY' and ev.get('fire_slot') is not None:
                    comp_of[seq] = nslots
                    src = ev['fire_slot']
                    if src in comp_coef:
                        comp_coef[nslots] = comp_coef[src]
                    slot_seq[nslots] = seq
                    ev.setdefault('push', []).append(('slot', nslots, src))
                    nslots += 1
                else:
                    comp_of[seq] = None            # fixed-timing (clock) delay
            # Zero-duration follower (this interval has h == 0): the event
            # fires at the same instant as the previous one and the whole
            # chain physically translates by ONE time shift, so the
            # saltations must telescope: the chain head's timing (slot or
            # g @ pre-chain state) against the SUMMED df = f_after_last -
            # f_before_first.  Evaluating the follower's own g on the
            # post-leader state instead feeds the leader's saltation
            # garbage into its crossing time; stiff dead-topology chains
            # (df ~ 1e12) amplify ~1e-12 dt corruption into an
            # O(true-answer) error and destroy the cancellation.
            if events and (r['t1'] - r['t0']) == 0.0:
                tgt = events[-1]
                if tgt.get('folded') is not None:
                    tgt = events[tgt['folded']]    # extend an existing chain
                if 'df' in ev:
                    tgt['df'] = ev['df'] if 'df' not in tgt \
                        else tgt['df'] + ev['df']
                for item in ev.get('push', ()):
                    if item[0] == 'g':
                        _, slot = item
                        seq = slot_seq[slot]
                        if 'g' in tgt:
                            comp_coef[slot] = (tgt['t_ev'], tgt['dt_coef'])
                            tgt.setdefault('push', []).append(item)
                        elif tgt.get('fire_slot') is not None:
                            src = tgt['fire_slot']
                            if src in comp_coef:
                                comp_coef[slot] = comp_coef[src]
                            tgt.setdefault('push', []).append(
                                ('slot', slot, src))
                        else:
                            comp_of[seq] = None    # clock-slammed chain
                    else:
                        tgt.setdefault('push', []).append(item)
                events.append({'kind': kind, 'tag': tag, 'i': i,
                               'folded': tgt['i']})
                continue
            events.append(ev)
        return events, nslots, comp_of, comp_coef

    # ---------------------------------------------------- terminal trigger
    def _terminal_event(self, records, e, n):
        """Closing saltation of the moving Poincare section (R1a-B).

        P(x) is the state at the run's OWN next trigger crossing, so a
        perturbation that shifts the crossing by dt changes the returned
        state by +f_end*dt (the same orbit evaluated a bit further along).
        Event saltations only cover topology jumps; the final crossing
        itself was missing.  On clock-locked triggers (the crossing rides
        a digital edge, |hdot| ~ V/tau_sw) the term is invisible -- which
        is why PS15 base and boost24 validated without it.  On analog
        mid-interval crossings (dead-time CSW ramp when the inductor
        valley is negative, hdot ~ IL/CSW ~ 1e8..1e10 V/s) it is O(1):
        the L=1.5u repro showed a flat, eps-independent Psi-vs-FD column
        error of ~1.75.  Expressed as a zero-duration event with
        df = -f_end and g = -n/hdot so every pass (Psi product, G(omega)
        input via dt_coef, validation) applies the same machinery:
        z -= df*dt == z + f_end*dt.

        EVENT-PINNED boundaries are exempt (R1 follow-up): when the
        trigger node jumps ALGEBRAICALLY at a coincident switching event
        (no CSW: v(sw) snaps to VIN at the clocked S1-close), the TRIG
        monitor fires the 'already past threshold' path at tau0 -- h(t1)
        sits VOLTS past the fire threshold and the record is degenerate.
        The boundary time is then the pinning event's own time (clocked
        => zero state sensitivity) and the smooth-section saltation would
        use a meaningless slow reconstruction slew: g = -xv/hdot against
        |f_end| ~ 2e8 injected O(100) per-column garbage (original-
        fidelity repro: INVALID 1.36e2 flat across every eps).  A
        genuine crossing lands ON the threshold: |h - fire_th| <= ~1e-6
        (root tolerance + evaluation noise; worst observed 2.3e-6 on the
        simplified base model's 9.5e13 V/s slew) -- seven decades below
        the band-wide gate used here.  Evidence: out/r1_term_probe.log.
        """
        trig = next((d for d in self.ckt.devices
                     if getattr(d, 'kind', '') == 'TRIG'), None)
        if trig is None:
            return None
        r = records[-1]
        w_ev = e._propagate(r['Aug'], r['w0'], r['t1'] - r['t0'])
        f_end = (r['Aug'] @ w_ev)[:n]
        xv, uv = self._rows_for(e, r['topo'], ('v', trig.node))
        # affine source ramps active at the boundary add uv @ s to dh/dt
        us = np.zeros(e.tc.n_u)
        for k, seg in e.seg.items():
            if seg.s and seg.t0 <= r['t1'] <= seg.t1:
                dev = self.ckt.find(k)
                us[e.tc.chan_idx[(dev.kind, k)]] = seg.s
        hdot = float(xv @ f_end + uv @ us)
        # event-pinned boundary guard: see the EVENT-PINNED docstring part
        r_of = e._row_builder(r['topo'], r['layout'])
        h_end = float(r_of(('v', trig.node)) @ w_ev) - trig.vref
        band = 1e-3
        fire_th = band if getattr(trig, 'edge', 'rising') != 'falling' \
            else -band
        if abs(h_end - fire_th) > band:
            return None          # event-pinned boundary: no moving section
        if abs(hdot) < 1e-30:
            raise SimError('PAC_SINGULAR', 'grazing trigger crossing')
        chI = e.tc.chan_idx[('V', self.inject_src)]
        u_in = float(uv[chI])
        ev = {'kind': 'TRIG_END', 'tag': f'TRIG:{trig.node}',
              'i': len(records), 'df': -f_end, 'g': -xv / hdot,
              'dt_coef': -u_in / hdot, 't_ev': r['t1']}
        rec = {'t0': r['t1'], 't1': r['t1'], 'Aug': r['Aug'], 'w0': w_ev,
               'topo': r['topo'], 'ev_kind': 'TRIG_END', 'ev_tag': 'TRIG',
               'heap_before': r['heap_after'],
               'heap_after': r['heap_after'], 'row': None}
        return ev, rec

    # -------------------------------------------------------- forward pass
    def _forward(self, records, events, e, z0, Phi, A, b, C, n, omega=None,
                 Ecache=None, want_out=False):
        """Propagate augmented slots through the cycle.

        omega=None : real homogeneous pass (build Psi columns).
        omega set  : complex pass, input always on; demodulated expm E per
                     interval from Ecache (built by caller if given).
        Returns (zend, y) with y the accumulated k=0 output integrals.
        """
        nz = len(z0)
        z = z0.astype(complex if omega is not None else float).copy()
        y = np.zeros(2, dtype=complex)
        for i, ev in enumerate(events):
            r = records[i]
            h = r['t1'] - r['t0']
            dt = None
            if omega is None:
                z[:n] = Phi[i] @ z[:n]
            else:
                E = Ecache[i]
                # augmented vector [x~; 1; q0; q1]: the demodulated input is
                # the constant 1, both output integrals restart at 0
                v = np.zeros(n + 3, dtype=complex)
                v[:n] = np.exp(-1j * omega * r['t0']) * z[:n]
                v[n] = 1.0
                out = E @ v
                z[:n] = np.exp(1j * omega * r['t1']) * out[:n]
                if want_out:
                    y += out[n + 1:n + 3]
            if 'df' in ev:
                if 'g' in ev:
                    # event-time perturbation from the PRE-jump state (the
                    # crossing happens before any saltation of this event)
                    dt = ev['g'] @ z[:n]
                    if omega is not None and ev.get('dt_coef', 0.0):
                        dt += ev['dt_coef'] * np.exp(1j * omega * ev['t_ev'])
                    # saltation: during [t*, t*+dt] the perturbed run is
                    # still on the pre-event topology while the unperturbed
                    # has switched -> z+ = z- + (f- - f+) dt = z- - df*dt.
                    # Zero-duration event pairs (dead topology -> diode on)
                    # telescope away under this same sign.
                    z[:n] -= ev['df'] * dt
                elif ev.get('fire_slot') is not None:
                    dt = z[ev['fire_slot']]
                    # delayed switch: on [t*, t*+dt] the state still evolves
                    # under the OLD topology -> perturbed minus unperturbed
                    # = (f- - f+) dt = -df*dt
                    z[:n] -= ev['df'] * dt
            for item in ev.get('push', ()):
                if item[0] == 'g':
                    # the pushed delay inherits the crossing-time
                    # perturbation of its cause event: pre-jump dt (which
                    # already carries the direct input term above), NOT
                    # g @ z+ contaminated by this event's own saltation
                    _, slot = item
                    z[slot] = 0.0 if dt is None else dt
                else:
                    _, slot, src = item
                    z[slot] = z[src]
        return z, y

    # ---------------------------------------------------------- validate
    def _validate(self, snap, Psi):
        """Analytic Psi vs central-difference columns of the true trigger
        map (assessment P0: a variational construction inconsistent with
        the engine map must not publish PM/GM/Floquet).  eps must keep the
        shifted event times resolvable: delta_t = eps/slew has to clear
        the crossing search's min_step (~span*2^-24); the state floor is
        1.0 (not 1e-3) because nano-scale cycle-reset states (shorted ramp
        caps) would give eps ~ 1e-8, below that resolution.  Columns whose
        true sensitivity vanishes (sampled-echo caps reset every cycle)
        have |DP| ~ FD noise, so the relative metric needs the same style
        of absolute floor POP's residual calibration uses: 1e-3."""
        n = len(self.x_star)
        T = self._T_nom()
        sc = np.maximum(np.abs(self.x_star), 1.0)
        # Per-column min over eps in {1e-5, 3e-5, 1e-4}: a WRONG
        # construction is flat-large across every eps (pre-fix evidence
        # ~1.75 over five decades), so the gate still fires; a correct one
        # has an orbit-dependent noise window (original 1e-5 / L=1.5u
        # 3e-5 / L=1.0u 1e-4) and the min lands inside it.  Cost x3, a few
        # seconds per column on PS15.  Evidence: out/r1_lscan_pac2.log /
        # r1_lscan_pac3.log / r1_orig_pac.log.
        # The ladder stops at 1e-4 ON PURPOSE: at L=0.68u the 1e-3 leg
        # explodes (|DP_y3| 2.1e-4 -> 1.9e3: a 2.5 mV nudge qualitatively
        # changes the deep-valley orbit) while the wrong-construction
        # control stays flat 1.75 through 1e-3 -- and that orbit's worst
        # residual shrinks 1/eps (pure FD noise, sub-floor column
        # |DP|~2e-4 vs floor 1e-3), so it reports INVALID = DECLARED
        # floor-limited, not a construction error.  Evidence:
        # out/r1_l068_pac.log / out/r1_l15_noterm.log.
        rels = [float('inf')] * n
        eps_win = [0.0] * n
        for eps in (1e-5, 3e-5, 1e-4):
            cols = []
            for j in range(n):
                h = eps * sc[j]
                c = np.zeros(n)
                for sgn in (+1.0, -1.0):
                    e2 = Engine(self.ckt, {'max_events': 2_000_000})
                    e2.restore(snap)
                    e2.x = self.x_star.copy()
                    e2.x[j] += sgn * h
                    e2.run(snap['t'] + 30 * T, stop_on_trigger=True)
                    c += sgn * e2.x
                cols.append(c / (2 * h))
            DP = np.column_stack(cols)
            for j in range(n):
                denom = max(np.linalg.norm(DP[:, j]), 1e-3)
                r = float(np.linalg.norm(Psi[:n, j] - DP[:, j]) / denom)
                if r < rels[j]:
                    rels[j] = r
                    eps_win[j] = eps
        tol = 5e-3
        return {'status': 'VALIDATED' if max(rels) < tol else 'INVALID',
                'max_col_rel': max(rels), 'tol': tol,
                'per_col_rel': rels, 'per_col_eps': eps_win}

    # -------------------------------------------------------------- solve
    def solve(self) -> Dict:
        records, e, T = self._walk()
        n = len(self.x_star)
        chI = e.tc.chan_idx[('V', self.inject_src)]
        nrec = len(records)
        A = np.zeros((nrec, n, n))
        b = np.zeros((nrec, n))
        Phi = np.zeros((nrec, n, n))
        for i, r in enumerate(records):
            h = r['t1'] - r['t0']
            A[i] = r['Aug'][:n, :n]
            b[i] = r['topo'].d_mat[:, chI]
            Phi[i] = expm(A[i] * h)
        events, nslots, comp_of, comp_coef = self._build_events(records, e)
        self._comp_coef = comp_coef

        # output rows are topology dependent (algebraic reconstruction of
        # probed nodes changes with switch state): per-interval C / urow
        Cs = np.zeros((nrec, 2, n))
        urows = np.zeros((nrec, 2))
        for i, r in enumerate(records):
            for k, p in enumerate((self.probe_a, self.probe_b)):
                spec = self.ckt.probes.get(p, ('v', p))
                xv, uv = self._rows_for(e, r['topo'], spec)
                Cs[i, k] = xv
                urows[i, k] = uv[chI]
        C = Cs[0]
        urow = urows[0]

        # terminal trigger saltation (see _terminal_event): appended as a
        # zero-duration record+event AFTER _build_events so the folding
        # logic never merges it into a real zero-duration chain.
        term = self._terminal_event(records, e, n)
        term_on = term is not None
        if term is not None:
            ev_t, rec_t = term
            records.append(rec_t)
            events.append(ev_t)
            A = np.concatenate([A, A[-1:]], axis=0)       # h == 0 -> Phi = I
            b = np.concatenate([b, b[-1:]], axis=0)
            Phi = np.concatenate([Phi, np.eye(n)[None]], axis=0)
            Cs = np.concatenate([Cs, np.zeros((1, 2, n))], axis=0)
            urows = np.concatenate([urows, np.zeros((1, 2))], axis=0)
            nrec += 1

        # homogeneous monodromy over all slots (columns = basis runs are
        # equivalent to the matrix product; the product form is cheaper)
        Psi = np.eye(nslots)
        for i, ev in enumerate(events):
            M = np.eye(nslots)
            M[:n, :n] = Phi[i]
            Psi = M @ Psi
            if 'df' in ev:
                if 'g' in ev:
                    # slot capture FIRST: the pushed delay inherits the
                    # pre-saltation crossing perturbation g @ z-
                    for item in ev.get('push', ()):
                        if item[0] == 'g':
                            R = np.eye(nslots)
                            R[item[1], :n] = ev['g']
                            Psi = R @ Psi
                    # saltation with the (f- - f+) sign, same as _forward
                    S = np.eye(nslots)
                    S[:n, :n] -= np.outer(ev['df'], ev['g'])
                    Psi = S @ Psi
                elif ev.get('fire_slot') is not None:
                    S = np.eye(nslots)
                    S[:n, ev['fire_slot']] -= ev['df']   # -(f+ - f-) dt
                    Psi = S @ Psi
                    for item in ev.get('push', ()):
                        R = np.eye(nslots)
                        R[item[1], item[2]] = 1.0
                        Psi = R @ Psi

        # boundary condition slots: x + heap present at start/end.  Entries
        # pair by CHANNEL (seqs differ across the cycle).  Snapshot entries
        # get free base slots, but consistency with the end side decides:
        # a channel whose end entry is clock-pushed (fixed timing, slot
        # None) has identically-zero perturbation, so its start unknown is
        # dropped too (z=0), keeping both sides symmetric.
        h0 = sorted(en[1] for en in self.snap['heap'])
        hT = sorted(en[1] for en in records[-1]['heap_after'])
        if h0 != hT:
            raise SimError('PAC_HEAP', f'heap mismatch {h0} vs {hT}')
        start_by_chan = {en[1]: comp_of[en[3]] for en in self.snap['heap']}
        end_by_chan = {en[1]: comp_of[en[3]]
                       for en in records[-1]['heap_after']}
        pairs = [(start_by_chan[ch], end_by_chan[ch]) for ch in h0]
        cols = list(range(n)) + [s for s, e_ in pairs if e_ is not None]
        rows = list(range(n)) + [e_ for s, e_ in pairs if e_ is not None]

        # validity gate BEFORE the sweep: a few seconds of engine cycles
        # against the true trigger map; an inconsistent construction must
        # not publish PM/GM/Floquet (assessment P0)
        validation = self._validate(self.snap, Psi)
        self.validation = validation

        freqs = self._freqs()
        Tabs = np.zeros(len(freqs), dtype=complex)
        for fi, f in enumerate(freqs):
            w = 2 * math.pi * f
            # demodulated augmented expm per interval:
            #   x~' = (A - jw I) x~ + b,  q' = C x~     (input demodulates
            # to a constant; q accumulates the k=0 output integral)
            Ecache = []
            for i in range(nrec):
                L = np.zeros((n + 3, n + 3), dtype=complex)
                L[:n, :n] = A[i] - 1j * w * np.eye(n)
                L[:n, n] = b[i]
                L[n + 1:n + 3, :n] = Cs[i]
                # direct feedthrough of the demodulated input (v[n] = 1)
                # into the k=0 output integrals, per interval topology
                L[n + 1:n + 3, n] = urows[i]
                Ecache.append(expm(L * (records[i]['t1'] - records[i]['t0'])))
            zend, _ = self._forward(records, events, e, np.zeros(nslots),
                                    Phi, A, b, C, n, omega=w, Ecache=Ecache)
            G = zend
            lam = np.exp(1j * w * T)
            Msys = Psi[np.ix_(rows, cols)] - lam * np.eye(len(cols))
            dz0c, *_ = np.linalg.lstsq(Msys, -G[cols], rcond=self.svd_rcond)
            z0 = np.zeros(nslots, dtype=complex)
            z0[cols] = dz0c
            _, y = self._forward(records, events, e, z0, Phi, A, b, C, n,
                                 omega=w, Ecache=Ecache, want_out=True)
            y = y / T          # direct term already inside the integrals
            Tabs[fi] = -y[0] / y[1] if abs(y[1]) > 0 else complex(np.nan)
        # report the analytic Floquet spectrum under the same structural
        # isolation filter POP uses: frozen-island |lambda| ~ 1 modes are
        # genuine invariants of galvanically separate storage but say
        # nothing about main-circuit stability
        ev_p, vec_p = np.linalg.eig(Psi[:n, :n])
        iso_p = PopSolver(e)._isolated_modes(self.snap, vec_p)
        if iso_p is not None and iso_p.any():
            flo = np.sort(np.abs(ev_p)[~iso_p])[::-1]
            flo_frozen = np.sort(np.abs(ev_p)[iso_p])[::-1]
        else:
            flo = np.sort(np.abs(ev_p))[::-1]
            flo_frozen = np.array([])
        info = {'period': T, 'n_intervals': nrec, 'n_slots': nslots,
                'heap_chans': h0,
                'validation': validation,
                'trigger_saltation': term_on,
                'floquet_abs': flo,
                'floquet_frozen_abs': flo_frozen}
        return {'freqs': freqs, 'T': Tabs, 'info': info}

    # -------------------------------------------------------- row helpers
    def _rows_for(self, e, topo, spec):
        n = len(self.x_star)
        if spec[0] == 'v':
            pairs, sgn = [topo.colmap[e.tc._n(spec[1])]], [1.0]
        elif spec[0] == 'dv':
            pairs, sgn = [], []
            if spec[1] != '0':
                pairs.append(topo.colmap[e.tc._n(spec[1])]); sgn.append(1.0)
            if spec[2] != '0':
                pairs.append(topo.colmap[e.tc._n(spec[2])]); sgn.append(-1.0)
        else:
            raise SimError('PAC_CONFIG', f'probe spec {spec} unsupported')
        xv = np.zeros(n)
        uv = np.zeros(e.tc.n_u)
        for (a, u_), s in zip(pairs, sgn):
            xv += s * a
            uv += s * u_
        return xv, uv

    def _mon_urow(self, e, topo, tag, chI):
        if tag.startswith('CMP:'):
            d = self.ckt.find(tag[4:])
            return self._dv_urow(e, topo, d.inp, d.inn, chI)
        if tag.startswith('D:'):
            d = self.ckt.find(tag[2:])
            return self._dv_urow(e, topo, d.a, d.k, chI)
        if tag.startswith('SW:'):
            d = self.ckt.find(tag[3:])
            return self._dv_urow(e, topo, d.c1, d.c2, chI)
        return 0.0

    def _dv_urow(self, e, topo, na, nb, chI):
        uv = np.zeros(e.tc.n_u)
        if na != '0':
            uv += topo.colmap[e.tc._n(na)][1]
        if nb != '0':
            uv -= topo.colmap[e.tc._n(nb)][1]
        return float(uv[chI])
