# popac — open switching-converter loop analysis

**Periodic Operating Point + AC**: an open-source, Python-native simulator for
DC-DC switching power converters that finds the **periodic steady state
directly** (POP) and computes the **small-signal loop gain analytically** (PAC)
— built on fully exposed, independently verifiable variational mathematics
(monodromy / saltation matrices).

If you design control loops for switching converters and you have ever waited
an hour for a transient-based Bode sweep — or wished you could see *why* a
simulator's crossover number should be trusted — popac is for you.

## Why

Loop design for a PWM converter needs the small-signal response around the
switching periodic steady state. The usual free path is a transient injection
sweep (LTspice / ngspice style): settle the converter, inject, FFT, repeat per
frequency. That costs minutes-to-hours per design point. Closed-source
commercial simulators solve this class of problem, but they are licensed
black boxes — you cannot audit how the crossover number was produced.

popac takes the analytical route: it builds the period map's variational
equation **event by event** (exact linear propagation between switching
events, saltation matrices at them), so the loop gain and the Floquet
stability spectrum come out of one cycle of linear algebra — no perturbation
simulation at all.

| same converter, 101-frequency Bode | wall time |
|---|---|
| transient injection sweep (`--analysis ac`) | ~55 min |
| **PAC analytic construction (`--analysis pac`)** | **19.5 s (~2800×)** |

Both agree at the crossover to 0.1–0.3 dB / 0.3° (asserted in the test suite),
and PAC additionally reports the **Floquet multipliers** of the orbit — the
sampled-data stability eigenvalues a transient Bode plot cannot give you.

## Quickstart

```bash
pip install numpy scipy pyyaml matplotlib      # (or: pip install -e .)
python -m popac.run models/buck5v.yaml --analysis pac
```

Output lands in `out/buck5v/`: `results.json`, `report.md`, waveform and Bode
plots. For the reference buck example you should see something like:

```
POP ok=True res=3.0e-06 T=5.000000e-06s max|lam|=0.9998
PAC: 38 points in 11 s, fc 5.30 kHz PM 47.7 deg
```

- `POP ... res` — the periodic steady state was found by Newton iteration on
  the one-period map (residual = how exactly the orbit closes)
- `max|lam|` — largest Floquet multiplier of the orbit (< 1 → stable)
- `PAC VALIDATED` — the analytic loop gain passed a built-in
  analytic-vs-finite-difference gate before being reported

Analyses: `--analysis transient|pop|ac|pac|all` (see `python -m popac.run -h`).

## Example gallery

All examples are self-contained YAML files with their design math in the
header comment; each is backed by closed-form physics assertions in the test
suite.

| model | topology | what it shows |
|---|---|---|
| `buck5v` | async buck 12→5 V/2 A @200 kHz, peak CMC | the reference example; DC laws, ripple bounds, dIL ∝ 1/L |
| `boost24` | boost 12→24.5 V/2 A, peak CMC + slope comp | RHP zero, D>0.5 subharmonic physics |
| `buck2ph` | two-phase interleaved buck, 180° | ripple cancellation ×10 (measured 9.95×) |
| `sepic24` | SEPIC, RC-damped flying cap | the undamped exchange resonance + why it needs damping |
| `flyback24` | DCM flyback, coupled windings (k=0.98) | RCD clamp energetics, leakage trend vs k |
| `buckboost24` | inverting buck-boost 12→−12 V | inverting feedback polarity; diode-fed ripple bounds |
| `cuk24` | Ćuk with coupled inductors (M = L₂) | **input-ripple cancellation ×21 vs closed-form ×21** |
| `zeta24` | Zeta 12→5 V | the four-constraint topology derivation |
| `pacmini` | 3-state toy | minimal PAC/PWM regression circuit |
| `vco_block` | triangle VCO from basic devices | variable-frequency control (LLC clock) |

## How it works

- **Piecewise-linear engine.** Between switching events the circuit is a
  linear (MNA-stamped, SVD pencil-reduced) system propagated by the **exact
  matrix exponential** — no numerical integration step error.
- **Event system.** Comparators/diodes/switches/flip-flops are monitored with
  a crossing search that guarantees the **leftmost root**, budgets its work,
  and declares its event bandwidth (dt/16) instead of silently dropping
  narrow pulse pairs.
- **POP.** Newton iteration on the one-period map (N = 1, period-2 fallback),
  with a discrete-signature check that the switching sequence really repeats.
- **PAC.** The loop gain is assembled from the analytic monodromy matrix:
  exact propagation between events, **saltation matrices** at events
  (including zero-duration event-chain folding and the moving-section terminal
  saltation). No perturbation is injected.
- **Self-validation gate.** Before any PM/GM is reported, the analytic
  construction is checked against central-difference columns of the true
  trigger map (multi-epsilon, minimum over a noise-aware ladder). A
  construction that disagrees is **reported INVALID and margins are
  suppressed** — the tool refuses to publish numbers it cannot verify.

## Verification

popac is unusually serious about being checkable:

- **Executable golden specs** (`tests/golden/`): every example model's frozen
  orbit — state vector, event stream, Floquet spectrum, PAC curve — is
  regenerated bit-for-bit in CI. Engine changes that move a golden fail the
  build; legitimate moves must regenerate with a documented reason.
- **Physics laws** (`tests/test_physics_laws.py`): regulation, charge and
  volt-second balance, ripple bounds asserted against **closed-form
  predictions from model parameters only** — an authority outside the
  engine's own math.
- **Independent reference**: a hand-written `solve_ivp` (RK45, rtol 1e-10)
  chattering-trajectory cross-check agrees to 4×10⁻⁷ V.
- **PAC vs injection**: analytic and perturbation-based loop gains are
  cross-asserted at crossover on multiple topologies.

## Positioning

| | transient SPICE (LTspice / ngspice) | **popac** |
|---|---|---|
| periodic steady state | — (transient only) | ✓ (POP) |
| loop gain | transient injection sweep, ~1 h | ✓ **analytic (PAC), ~seconds** |
| Floquet stability spectrum | — | ✓ |
| refuses unverifiable results | — | ✓ validation gate |
| license | free / open source | **Apache-2.0** |
| schematic GUI, thermal, HIL | ✓ (varies by tool) | — (Python DSL; see roadmap) |

popac is **not** a general-purpose SPICE and does not try to be: it is a
focused, scriptable, algorithmically transparent loop analyzer for switching
converters. POP (periodic operating point, i.e. the periodic steady state)
and PAC (periodic AC analysis) name generic published analysis methods.

## Modeling in 60 seconds

A converter is a YAML netlist of behavioral devices — power stage (R/L/C/V/D/
SW), sensing (H/G/E), control (CMP/GATE/SRFF/TRIG), and the analysis block:

```yaml
devices:
  - {kind: V, name: VIN, n1: vin, n2: "0", value: {dc: 12}}
  - {kind: SW, name: S1, d1: vin, d2: sw, c1: u2o, c2: "0",
     ron: 1m, roff: 10Meg, threshold: 2, hystwd: 100m, ic: OPEN}
  - {kind: L, name: L1, n1: sw, n2: vout, l: 33u, ic_i: 0}
  # ... diode, output cap, divider, EA, comparator, SR latch ...
analyses:
  pop: {pre_cycles: 100, max_iter: 100, period_guess: 5u}
  pac: {fstart: 10, fstop: 50k, per_decade: 10,
        inject_src: V16, probe_a: vout, probe_b: fbt}
```

See `models/buck5v.yaml` for a fully commented reference.

## Roadmap

- LLC resonant converter (variable-frequency VCO blocks already land in
  `models/vco_block.yaml`; the full model needs a monitor-cost optimization)
- forward / half-bridge / full-bridge / push-pull
- PFC (needs an analog multiplier device) and multi-winding transformers
- performance: monitor-batch evaluation (the known hot spot)

## Limitations (honest)

- Python-speed engine: a POP solve costs seconds-to-minutes per model, not
  sub-second.
- Electrical domain only — no thermal, magnetic, or loss models.
- Behavioral device set (see `popac/ir.py`); no SPICE netlist import yet.
- Convergence is honest but not guaranteed: models declare conditioning
  parasitics where the exact PWL formulation needs them, and failures are
  reported, not hidden.

## Development

```bash
python -m pytest -q              # full suite (goldens included)
python -m pytest -q -m "not tierb"   # fast lane (skip the slow golden tier)
python tests/test_physics_laws.py    # closed-form physics, any single suite
python debug/make_golden.py <name>   # regenerate a golden (document why!)
```

Contributions welcome — new topologies with closed-form physics assertions
are the ideal first PR (see any `tests/test_<topology>.py` for the pattern).

## License

Apache-2.0 (see `LICENSE`). Numerical output carries no warranty; verify
against your own measurements as you would with any simulator.
