"""P4 acceptance cross-validation on the ORIGINAL-fidelity model.

PAC predicts fc=30.03kHz / PM=57.2deg there. Run direct-injection AC at the
same single frequency (discard_cycles=300 default) and compare |T| and phase.
Target: |dT| < 0.5 dB, dphi < 1 deg (review P4 criteria).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from popac.model_loader import load_yaml
from popac.engine import Engine
from popac.pop import PopSolver
from popac.pac import PacSolver
from popac.ac import AcSolver

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'models/boost24.yaml'
f_test = float(sys.argv[2]) if len(sys.argv) > 2 else 30.03e3

ckt, ana = load_yaml(MODEL)
e = Engine(ckt, {'max_events': 50_000_000})
pop = PopSolver(e, dict(ana.get('pop', {}))).solve()
print(f'POP ok={pop.residual:.3e} T={pop.period:.6e}', flush=True)
assert pop.ok

e.restore(getattr(pop, 'snap0', None) or e.snapshot())
e.x = pop.x.copy()
snap = e.snapshot()

pac = PacSolver(ckt, snap, pop.x,
                {'fstart': f_test, 'fstop': f_test, 'per_decade': 1,
                 'inject_src': 'V16', 'probe_a': 'vout', 'probe_b': 'fbt'})
res = pac.solve()
T_pac = res['T'][0]
print(f'PAC  T({f_test/1e3:g} kHz) = |{abs(T_pac):.4f}| '
      f'{np.degrees(np.angle(T_pac)):.2f} deg', flush=True)

ac = AcSolver(ckt, {'inject_src': 'V16', 'probe_a': 'vout', 'probe_b': 'fbt'},
              pop_snapshot=snap, pop_x=pop.x)
pt = ac.run_point(f_test)
print(f'AC   T({f_test/1e3:g} kHz) = |{pt["T"]:.4f}| '
      f'{np.degrees(np.angle(pt["T"])):.2f} deg status={ac._point_status(pt)}',
      flush=True)
ratio = T_pac / pt['T']
ddb = 20 * np.log10(abs(ratio))
dphi = np.degrees(np.angle(ratio))
print(f'PAC vs AC: d|T|={ddb:+.3f} dB, dphi={dphi:+.3f} deg '
      f'-> {"PASS" if abs(ddb) < 0.5 and abs(dphi) < 1.0 else "FAIL"} '
      f'(targets <0.5dB / <1deg)', flush=True)
