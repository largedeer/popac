# -*- coding: utf-8 -*-
"""Shared helpers for the Tier A goldens (used by debug/make_golden.py and
tests/test_golden.py -- kept in ONE place so generator and checker can
never drift apart).

Canonical event stream (plan §5.2): chunk-end records with no event are
dropped; events within CLUSTER_T_ABS of their predecessor are merged into
a zero-time cluster whose members are sorted by (kind, tag) so ordering
within an instant is not significant; times are relative to the POP
boundary."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CLUSTER_T_ABS = 1e-12      # s: instant-cluster radius
STREAM_T_ABS = 1e-12       # s: cluster-time comparison absolute floor
STREAM_T_REL = 1e-9        # x period: cluster-time comparison relative

GOLDEN_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'golden')
SCHEMA_VERSION = 1


def canonical_stream(records, t_boundary):
    """[(t_rel, [(kind, tag), ...]), ...], clusters merged and sorted."""
    evs = []
    for r in records:
        k = r.get('ev_kind')
        if not k:
            continue                      # plain chunk end, no event
        evs.append((r['t1'] - t_boundary, str(k), str(r.get('ev_tag') or '')))
    out = []
    for t, k, tag in evs:
        if out and abs(t - out[-1][0]) <= CLUSTER_T_ABS:
            out[-1][1].append((k, tag))
        else:
            out.append((t, [(k, tag)]))
    for c in out:
        c[1].sort()
    return [(float(t), members) for t, members in out]


def streams_match(a, b, period):
    """(ok, message): same clusters, same members, times within
    STREAM_T_ABS + STREAM_T_REL * period.  Member entries normalized to
    tuples (JSON round-trips them as lists)."""
    if len(a) != len(b):
        return False, f'cluster count {len(a)} vs {len(b)}'
    tol = STREAM_T_ABS + STREAM_T_REL * period
    for i, ((ta, ma), (tb, mb)) in enumerate(zip(a, b)):
        ma = sorted(tuple(m) for m in ma)
        mb = sorted(tuple(m) for m in mb)
        if ma != mb:
            return False, (f'cluster {i} members differ: '
                           f'{ma} vs {mb}')
        if abs(ta - tb) > tol:
            return False, f'cluster {i} time {ta!r} vs {tb!r} (tol {tol:g})'
    return True, ''
