# -*- coding: utf-8 -*-
"""Engineering-notation number parsing: '4.7u' -> 4.7e-6, '100E18' -> 1e20 ..."""

_SUF = {
    'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6, 'μ': 1e-6,
    'm': 1e-3, 'k': 1e3, 'K': 1e3, 'meg': 1e6, 'Meg': 1e6, 'MEG': 1e6,
    'Meg'.upper(): 1e6, 'g': 1e9, 'G': 1e9, 't': 1e12, 'T': 1e12,
}


def eng(value):
    """Parse int/float/'4.7u'/'100E18'/'1meg' -> float. Raises ValueError."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        raise ValueError("empty value")
    # plain scientific / decimal first
    try:
        return float(s)
    except ValueError:
        pass
    low = s.lower()
    if low in ('meg',):
        raise ValueError(f"bad number {value!r}")
    # meg* before single-letter m
    if low.endswith('meg'):
        return float(s[:-3]) * 1e6
    if s[-1] in _SUF:
        return float(s[:-1]) * _SUF[s[-1]]
    raise ValueError(f"cannot parse number {value!r}")
