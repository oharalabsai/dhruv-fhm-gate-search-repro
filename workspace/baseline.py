"""Immutable baseline candidate for dhruv-fhm-gatecount (copied to incumbent.py
at setup).  A naive-but-EXACT native compilation of one Trotter step of the N=6
Fermi-Hubbard Z2-LGT Hamiltonian: each 3-qubit hopping block is expanded as two
weight-3 Pauli exponentials via the standard CNOT-ladder gadget (4 MS each),
each on-site block is one MS.  Total = 12*8 + 6 = 102 MS gates.

Self-contained by design: build_circuit() returns an ordered native-gate list
and imports nothing from the workspace (the blinded scorer scores a COPY of this
file from the experiments/ dir, where workspace modules are not importable).

NOTE (honesty): 102 MS is OUR naive ladder ruler, NOT a reproduction of Dhruv's
reported direct 14N=84.  His 84 uses a tighter native decomposition; his hand-
optimized result is 9N=54.  The campaign's win bar is the absolute 54.
"""
from __future__ import annotations
import numpy as np

N, J, U, DT = 6, 1.0, 2.0, 0.3


def m_up(i): return 3 * (i % N)
def m_dn(i): return 3 * (i % N) + 1
def tau(b):  return 3 * (b % N) + 2


def _cnot(c, t):
    """CNOT(c->t) up to global phase, one MS (locals verified by grid search)."""
    return [("GPi2", c, np.pi / 2), ("MS", c, t, 0.0, 0.0, np.pi / 2),
            ("RZ", c, -np.pi), ("GPi2", t, np.pi), ("GPi2", c, np.pi / 2),
            ("RZ", c, -3 * np.pi / 2)]


def _basis_in(q, p):
    return {"X": [("GPi2", q, np.pi / 2)], "Y": [("GPi2", q, 0.0)]}.get(p, [])


def _basis_out(q, p):
    return {"X": [("GPi2", q, np.pi / 2 + np.pi)], "Y": [("GPi2", q, np.pi)]}.get(p, [])


def _pauli_exp(paulis, gamma):
    """exp(-i gamma prod P) as native gates (basis change + CNOT ladder + RZ)."""
    qs = [q for q, _ in paulis]
    c = []
    for q, p in paulis:
        c += _basis_in(q, p)
    for a, b in zip(qs[:-1], qs[1:]):
        c += _cnot(a, b)
    c += [("RZ", qs[-1], 2 * gamma)]
    for a, b in reversed(list(zip(qs[:-1], qs[1:]))):
        c += _cnot(a, b)
    for q, p in paulis:
        c += _basis_out(q, p)
    return c


def build_circuit():
    """Return the baseline native-gate circuit (102 MS) for one Trotter step."""
    circ = []
    g = 2 * J * DT  # hopping: exp(+i g tau^z (XX+YY)) = exp(i g ZXX) exp(i g ZYY)
    for b in range(N):
        i, jn = b, (b + 1) % N
        for mfun in (m_up, m_dn):
            c, a, bb = tau(b), mfun(i), mfun(jn)
            circ += _pauli_exp([(c, "Z"), (a, "X"), (bb, "X")], -g)
            circ += _pauli_exp([(c, "Z"), (a, "Y"), (bb, "Y")], -g)
    a_int = (U / 2) * DT  # on-site: exp(-i a_int XX) = MS(0,0, 2 a_int)
    for i in range(N):
        circ.append(("MS", tau((i - 1) % N), tau(i), 0.0, 0.0, 2 * a_int))
    return circ
