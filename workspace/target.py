"""PUBLIC coder toolkit for the dhruv-fhm-gatecount campaign (PROTECTED: read
only -- the diff guard rejects any edit outside candidate.py).

The target: one first-order Trotter step of the N=6 Fermi-Hubbard model as a Z2
lattice gauge theory (arXiv:2411.07778), J=1, U=2, dt=0.3, 18 qubits (3/site:
up-matter, down-matter, one bond tau; periodic).  H (Eq. 7):
  H_J = -4J sum_{i,sigma}(tau^z S+_{i,s} S-_{i+1,s} + h.c.) = -2J sum tau^z(XX+YY)
  H_U = (U/2) sum_i tau^x_{i-1,i} tau^x_{i,i+1}
The IDEAL unitary U_ideal = the exact ordered product of block exponentials for
one Trotter step (Dhruv's uncompiled C-hat).  Your job: return a circuit over
IonQ Aria native gates that reproduces U_ideal to mean infidelity <= 1e-4 using
as FEW two-qubit (MS) gates as possible.

Native gate set (the scored metric counts MS gates only; GPi/GPi2/RZ are free):
  ("GPi",  q, phi)                 pi rotation about axis phi in XY plane
  ("GPi2", q, phi)                 pi/2 rotation about axis phi in XY plane
  ("RZ",   q, theta)               virtual-Z frame change (exact, free)
  ("MS", q0, q1, phi0, phi1, theta) arbitrary-angle Molmer-Sorensen (ONLY entangler)

Reference points (per Dhruv, N=6): direct compilation 14N=84 MS; his hand-
optimized 9N=54 MS at ~1e-4 infidelity.  The immutable baseline here is a naive
exact ladder compilation (102 MS).  Match 54 to equal his hand-design; beat it
to surpass it.  Self-check with estimate_infidelity(); the blinded scorer
certifies independently on hidden random states.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

# ---- constants (FROZEN) ----
N = 6
J = 1.0
U = 2.0
DT = 0.3
NQ = 3 * N  # 18
EPS = 1e-4  # infidelity bar (Dhruv's ~1e-4)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _sigma(phi):
    return np.cos(phi) * X + np.sin(phi) * Y


# ---- qubit layout ----
def m_up(i):  return 3 * (i % N)
def m_dn(i):  return 3 * (i % N) + 1
def tau(b):   return 3 * (b % N) + 2   # bond b joins site b and (b+1)%N


# ---- native gate matrices ----
def gpi(phi):
    return np.array([[0, np.exp(-1j * phi)], [np.exp(1j * phi), 0]], dtype=complex)


def gpi2(phi):
    c = 1 / np.sqrt(2)
    return c * np.array([[1, -1j * np.exp(-1j * phi)],
                         [-1j * np.exp(1j * phi), 1]], dtype=complex)


def rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)


def ms(phi0, phi1, theta):
    return expm(-1j * (theta / 2) * np.kron(_sigma(phi0), _sigma(phi1)))


NATIVE = {"GPi", "GPi2", "RZ", "MS"}


def is_native(circuit):
    for g in circuit:
        op = g[0]
        if op not in NATIVE:
            return False, f"non-native op {op!r}"
        if op in ("GPi", "GPi2", "RZ"):
            if len(g) != 3 or not (0 <= g[1] < NQ):
                return False, f"bad {op}: {g}"
        elif op == "MS":
            if len(g) != 6 or g[1] == g[2] or not (0 <= g[1] < NQ and 0 <= g[2] < NQ):
                return False, f"bad MS: {g}"
    return True, "ok"


def ms_count(circuit):
    return sum(1 for g in circuit if g[0] == "MS")


# ---- statevector simulator ----
def apply_gate(psi, Umat, qubits):
    k = len(qubits)
    Ur = Umat.reshape((2,) * (2 * k))
    psi = np.tensordot(Ur, psi, axes=(list(range(k, 2 * k)), qubits))
    return np.moveaxis(psi, list(range(k)), qubits)


def apply_circuit(psi, circuit):
    for g in circuit:
        if g[0] == "GPi":
            psi = apply_gate(psi, gpi(g[2]), [g[1]])
        elif g[0] == "GPi2":
            psi = apply_gate(psi, gpi2(g[2]), [g[1]])
        elif g[0] == "RZ":
            psi = apply_gate(psi, rz(g[2]), [g[1]])
        else:
            psi = apply_gate(psi, ms(g[3], g[4], g[5]), [g[1], g[2]])
    return psi


# ---- ideal one-Trotter-step operator (block exponentials, FROZEN order) ----
def _hop_block():
    XX, YY = np.kron(X, X), np.kron(Y, Y)
    return expm(-1j * DT * (-2 * J * np.kron(Z, XX + YY)))  # order [tau, a, b]


def _int_block():
    return expm(-1j * DT * (U / 2) * np.kron(X, X))         # order [tau_prev, tau_i]


HOP_UNITARY = _hop_block()   # 8x8 on [tau, m_a, m_b]
INT_UNITARY = _int_block()   # 4x4 on [tau_prev, tau_i]


def hop_qubits(b):
    """The two 3-qubit hopping blocks on bond b: (up, down). Order [tau,a,b]."""
    i, jn = b, (b + 1) % N
    return ([tau(b), m_up(i), m_up(jn)], [tau(b), m_dn(i), m_dn(jn)])


def int_qubits(i):
    """On-site block at site i, order [tau_prev, tau_i]."""
    return [tau((i - 1) % N), tau(i)]


def ideal_step(psi):
    """U_ideal. FROZEN order: hopping (bond 0..5, up then down), then on-site
    (site 0..5)."""
    for b in range(N):
        qa, qb = hop_qubits(b)
        psi = apply_gate(psi, HOP_UNITARY, qa)
        psi = apply_gate(psi, HOP_UNITARY, qb)
    for i in range(N):
        psi = apply_gate(psi, INT_UNITARY, int_qubits(i))
    return psi


# ---- random states + self-check infidelity (coder-side; NOT the hidden battery) ----
def haar_state(rng):
    v = rng.standard_normal(2 ** NQ) + 1j * rng.standard_normal(2 ** NQ)
    return (v / np.linalg.norm(v)).reshape((2,) * NQ)


def product_state(rng):
    psi = np.array([1.0], dtype=complex)
    for _ in range(NQ):
        a = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        psi = np.kron(psi, a / np.linalg.norm(a))
    return psi.reshape((2,) * NQ)


def estimate_infidelity(circuit, n_states=24, seed=12345):
    """Mean 1-|<ideal|cand>|^2 over your own random states. Use this to self-
    check; the blinded scorer uses different, hidden states."""
    rng = np.random.default_rng(seed)
    tot = 0.0
    for k in range(n_states):
        psi = haar_state(rng) if k % 2 == 0 else product_state(rng)
        a = ideal_step(psi).ravel()
        b = apply_circuit(psi, circuit).ravel()
        tot += 1 - abs(np.vdot(a, b)) ** 2
    return tot / n_states


# ---- single-qubit native decomposition (exact, verified) ----
def _mat1(circ):
    """2x2 matrix of a single-qubit native gate list (all on qubit 0)."""
    M = np.eye(2, dtype=complex)
    for g in circ:
        m = {"GPi": gpi, "GPi2": gpi2, "RZ": rz}[g[0]](g[2])
        M = m @ M
    return M


def _eq_phase1(A, B, tol=1e-9):
    i = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    return np.allclose(A * (B[i] / A[i]), B, atol=tol)


def u3_to_native(Umat, q):
    """Exact native {RZ, GPi2} sequence for an arbitrary single-qubit unitary,
    up to global phase.  Closed-form ZYZ -> RZ(lam) GPi2(pi) RZ(-theta) GPi2(0)
    RZ(phi), where RY(theta)=SX.RZ(-theta).SX+ with SX=GPi2(0), SX+=GPi2(pi).
    Verified per call with a BFGS fallback."""
    U00, U01, U10, U11 = Umat[0, 0], Umat[0, 1], Umat[1, 0], Umat[1, 1]
    theta = 2 * np.arctan2(abs(U10), abs(U00))
    ppl = np.angle(U11) - np.angle(U00)              # phi + lam
    pml = np.angle(U10) - np.angle(U01) + np.pi      # phi - lam
    phi, lam = (ppl + pml) / 2, (ppl - pml) / 2
    circ = [("RZ", q, lam), ("GPi2", q, np.pi), ("RZ", q, -theta),
            ("GPi2", q, 0.0), ("RZ", q, phi)]
    if _eq_phase1(_mat1([(g[0], 0, g[2]) for g in circ]), Umat, tol=1e-11):
        return circ
    # polish/fallback in the RZ.SX.RZ.SX.RZ ansatz, tight convergence
    def loss(x):
        M = _mat1([("RZ", 0, x[0]), ("GPi2", 0, 0.0), ("RZ", 0, x[1]),
                   ("GPi2", 0, 0.0), ("RZ", 0, x[2])])
        return 1 - abs(np.trace(M.conj().T @ Umat)) ** 2 / 4
    starts = [np.array([lam, np.pi - theta, phi])] + \
             list(np.random.default_rng(0).uniform(-np.pi, np.pi, (10, 3)))
    best = min((minimize(loss, x0, method="BFGS", options={"gtol": 1e-14, "maxiter": 4000})
                for x0 in starts), key=lambda r: r.fun)
    a, b, c = best.x
    return [("RZ", q, a), ("GPi2", q, 0.0), ("RZ", q, b), ("GPi2", q, 0.0), ("RZ", q, c)]


# ---- IPG-style block fitter (angle re-optimization of a k-MS ansatz) ----
def fit_block(target_unitary, qubits, n_ms, pairs=None, restarts=30, seed=0, tol=1e-9):
    """Fit a native circuit of `n_ms` MS gates (with free single-qubit layers)
    to `target_unitary` on the given global `qubits` (order matches the unitary's
    tensor order).  Returns (native_circuit_on_global_qubits, infidelity).

    This is the sanctioned IPG-style tool: you choose structure (how many MS,
    which local pairs, which blocks) and this re-optimizes the continuous angles.
    """
    nq = len(qubits)
    dim = 2 ** nq

    def u3(p):
        return rz(p[1]) @ _ry(p[0]) @ rz(p[2])

    def _emb(U1, qs):
        op = U1.reshape((2,) * (2 * len(qs)))
        psi = np.eye(dim, dtype=complex).reshape((2,) * nq + (dim,))
        psi = np.tensordot(op, psi, axes=(list(range(len(qs), 2 * len(qs))), list(qs)))
        return np.moveaxis(psi, list(range(len(qs))), list(qs)).reshape(dim, dim)

    def full_layer(x, M):
        for qi in range(nq):
            M = _emb(u3(x[qi * 3:qi * 3 + 3]), [qi]) @ M
        return M

    def build(x, seq):
        off = 0
        M = full_layer(x[off:off + 3 * nq], np.eye(dim, dtype=complex)); off += 3 * nq
        for (pa, pb) in seq:
            phi0, phi1, th = x[off:off + 3]; off += 3
            M = _emb(ms(phi0, phi1, th), [pa, pb]) @ M
            M = full_layer(x[off:off + 3 * nq], M); off += 3 * nq
        return M

    ndim = 3 * nq * (n_ms + 1) + 3 * n_ms  # full single-qubit layer around every entangler

    def fit_seq(seq, nstarts, rng):
        bx, bf = None, 2.0
        for _ in range(nstarts):
            x0 = rng.uniform(-np.pi, np.pi, ndim)
            r = minimize(lambda x: 1 - abs(np.trace(build(x, seq).conj().T @ target_unitary)) ** 2 / dim ** 2,
                         x0, method="BFGS", options={"maxiter": 3000, "gtol": 1e-13})
            if r.fun < bf:
                bf, bx = r.fun, r.x
            if bf < tol:
                break
        return bx, bf

    allp = [(i, j) for i in range(nq) for j in range(i + 1, nq)]
    if pairs is not None:
        seqs = [list(pairs)]
    else:  # search a curated + random set of pair sequences
        seqs = [[allp[k % len(allp)] for k in range(n_ms)]]          # round-robin
        seqs += [[p] * n_ms for p in allp]                           # single-pair
        srng = np.random.default_rng(seed + 1)
        for _ in range(8):
            seqs.append([tuple(allp[i]) for i in srng.integers(0, len(allp), n_ms)])
    best_x, best, best_seq = None, 2.0, seqs[0]
    for seq in seqs:
        rng = np.random.default_rng(seed)
        bx, bf = fit_seq(seq, restarts, rng)
        if bf < best:
            best, best_x, best_seq = bf, bx, seq
        if best < tol:
            break

    # materialize best_x into native gates on the GLOBAL qubits
    circ, off = [], 0
    for qi in range(nq):
        circ += u3_to_native(u3(best_x[off + qi * 3:off + qi * 3 + 3]), qubits[qi])
    off += 3 * nq
    for (pa, pb) in best_seq:
        phi0, phi1, th = best_x[off:off + 3]; off += 3
        circ.append(("MS", qubits[pa], qubits[pb], float(phi0), float(phi1), float(th)))
        for qi in range(nq):
            circ += u3_to_native(u3(best_x[off + qi * 3:off + qi * 3 + 3]), qubits[qi])
        off += 3 * nq
    return circ, float(best)


def _ry(theta):
    return expm(-1j * theta / 2 * Y)
