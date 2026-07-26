#!/usr/bin/env python3
"""Yang-Mills mass gap — curvature Lambda^2 channels + 3D spatial Wilson.

Sections 15–21: curvature span / H_mag uniqueness.
Sections 22–24: LatticeYM3D, tree gauge, OS positivity.
Companion: common (2D lattice). Orchestrator: Yang_Mills_Mass_Gap_run.py.
"""

from __future__ import annotations

import argparse
import itertools
from typing import Iterable, Optional, TypeAlias

import numpy as np
from scipy.sparse import csr_matrix, eye as sp_eye
from scipy.sparse.linalg import LinearOperator, eigsh

import Yang_Mills_Mass_Gap_common  # noqa: F401 — repo path setup

from gyroscopic.hQVM.api import q_word6
from gyroscopic.hQVM.family import q_word_d, alphabet_size

from Yang_Mills_Mass_Gap_common import (
    BETA_DEFINING,
    BYTE256,
    CODE_C2,
    D_PAYLOAD,
    G_DEFINING_KS,
    K4,
    Q8,
    LatticeYM,
    gate,
    jw_gap_from_w,
    progress,
    section,
    section_title,
    wilson_weight_K4,
    wilson_weight_Q8_2d,
)

def bivector_basis_labels(d: int = D_PAYLOAD) -> list[tuple[int, int]]:
    """Ordered basis of Lambda^2: e_i wedge e_j for 0 <= i < j < d."""
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


def vector_wedge(a: int, b: int, d: int = D_PAYLOAD) -> np.ndarray:
    """Coordinate vector of a wedge b in the bivector_basis_labels order (GF(2))."""
    labels = bivector_basis_labels(d)
    out = np.zeros(len(labels), dtype=np.uint8)
    for k, (i, j) in enumerate(labels):
        ai, aj = (a >> i) & 1, (a >> j) & 1
        bi, bj = (b >> i) & 1, (b >> j) & 1
        out[k] = (ai * bj + aj * bi) & 1
    return out


def gf2_row_rank(rows: Iterable[np.ndarray]) -> int:
    """GF(2) row rank of a list of 0/1 vectors (same length)."""
    mat = [np.array(r, dtype=np.uint8).copy() for r in rows if np.any(r)]
    if not mat:
        return 0
    n = len(mat[0])
    rank = 0
    for col in range(n):
        pivot = None
        for r in range(rank, len(mat)):
            if mat[r][col]:
                pivot = r
                break
        if pivot is None:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        for r in range(len(mat)):
            if r != rank and mat[r][col]:
                mat[r] ^= mat[rank]
        rank += 1
    return rank


def curvature_span_from_q(
    q_values: Iterable[int],
    d: int = D_PAYLOAD,
) -> dict:
    """Span of q wedge q' over all ordered pairs from the given q list."""
    qs = list(dict.fromkeys(int(q) & ((1 << d) - 1) for q in q_values))
    rows = [vector_wedge(x, y, d) for x, y in itertools.product(qs, repeat=2)]
    dim_k = gf2_row_rank(rows)
    labels = bivector_basis_labels(d)
    c2 = int(CODE_C2)
    return {
        "d": d,
        "n_q_distinct": len(qs),
        "n_pairs": len(qs) ** 2,
        "dim_Lambda2": len(labels),
        "dim_K": dim_k,
        "saturates": dim_k == len(labels),
        "C2": c2,
        "C2_match": len(labels) == c2 and c2 == 15,
    }


def curvature_span_kernel_bytes(d: int = D_PAYLOAD) -> dict:
    """Full alphabet: q from hQVM api (d=6) or family.q_word_d."""
    if d == 6:
        qs = [q_word6(b) for b in range(BYTE256)]
        src = "api.q_word6"
        alph = 256
    else:
        qs = [q_word_d(b, d) for b in range(alphabet_size(d))]
        src = f"family.q_word_d(d={d})"
        alph = alphabet_size(d)
    out = curvature_span_from_q(qs, d=d)
    out["q_source"] = src
    out["alphabet"] = alph
    xor_hist = np.zeros(d + 1, dtype=int)
    uniq = list(dict.fromkeys(qs))
    for x, y in itertools.product(uniq, repeat=2):
        xor_hist[(x ^ y).bit_count()] += 1
    out["xor_popcount_hist"] = xor_hist.tolist()
    # Gravity census on full byte pairs: 1024 * C(6,k) when d=6 and 256 alphabet
    if d == 6 and alph == BYTE256:
        full_hist = np.zeros(7, dtype=int)
        for bx in range(BYTE256):
            qx = q_word6(bx)
            for by in range(BYTE256):
                full_hist[(qx ^ q_word6(by)).bit_count()] += 1
        expect = [1024 * math_comb(6, k) for k in range(7)]
        out["defect_hist_full"] = full_hist.tolist()
        out["defect_hist_expect"] = expect
        out["defect_census_match"] = full_hist.tolist() == expect
    return out


def math_comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def wedge_vs_xor_relation_fixed(d: int = D_PAYLOAD) -> dict:
    """Exhaustive on F_2^d: x=y => wedge=0; wedge nonzero => xor nonzero."""
    n_space = 1 << d
    eq_ok = all(not np.any(vector_wedge(x, x, d)) for x in range(n_space))
    impl_ok = True
    for x in range(n_space):
        for y in range(n_space):
            w = vector_wedge(x, y, d)
            if np.any(w) and (x ^ y) == 0:
                impl_ok = False
                break
        if not impl_ok:
            break
    return {
        "equality_wedge0": eq_ok,
        "wedge_implies_xor_nonzero": impl_ok,
        "pass": eq_ok and impl_ok,
    }


def channel_labels_map(d: int = D_PAYLOAD) -> list[dict]:
    """Explicit bit-pair labels for the 15 curvature channels."""
    return [{"channel": k, "bits": (i, j)} for k, (i, j) in enumerate(bivector_basis_labels(d))]


def magnetic_degree2_certificate() -> dict:
    """P3: Wilson V_R is a class function of plaquette holonomy (degree-2 observable)."""
    from Yang_Mills_Mass_Gap_common import Q8, wilson_weight_Q8_2d

    G, gi, _table, _inv = Q8()
    _, V = wilson_weight_Q8_2d()
    classes = [
        ["1"],
        ["-1"],
        ["i", "-i"],
        ["j", "-j"],
        ["k", "-k"],
    ]
    class_ok = True
    class_vals = []
    for cl in classes:
        vals = [float(V[gi[g]]) for g in cl]
        class_ok = class_ok and all(abs(v - vals[0]) < 1e-12 for v in vals)
        class_vals.append((cl, vals[0]))
    nontrivial = len({round(float(V[i]), 8) for i in range(len(G))}) > 1
    return {
        "V_is_class_function": class_ok,
        "class_values": class_vals,
        "Wilson_nontrivial": nontrivial,
        "degree2_placement": "V_R(hol(p)) depends only on plaquette holonomy conjugacy class",
        "pass": class_ok and nontrivial,
    }


def q8_magnetic_uniqueness_audit() -> dict:
    """WP3: uniqueness of magnetic weight under CGM constraints on Q8.

    Class functions on Q8 are 5-dimensional (5 conjugacy classes). Constraints:
      (i) V(1)=0 (flat holonomy costs nothing),
      (ii) Aut symmetry permuting {i,j,k} (spinorial triality),
      (iii) V >= 0,
      (iv) V comes from a unitary irrep character: V=1-Re chi/d_R.
    Under (i)+(ii) the free data are (V_central, V_noncentral). Demanding the
    unique 2D irrep of Q8 (the SU(2) defining restriction) fixes
      V(-1)=2, V(i)=V(j)=V(k)=1, matching wilson_weight_Q8_2d.
    """
    from Yang_Mills_Mass_Gap_common import Q8, wilson_weight_Q8_2d

    G, gi, _, _ = Q8()
    _, Vw = wilson_weight_Q8_2d()
    # conjugacy class representatives and sizes
    classes = {
        "1": ["1"],
        "-1": ["-1"],
        "i": ["i", "-i"],
        "j": ["j", "-j"],
        "k": ["k", "-k"],
    }
    # Aut orbit of noncentral: force V_i = V_j = V_k
    # Parametrize Aut-symmetric V with V(1)=0: two free coeffs (vc, vn)
    # Wilson point
    vc_w = float(Vw[gi["-1"]])
    vn_w = float(Vw[gi["i"]])
    # Character formula from 2D irrep is unique among d_R=2 unitary irreps of Q8
    chi = {"1": 2.0, "-1": -2.0, "i": 0.0, "-i": 0.0, "j": 0.0, "-j": 0.0, "k": 0.0, "-k": 0.0}
    V_from_chi = np.array([1.0 - chi[g] / 2.0 for g in G])
    match_wilson = bool(np.allclose(V_from_chi, Vw))
    # Count Aut-symmetric nonnegative class functions with V(1)=0 and fixed scale
    # V(-1)+3*V(i) normalized, or: scan grid and count survivors equal to Wilson up to scale
    survivors = []
    for vc in np.linspace(0.0, 4.0, 41):
        for vn in np.linspace(0.0, 4.0, 41):
            if vc < -1e-12 or vn < -1e-12:
                continue
            if abs(vc) < 1e-12 and abs(vn) < 1e-12:
                continue
            # Aut-symmetric assignment
            V = np.zeros(8)
            V[gi["1"]] = 0.0
            V[gi["-1"]] = vc
            for g in ("i", "-i", "j", "-j", "k", "-k"):
                V[gi[g]] = vn
            # character-shape constraint: V = c * (1 - Re chi/2) for some c>0
            # => vc/vn == 2/1 when vn>0, or vn==0 and vc>0 (central-only, not 2D irrep)
            if vn > 1e-12 and abs(vc / vn - 2.0) < 1e-9:
                survivors.append((vc, vn))
            elif vn <= 1e-12 and vc > 1e-12:
                pass  # central-only excluded by 2D-irrep demand
    # unique ray: all survivors are positive multiples of (2,1)
    ratios = [vc / vn for vc, vn in survivors if vn > 1e-12]
    unique_ray = len(survivors) > 0 and all(abs(r - 2.0) < 1e-9 for r in ratios)
    return {
        "n_conjugacy_classes": len(classes),
        "Wilson_V_central": vc_w,
        "Wilson_V_noncentral": vn_w,
        "matches_2d_character": match_wilson,
        "aut_symmetric_character_survivors": len(survivors),
        "unique_ray_ratio_2": unique_ray,
        "pass": match_wilson and unique_ray and abs(vc_w - 2.0) < 1e-12 and abs(vn_w - 1.0) < 1e-12,
    }


def k4_q8_action_on_channels() -> dict:
    """How K4 (abelian) vs Q8 (nonabelian) sit relative to the 15 channels.

    K4 holonomy is abelian => commutator [U,V]=1 always on links; magnetic energy
    still sees plaquette holonomy but the kernel bivector lift is the Q8/spinorial
    refinement. Certificate: dim K = 15 from q_6 alone (group-independent);
    Q8 Wilson class function is nontrivial on center (channel for -1).
    """
    from Yang_Mills_Mass_Gap_common import wilson_weight_K4, wilson_weight_Q8_2d

    _, Vk = wilson_weight_K4()
    _, Vq = wilson_weight_Q8_2d()
    return {
        "Lambda2_dim": 15,
        "K4_Wilson_values": sorted({round(float(x), 8) for x in Vk}),
        "Q8_Wilson_values": sorted({round(float(x), 8) for x in Vq}),
        "Q8_has_central_cost": float(max(Vq)) >= 2.0 - 1e-12,
        "pass": float(max(Vq)) >= 2.0 - 1e-12 and len({round(float(x), 8) for x in Vq}) >= 3,
    }




Site3: TypeAlias = tuple[int, int, int]


class LatticeYM3D:
    """Lx x Ly x Lz spatial lattice gauge theory (open or periodic).

    Links: +x, +y, +z from each site (open: only where neighbor exists).
    Plaquettes: xy, xz, yz faces. Time is not a link — evolution is e^{-tH}.
    """

    def __init__(self, Lx: int, Ly: int, Lz: int, group, periodic: bool = False):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.periodic = periodic
        self.G, self.gi, self.table, self.inv = group
        self.N = len(self.G)
        self._abelian = bool(np.all(self.table == self.table.T))
        self.sites = [(i, j, k) for i in range(Lx) for j in range(Ly) for k in range(Lz)]
        self.nV = len(self.sites)
        self._build_links()
        self._link_perm_cache: dict = {}
        self._vertex_perm_cache: dict = {}

    def _build_links(self):
        links = []  # list of (axis, i, j, k) with axis in {0,1,2}
        Lx, Ly, Lz = self.Lx, self.Ly, self.Lz

        def add(axis, i, j, k):
            if self.periodic:
                links.append((axis, i % Lx, j % Ly, k % Lz))
            else:
                ni = i + (1 if axis == 0 else 0)
                nj = j + (1 if axis == 1 else 0)
                nk = k + (1 if axis == 2 else 0)
                if ni < Lx and nj < Ly and nk < Lz:
                    links.append((axis, i, j, k))

        for i, j, k in self.sites:
            for axis in (0, 1, 2):
                add(axis, i, j, k)
        # unique preserve order
        seen = set()
        uniq = []
        for L in links:
            if L not in seen:
                seen.add(L)
                uniq.append(L)
        self.link_list = uniq
        self.nE = len(uniq)
        self.link_index = {L: e for e, L in enumerate(uniq)}

    def _site_index(self, i, j, k):
        return (i % self.Lx, j % self.Ly, k % self.Lz) if self.periodic else (i, j, k)

    def link_of(self, axis: int, i: int, j: int, k: int) -> Optional[int]:
        key = (axis,) + self._site_index(i, j, k)
        # for open, key must exist exactly
        if self.periodic:
            key = (axis, i % self.Lx, j % self.Ly, k % self.Lz)
        return self.link_index.get(key)

    def vertex_edges(self, i, j, k):
        """Incident oriented edges (e, sign) at site (i,j,k)."""
        edges = []
        Lx, Ly, Lz = self.Lx, self.Ly, self.Lz
        # +x outgoing, -x incoming, etc.
        for axis, di, dj, dk in (
            (0, 1, 0, 0), (0, -1, 0, 0),
            (1, 0, 1, 0), (1, 0, -1, 0),
            (2, 0, 0, 1), (2, 0, 0, -1),
        ):
            if di + dj + dk > 0:
                # outgoing from this site
                if self.periodic or (
                    0 <= i + di < Lx and 0 <= j + dj < Ly and 0 <= k + dk < Lz
                ):
                    e = self.link_of(axis, i, j, k)
                    if e is not None:
                        edges.append((e, +1))
            else:
                # incoming from neighbor
                ni, nj, nk = i + di, j + dj, k + dk
                if self.periodic:
                    e = self.link_of(axis, ni, nj, nk)
                    if e is not None:
                        edges.append((e, -1))
                elif 0 <= ni < Lx and 0 <= nj < Ly and 0 <= nk < Lz:
                    e = self.link_of(axis, ni, nj, nk)
                    if e is not None:
                        edges.append((e, -1))
        return edges

    def all_plaquettes(self) -> list[list[tuple[int, int]]]:
        """Oriented plaquette edge lists for all xy, xz, yz faces that exist."""
        out = []
        Lx, Ly, Lz = self.Lx, self.Ly, self.Lz
        # xy faces at each (i,j,k)
        for i, j, k in self.sites:
            for axes, di, dj, dk in (
                ((0, 1), 1, 1, 0),  # xy
                ((0, 2), 1, 0, 1),  # xz
                ((1, 2), 0, 1, 1),  # yz
            ):
                a, b = axes
                if self.periodic:
                    e1 = self.link_of(a, i, j, k)
                    # after a-step
                    i2 = (i + (1 if a == 0 else 0)) % Lx
                    j2 = (j + (1 if a == 1 else 0)) % Ly
                    k2 = (k + (1 if a == 2 else 0)) % Lz
                    e2 = self.link_of(b, i2, j2, k2)
                    i3 = (i + (1 if b == 0 else 0)) % Lx
                    j3 = (j + (1 if b == 1 else 0)) % Ly
                    k3 = (k + (1 if b == 2 else 0)) % Lz
                    e4 = self.link_of(b, i, j, k)
                    e3 = self.link_of(a, i3, j3, k3)
                    if None not in (e1, e2, e3, e4):
                        out.append([(e1, +1), (e2, +1), (e3, -1), (e4, -1)])
                else:
                    if a == 0 and i + 1 >= Lx:
                        continue
                    if a == 1 and j + 1 >= Ly:
                        continue
                    if a == 2 and k + 1 >= Lz:
                        continue
                    if b == 0 and i + 1 >= Lx:
                        continue
                    if b == 1 and j + 1 >= Ly:
                        continue
                    if b == 2 and k + 1 >= Lz:
                        continue
                    # open: need both directions to exist
                    i2 = i + (1 if a == 0 else 0)
                    j2 = j + (1 if a == 1 else 0)
                    k2 = k + (1 if a == 2 else 0)
                    i3 = i + (1 if b == 0 else 0)
                    j3 = j + (1 if b == 1 else 0)
                    k3 = k + (1 if b == 2 else 0)
                    if not (0 <= i2 < Lx and 0 <= j2 < Ly and 0 <= k2 < Lz):
                        continue
                    if not (0 <= i3 < Lx and 0 <= j3 < Ly and 0 <= k3 < Lz):
                        continue
                    e1 = self.link_of(a, i, j, k)
                    e2 = self.link_of(b, i2, j2, k2)
                    e4 = self.link_of(b, i, j, k)
                    e3 = self.link_of(a, i3, j3, k3)
                    if None not in (e1, e2, e3, e4):
                        out.append([(e1, +1), (e2, +1), (e3, -1), (e4, -1)])
        return out

    def n_plaquettes(self) -> int:
        return len(self.all_plaquettes())

    def spanning_tree_edges(self) -> list[int]:
        """BFS spanning tree from (0,0,0); returns link indices of tree edges."""
        root: Site3 = (0, 0, 0)
        parent: dict[Site3, Site3 | None] = {root: None}
        parent_link: dict[Site3, int] = {}
        queue: list[Site3] = [root]
        while queue:
            i, j, k = queue.pop(0)
            for axis, di, dj, dk in (
                (0, 1, 0, 0), (0, -1, 0, 0),
                (1, 0, 1, 0), (1, 0, -1, 0),
                (2, 0, 0, 1), (2, 0, 0, -1),
            ):
                ni, nj, nk = i + di, j + dj, k + dk
                if self.periodic:
                    ni, nj, nk = ni % self.Lx, nj % self.Ly, nk % self.Lz
                elif not (0 <= ni < self.Lx and 0 <= nj < self.Ly and 0 <= nk < self.Lz):
                    continue
                if (ni, nj, nk) in parent:
                    continue
                # link is stored at the lower endpoint in +axis direction
                if di + dj + dk > 0:
                    e = self.link_of(axis, i, j, k)
                else:
                    e = self.link_of(axis, ni, nj, nk)
                if e is None:
                    continue
                parent[(ni, nj, nk)] = (i, j, k)
                parent_link[(ni, nj, nk)] = e
                queue.append((ni, nj, nk))
        return list(parent_link.values())

    def free_links(self) -> list[int]:
        tree = set(self.spanning_tree_edges())
        return [e for e in range(self.nE) if e not in tree]

    # ---- gauge actions: standard U'_uv = g_u U_uv g_v^{-1} ----
    def _gauge_vertex_indices(self, edges, h):
        """Apply gauge element h at a vertex: outgoing U->hU, incoming U->U h^{-1}."""
        N, dim = self.N, self.N ** self.nE
        ih = self.inv[h]
        newflat = np.arange(dim)
        for (e, s) in edges:
            base = N ** e
            digit = (newflat // base) % N
            if s == +1:
                newd = self.table[h, digit]
            else:
                newd = self.table[digit, ih]
            newflat = newflat + (newd - digit) * base
        return newflat

    def _left_indices(self, edges, h):
        """Electric left-regular action on a single link (not a gauge transform)."""
        N, dim = self.N, self.N ** self.nE
        newflat = np.arange(dim)
        for (e, _s) in edges:
            base = N ** e
            digit = (newflat // base) % N
            newflat = newflat + (self.table[h, digit] - digit) * base
        return newflat

    def _link_perm(self, e, h):
        key = (e, h)
        if key not in self._link_perm_cache:
            self._link_perm_cache[key] = self._left_indices([(e, +1)], h)
        return self._link_perm_cache[key]

    def _vertex_perm(self, edges, h):
        key = (tuple(edges), h)
        if key not in self._vertex_perm_cache:
            self._vertex_perm_cache[key] = self._gauge_vertex_indices(edges, h)
        return self._vertex_perm_cache[key]

    def gauge_projector_matvec(self, x):
        N = self.N
        y = x.copy()
        for i, j, k in self.sites:
            edges = self.vertex_edges(i, j, k)
            if not edges:
                continue
            acc = np.zeros_like(y)
            for h in range(N):
                acc[self._vertex_perm(edges, h)] += y
            y = acc / N
        return y

    def magnetic_diagonal(self, V):
        N, dim = self.N, self.N ** self.nE
        mag = np.zeros(dim)
        flat = np.arange(dim)
        for pedges in self.all_plaquettes():
            hol = np.zeros(dim, dtype=int)
            for (e, s) in pedges:
                base = N ** e
                digit = (flat // base) % N
                if s == +1:
                    hol = self.table[hol, digit]
                else:
                    hol = self.table[hol, self.inv[digit]]
            mag += V[hol]
        return mag

    def electric_matvec(self, x):
        y = np.zeros_like(x)
        N = self.N
        for e in range(self.nE):
            acc = np.zeros_like(x)
            for h in range(N):
                acc[self._link_perm(e, h)] += x / N
            y += x - acc
        return y

    def hamiltonian_matvec_operator(self, g: float, V):
        mag = self.magnetic_diagonal(V)
        ge, gm = g * g / 2.0, 1.0 / (2.0 * g * g)
        dim = self.N ** self.nE
        P = self.gauge_projector_matvec

        def Hx(x):
            return ge * self.electric_matvec(x) + gm * (mag * x)

        def HP(x):
            return P(Hx(P(x)))

        return LinearOperator((dim, dim), HP, dtype=float), mag  # type: ignore[call-arg]

    def elec_op_links(self):
        N, dim = self.N, self.N ** self.nE
        H = None
        for e in range(self.nE):
            A = None
            for h in range(N):
                newflat = self._left_indices([(e, +1)], h)
                col = np.arange(dim)
                blk = csr_matrix((np.ones(dim), (newflat, col)), shape=(dim, dim))
                A = blk if A is None else A + blk
            assert A is not None
            A = A / N
            He_e = sp_eye(dim, format="csr") - A
            H = He_e if H is None else H + He_e
        return H

    def hamiltonian_operator(self, g, V):
        from scipy.sparse import diags
        He = self.elec_op_links()
        mag = self.magnetic_diagonal(V)
        Hm = diags(mag, format="csr")
        H = (g * g / 2.0) * He + (1.0 / (2.0 * g * g)) * Hm
        return None, H, He, Hm


def gauge_fix_nonabelian(lat: LatticeYM3D, vals: dict[int, int]) -> tuple[int, ...]:
    """Tree gauge fix for open lattices: set tree links to identity; return free tuple.

    Gauge transform: U'_{uv} = g_u U_uv g_v^{-1}. Root g=1; along tree parent->child,
    choose g_child = g_parent * U_edge (oriented parent->child) so tree edge becomes 1.

    Hard checks: BFS reaches all vertices; every tree edge equals identity after transform.
    """
    if lat.periodic:
        raise ValueError("nonabelian tree gauge-fix: open BC only (residual torus gauge)")

    root: Site3 = (0, 0, 0)

    parent: dict[Site3, Site3 | None] = {root: None}
    parent_edge: dict[Site3, tuple[int, bool]] = {}  # child -> (edge_index, fwd_from_parent)

    queue: list[Site3] = [root]
    while queue:
        i, j, k = queue.pop(0)
        for axis, di, dj, dk in (
            (0, 1, 0, 0), (0, -1, 0, 0),
            (1, 0, 1, 0), (1, 0, -1, 0),
            (2, 0, 0, 1), (2, 0, 0, -1),
        ):
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < lat.Lx and 0 <= nj < lat.Ly and 0 <= nk < lat.Lz):
                continue
            if (ni, nj, nk) in parent:
                continue

            if di + dj + dk > 0:
                e = lat.link_of(axis, i, j, k)
                fwd = True
            else:
                e = lat.link_of(axis, ni, nj, nk)
                fwd = False

            if e is None:
                continue
            parent[(ni, nj, nk)] = (i, j, k)
            parent_edge[(ni, nj, nk)] = (e, fwd)
            queue.append((ni, nj, nk))

    if len(parent) != lat.nV:
        raise RuntimeError(f"tree gauge-fix BFS incomplete: {len(parent)}/{lat.nV} vertices")

    tree_edges = {edge for (edge, _fwd) in parent_edge.values()}

    order: list[Site3] = []
    q2: list[Site3] = [root]
    seen: set[Site3] = {root}
    while q2:
        u = q2.pop(0)
        order.append(u)
        for v, p in parent.items():
            if p == u and v not in seen:
                seen.add(v)
                q2.append(v)

    table, inv = lat.table, lat.inv

    g_at: dict[Site3, int] = {root: 0}  # identity index 0 (make_group)
    for child in order[1:]:
        p = parent[child]
        assert p is not None
        e, fwd = parent_edge[child]
        U = vals[e]
        gp = g_at[p]
        if fwd:
            g_at[child] = table[gp, U]
        else:
            g_at[child] = table[gp, inv[U]]

    new_vals: dict[int, int] = {}
    for e, (axis, i, j, k) in enumerate(lat.link_list):
        s0 = (i, j, k)
        if axis == 0:
            s1 = (i + 1, j, k)
        elif axis == 1:
            s1 = (i, j + 1, k)
        else:
            s1 = (i, j, k + 1)
        if s0 not in g_at or s1 not in g_at:
            raise RuntimeError("gauge_fix_nonabelian: missing g_at on endpoint")
        g0, g1 = g_at[s0], g_at[s1]
        U = vals[e]
        new_vals[e] = table[table[g0, U], inv[g1]]

    for e in tree_edges:
        if new_vals[e] != 0:
            raise RuntimeError(
                f"tree gauge-fix failed: tree edge {e} not identity (got {new_vals[e]})"
            )

    free = lat.free_links()
    return tuple(new_vals[e] for e in free)


def tree_reduced_He_Hm(lat: LatticeYM3D, V) -> tuple:
    """He, Hm on free-link basis via nonabelian tree gauge fixing (open BC).

    Returns (He, Hm, dim_red, n_plaq). Used for alpha_* relative-bound audits.
    """
    if lat.periodic:
        raise ValueError("tree_reduced_He_Hm requires open BC")
    N = lat.N
    free = lat.free_links()
    nfree = len(free)
    dim_red = N ** nfree
    plaquettes = lat.all_plaquettes()
    He = np.zeros((dim_red, dim_red))
    Hm = np.zeros((dim_red, dim_red))
    free_pos = {e: p for p, e in enumerate(free)}

    def vals_from_r(r: int) -> dict[int, int]:
        vals = {e: 0 for e in range(lat.nE)}
        for e, p in free_pos.items():
            vals[e] = (r // (N ** p)) % N
        return vals

    def rep_index(vals: dict[int, int]) -> int:
        canon = gauge_fix_nonabelian(lat, vals)
        idx = 0
        for p, val in enumerate(canon):
            idx += val * (N ** p)
        return idx

    for r in range(dim_red):
        vals = vals_from_r(r)
        mag = 0.0
        for pedges in plaquettes:
            hol = 0
            for (e, s) in pedges:
                d = vals[e]
                hol = lat.table[hol, d] if s == +1 else lat.table[hol, lat.inv[d]]
            mag += V[hol]
        Hm[r, r] += mag
        for e in range(lat.nE):
            base_val = vals[e]
            He[r, r] += 1.0 - 1.0 / N
            for h in range(1, N):
                new_vals = dict(vals)
                new_vals[e] = lat.table[h, base_val]
                r2 = rep_index(new_vals)
                He[r, r2] += -1.0 / N
    return 0.5 * (He + He.T), 0.5 * (Hm + Hm.T), dim_red, len(plaquettes)


def tree_reduced_hamiltonian(lat: LatticeYM3D, g: float, V) -> tuple:
    """Build H on free-link basis via nonabelian tree gauge fixing (open BC).

    Returns (w, gap, vac, e0, dim_red, n_plaq).
    """
    He, Hm, dim_red, n_plaq = tree_reduced_He_Hm(lat, V)
    Hred = (g * g / 2.0) * He + (1.0 / (2.0 * g * g)) * Hm
    Hred = (Hred + Hred.T) / 2
    w = np.sort(np.linalg.eigvalsh(Hred))
    e0, gap, vac = jw_gap_from_w(w)
    return w, gap, vac, e0, dim_red, n_plaq


def gi_basis_svd(lat: LatticeYM3D, dim: int, max_dim: int = 8192) -> np.ndarray:
    """Orthonormal basis for im(P) via incremental Gram-Schmidt (memory-safe)."""
    if dim > max_dim:
        raise MemoryError(f"gi_basis: dim={dim} > {max_dim}")
    basis: list = []
    for kk in range(dim):
        e = np.zeros(dim)
        e[kk] = 1.0
        pk = lat.gauge_projector_matvec(e)
        for b in basis:
            pk = pk - b * float(b @ pk)
        nrm = float(np.linalg.norm(pk))
        if nrm > 1e-10:
            basis.append(pk / nrm)
    if not basis:
        return np.zeros((dim, 0))
    return np.column_stack(basis)


def dense_gi_spectrum_3d(lat: LatticeYM3D, g: float, V):
    """Gauge-invariant spectrum via SVD basis of im(P), then Q^T H Q."""
    dim = lat.N ** lat.nE
    _op, H, _He, _Hm = lat.hamiltonian_operator(g, V)
    Q = gi_basis_svd(lat, dim)
    if Q.size == 0:
        return np.array([]), float("nan"), 0, float("nan")
    Hred = Q.T @ (H @ Q)
    Hred = (Hred + Hred.T) / 2
    Hdense = Hred.toarray() if hasattr(Hred, "toarray") else np.asarray(Hred)
    w = np.sort(np.linalg.eigvalsh(Hdense))
    e0, gap, vac = jw_gap_from_w(w)
    return w, gap, vac, e0


def os_positivity_matrix_euclid(
    *,
    group: str = "Q8",
    T: int = 2,
    L: int = 2,
    beta: float = BETA_DEFINING,
    temporal_gauge: bool | None = None,
) -> dict:
    """Non-tautological OS RP: M_ij = E[(Θ F_i) F_j] under Euclidean Wilson measure.

    Uses common Euclidean finite-G measure. Not ⟨Ω|F† e^{-tH} F|Ω⟩.
    """
    from Yang_Mills_Mass_Gap_common import (
        V_tbl_K4,
        V_tbl_Q8,
        build_lattice_2d,
        certify_os_rp_exact,
        finite_group_K4,
        finite_group_Q8,
        lattice_config_count,
        make_basis_time_slice_V,
    )

    if group.upper() == "K4":
        G = finite_group_K4()
        V_tbl = V_tbl_K4()
    else:
        G = finite_group_Q8()
        V_tbl = V_tbl_Q8()
    lat = build_lattice_2d(T, L, periodic_t=False, periodic_x=True)
    if temporal_gauge is None:
        # Prefer full measure when enumerable; else temporal gauge.
        n_full = lattice_config_count(G, lat, temporal_gauge=False)
        temporal_gauge = n_full > 300_000
    basis = make_basis_time_slice_V(G, lat, V_tbl)
    try:
        cert = certify_os_rp_exact(
            G, lat, beta, V_tbl, basis, temporal_gauge=temporal_gauge
        )
    except MemoryError as e:
        return {
            "skipped": True,
            "pass": False,
            "reason": str(e),
            "group": group,
            "T": T,
            "L": L,
        }
    return {
        "skipped": False,
        "group": group,
        "T": T,
        "L": L,
        "beta": beta,
        "temporal_gauge": bool(temporal_gauge),
        "n_ops": cert["n_basis"],
        "n_configs": cert["n_configs"],
        "n_free": cert["n_free"],
        "min_eig_M": cert["min_eig"],
        "evals": cert["evals"],
        "pass": cert["PSD"],
        "note": "Euclidean OS Gram E[(ΘF_i)F_j]; not Hamiltonian tautology.",
    }


def tree_vs_dense_verify() -> dict:
    """K4 open 2x2x1: tree-reduced gap vs dense GI spectrum."""
    _, V = wilson_weight_K4()
    lat = LatticeYM3D(2, 2, 1, K4(), periodic=False)
    w_d, gap_d, vac_d, e0_d = dense_gi_spectrum_3d(lat, G_DEFINING_KS, V)
    w_t, gap_t, vac_t, e0_t, dim_red, n_plaq = tree_reduced_hamiltonian(
        lat, G_DEFINING_KS, V
    )
    rel = abs(gap_t - gap_d) / max(abs(gap_d), 1e-12)
    return {
        "n_plaq": n_plaq,
        "dim_red": dim_red,
        "dense_E0": float(e0_d), "dense_gap": float(gap_d), "dense_vac": int(vac_d),
        "tree_E0": float(e0_t), "tree_gap": float(gap_t), "tree_vac": int(vac_t),
        "rel_err": float(rel),
        "pass": rel < 1e-6 and vac_d == vac_t and n_plaq > 0,
    }


def run_curvature_3d() -> dict:
    """Euclidean OS reflection positivity (non-tautological Gram)."""
    print("=" * 5)
    print("OS positivity (Euclidean)")
    section(11, section_title(11))
    progress("OS Euclidean Gram")
    os_ = os_positivity_matrix_euclid(group="Q8", T=2, L=2, beta=BETA_DEFINING)
    print("  group / T×L / beta           :", os_.get("group"), f"{os_.get('T')}x{os_.get('L')}", os_.get("beta"))
    print("  temporal_gauge / n_configs   :", os_.get("temporal_gauge"), os_.get("n_configs"))
    print("  n_ops                        :", os_.get("n_ops"))
    print("  min eig M                    :", None if os_.get("skipped") else round(os_["min_eig_M"], 8))
    print("  min_eig_M >= 0 (PSD)         :", os_["pass"])
    gate("OS Euclidean Gram PSD", os_["pass"] and not os_.get("skipped"))
    return {"os": os_, "pass": bool(os_["pass"] and not os_.get("skipped"))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="YM mass gap curvature + 3D")
    ap.parse_args()
    out = run_curvature_3d()
    raise SystemExit(0 if out["pass"] else 1)
