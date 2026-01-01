"""
CGM-KMS Analysis Runner
======================

Goal
----
Compute KMS equilibrium diagnostics on real C. elegans connectome CSV files and
compare against a CGM-motivated reference point:

    A*(CGM) = 1 - δ_BU / m_a ≈ 0.0207

Bridge quantity (dimensionless)
-------------------------------
Given a directed adjacency A with convention A[i,j] = weight from neuron j -> i,
define the KMS critical inverse temperature:

    βc = log r(A)     where r(A) is the spectral radius.

Define a normalized supercriticality margin:

    A_KMS(β) = 1 - βc/β.

Then the CGM-bridge reference point is the β* satisfying A_KMS(β*) = A*:

    β* = βc / (1 - A*).

Important notes (to keep this honest)
-------------------------------------
- A_KMS(β*) = A* is algebraic by construction; it is not evidence.
- The empirical question is whether other KMS observables (similarity breaking,
  adjacency-fit scale, entropy structure) show distinctive behavior near β*.
- The script uses only information available in the data files. It does not
  assume neuron classes (locomotor etc).

What is computed per dataset
----------------------------
1) βc, β*, and A_KMS(β*) (sanity).
2) Mean pairwise similarity between pure KMS columns X(:,j) using the
   Uhlmann/Jozsa-style overlap used in the KMS paper:

       P(j,k,β) = (Σ_i sqrt(x_i^j x_i^k))^2.

   We look for the β where symmetry-breaking rate is largest:
       maximize -d/dβ (mean P).

3) "Adjacency-fit" scale:
   Compare X(β) to the outgoing-normalized adjacency P_out (column-stochastic)
   and report β that minimizes ||X(β) - P_out||_F over the scanned range.
   (This is a proxy for the paper’s statement that at certain β the KMS matrix
   approximates structural connectivity.)

4) Entropy diagnostics for three mixing distributions p:
   - uniform over nodes
   - p proportional to out-degree (column sum of A)
   - p proportional to in-degree (row sum of A)

Data loading
------------
This runner focuses on CSVs. It supports:
- Edge list CSV with header: Source,Target,Weight,Type
- Numeric adjacency matrix CSV (square), possibly with a header row/col of labels

It will scan the local "data/" directory and attempt to load all .csv files.
Files that cannot be interpreted are skipped with a factual reason.

No file outputs, no CLI flags. Prints are intended for inspection.

Refs
----
- Moutuou & Benali (2025): arXiv:2410.18222v2
- Korompilias (2025): Zenodo 10.5281/zenodo.17521384
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


SEPARATOR = "=" * 10
SUBSEP = "-" * 10


def clip_prob(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.clip(x, eps, 1.0)


@dataclass(frozen=True)
class CGM:
    """CGM constants used here only to define A* and related derived values."""
    m_a: float = 1.0 / (2.0 * math.sqrt(2.0 * math.pi))
    delta_BU: float = 0.195342176580

    @property
    def A_star(self) -> float:
        return 1.0 - (self.delta_BU / self.m_a)

    @property
    def closure(self) -> float:
        return 1.0 - self.A_star


def spectral_radius_power(A: np.ndarray, iters: int = 200, tol: float = 1e-10) -> float:
    """
    Estimate spectral radius of a nonnegative matrix using power iteration.

    This is stable for large matrices and avoids full eigendecomposition.
    For nonnegative irreducible A, it converges to Perron root.

    Returns
    -------
    float : estimated spectral radius.
    """
    n = A.shape[0]
    x = np.ones(n, dtype=float) / math.sqrt(n)
    last = 0.0
    for _ in range(iters):
        y = A @ x
        norm = float(np.linalg.norm(y))
        if norm == 0.0:
            return 0.0
        x = y / norm
        # Rayleigh quotient as estimate
        lam = float((x @ (A @ x)) / (x @ x))
        if abs(lam - last) < tol * max(1.0, abs(lam)):
            return max(lam, 0.0)
        last = lam
    return max(last, 0.0)


def spectral_radius(A: np.ndarray) -> float:
    """
    Compute spectral radius r(A).
    Uses eigvals for smaller matrices, power iteration for larger.
    """
    n = A.shape[0]
    if n <= 600:
        eig = np.linalg.eigvals(A)
        return float(np.max(np.abs(eig)))
    return spectral_radius_power(A)


def try_read_csv_header(path: str) -> List[str]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        first = f.readline()
    return [c.strip() for c in first.strip().split(",") if c.strip()]


def load_edgelist_csv(
    path: str,
    allowed_types: Optional[Tuple[str, ...]] = ("chemical",),
    strip_names: bool = True,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float]]:
    """
    Load edge list CSV with header: Source,Target,Weight,Type

    Builds adjacency A with convention:
        A[i,j] = total weight from neuron j -> neuron i

    Parameters
    ----------
    allowed_types:
        If not None, keep only edges whose Type is in allowed_types.
        Pass None to include all types exactly as directed in the file.

    Returns
    -------
    A, index_map, stats
    """
    edges: List[Tuple[str, str, float]] = []
    neurons: set[str] = set()
    type_counts: Dict[str, int] = {}
    total_weight_by_type: Dict[str, float] = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"Source", "Target", "Weight", "Type"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"missing required columns; got {reader.fieldnames}")

        for row in reader:
            src = row["Source"]
            tgt = row["Target"]
            typ = row["Type"]
            w = row["Weight"]

            if strip_names:
                src = src.strip()
                tgt = tgt.strip()
                typ = typ.strip()

            if allowed_types is not None and typ not in allowed_types:
                continue

            weight = float(str(w).strip())
            if weight < 0:
                continue

            neurons.add(src)
            neurons.add(tgt)
            edges.append((src, tgt, weight))

            type_counts[typ] = type_counts.get(typ, 0) + 1
            total_weight_by_type[typ] = total_weight_by_type.get(typ, 0.0) + weight

    names = sorted(neurons)
    idx = {name: i for i, name in enumerate(names)}
    n = len(names)
    A = np.zeros((n, n), dtype=float)

    for src, tgt, weight in edges:
        j = idx[src]  # source column
        i = idx[tgt]  # target row
        A[i, j] += weight

    stats = {
        "nodes": float(n),
        "edges_rows_used": float(len(edges)),
        "nnz": float(np.count_nonzero(A)),
        "total_weight": float(np.sum(A)),
    }
    # embed type summaries with stable keys
    for t, c in sorted(type_counts.items()):
        stats[f"type_rows_{t}"] = float(c)
    for t, tw in sorted(total_weight_by_type.items()):
        stats[f"type_weight_{t}"] = float(tw)

    return A, idx, stats


def load_numeric_matrix_csv(path: str) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float]]:
    """
    Attempt to load a CSV as a numeric adjacency matrix.
    Handles:
    - pure numeric CSV (no labels)
    - labeled CSV with header row/col (first row/col non-numeric)

    Returns
    -------
    A, index_map, stats
    """
    # Try reading with genfromtxt; if labels exist, they become nan
    raw = np.genfromtxt(path, delimiter=",", dtype=float, encoding="utf-8")
    if raw.ndim != 2:
        raise ValueError("not a 2D matrix")

    # Case 1: fully numeric square
    if raw.shape[0] == raw.shape[1] and np.isfinite(raw).all():
        n = raw.shape[0]
        idx = {str(i): i for i in range(n)}
        A = raw.astype(float)
        return A, idx, {
            "nodes": float(n),
            "edges_rows_used": float("nan"),
            "nnz": float(np.count_nonzero(A)),
            "total_weight": float(np.nansum(A)),
        }

    # Case 2: likely labeled; try dropping first row/col if that yields a finite square
    # Common layout: first row labels, first col labels
    sub = raw[1:, 1:]
    if sub.ndim == 2 and sub.shape[0] == sub.shape[1] and np.isfinite(sub).all():
        n = sub.shape[0]
        # Try get labels from first row/col as strings by rereading header line
        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        # labels often in first row, starting at col 1
        col_labels = [c.strip() for c in rows[0][1:1 + n]]
        # If labels are empty, fall back to indices
        if all(lbl == "" for lbl in col_labels):
            col_labels = [str(i) for i in range(n)]
        idx = {name: i for i, name in enumerate(col_labels)}
        A = sub.astype(float)
        return A, idx, {
            "nodes": float(n),
            "edges_rows_used": float("nan"),
            "nnz": float(np.count_nonzero(A)),
            "total_weight": float(np.sum(A)),
        }

    raise ValueError("matrix could not be interpreted as a finite square adjacency")


class KMSSystem:
    """
    Computes KMS matrices and derived diagnostics for a fixed adjacency A.

    Convention:
        A[i,j] = weight from node j -> node i  (columns are sources).
    """

    def __init__(self, A: np.ndarray, name: str):
        self.A = np.array(A, dtype=float)
        self.N = self.A.shape[0]
        self.name = name

        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square")
        if np.any(self.A < 0):
            raise ValueError("A must be nonnegative")

        self.r = spectral_radius(self.A)
        self.beta_c = float(math.log(self.r)) if self.r > 1.0 else 0.0

        self.out_strength = np.sum(self.A, axis=0)  # column sums
        self.in_strength = np.sum(self.A, axis=1)   # row sums

    def kms_matrix(self, beta: float) -> np.ndarray:
        """
        X(β) = (I - e^{-β}A)^{-1} column-normalized to be column-stochastic.
        """
        if beta <= self.beta_c:
            raise ValueError("β must be > βc")
        s = float(math.exp(-beta))
        M = np.eye(self.N) - s * self.A
        # Solve M R = I for R (inverse)
        R = np.linalg.solve(M, np.eye(self.N))
        Z = np.sum(R, axis=0)  # column sums
        X = R / Z[np.newaxis, :]
        # numerical cleanup
        X = np.clip(X, 0.0, None)
        X /= np.sum(X, axis=0, keepdims=True)
        return X

    def outgoing_normalized_adjacency(self) -> np.ndarray:
        """
        P_out[i,j] = A[i,j] / sum_i A[i,j] (if column sum > 0), else identity.
        """
        P = np.zeros_like(self.A)
        for j in range(self.N):
            col = self.out_strength[j]
            if col > 0:
                P[:, j] = self.A[:, j] / col
            else:
                P[j, j] = 1.0
        return P

    def mixing_distributions(self) -> Dict[str, np.ndarray]:
        """
        Return canonical p vectors used as mixing distributions.
        """
        p_uniform = np.ones(self.N) / self.N

        out = self.out_strength.copy()
        if out.sum() > 0:
            p_out = out / out.sum()
        else:
            p_out = p_uniform

        inn = self.in_strength.copy()
        if inn.sum() > 0:
            p_in = inn / inn.sum()
        else:
            p_in = p_uniform

        return {"uniform": p_uniform, "out_strength": p_out, "in_strength": p_in}


def mean_pair_similarity(
    X: np.ndarray,
    max_pairs: Optional[int] = None,
    seed: int = 42,
) -> float:
    """
    Mean over pairs j<k of:
        P(j,k) = (Σ_i sqrt(X[i,j] X[i,k]))^2

    If max_pairs is None, uses all pairs for N<=220 else uses sampling.
    If max_pairs is provided, uses sampling of that many pairs.

    Sampling is vectorized and does not materialize all pairs.
    """
    n = X.shape[1]
    if n < 2:
        return 1.0

    # default strategy
    if max_pairs is None:
        if n <= 220:
            S = np.sqrt(clip_prob(X))
            G = S.T @ S  # Gram matrix
            # take upper triangle without diag
            iu = np.triu_indices(n, k=1)
            vals = (G[iu] ** 2).astype(float)
            return float(np.mean(vals)) if vals.size else 1.0
        max_pairs = 8000

    total_pairs = n * (n - 1) // 2
    m = int(min(max_pairs, total_pairs))
    rng = np.random.default_rng(seed)

    # sample indices j,k uniformly with j != k
    j = rng.integers(0, n, size=m)
    k = rng.integers(0, n - 1, size=m)
    k = k + (k >= j)  # ensure k != j

    # enforce ordering j<k for interpretation consistency
    j2 = np.minimum(j, k)
    k2 = np.maximum(j, k)

    Xj = X[:, j2]
    Xk = X[:, k2]
    bc = np.sum(np.sqrt(clip_prob(Xj * Xk)), axis=0)  # Bhattacharyya coefficient
    return float(np.mean(bc ** 2))


def entropy_of_mixed(X: np.ndarray, p: np.ndarray) -> float:
    """
    Mixed-state entropy:
        y = X p,  S = -Σ_i y_i log y_i
    """
    y = X @ p
    y = clip_prob(y)
    return float(-np.sum(y * np.log(y)))


def smooth_1d(x: np.ndarray, w: int = 5) -> np.ndarray:
    """
    Simple moving average smoothing. w must be odd.
    """
    if w <= 1 or x.size < w:
        return x
    if w % 2 == 0:
        w += 1
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(xp, kernel, mode="valid")


def build_beta_grid(beta_c: float, include_points: List[float]) -> np.ndarray:
    """
    Build a β grid covering [~βc, 3βc] plus specific included points.
    Uses a two-stage grid: coarse + refined around βc and included points.
    """
    # Coarse
    lo = beta_c * 1.0005
    hi = beta_c * 3.0
    coarse = np.linspace(lo, hi, 40)

    # Extra resolution near βc (where things can change fast)
    near = np.linspace(beta_c * 1.0005, beta_c * 1.10, 25)

    betas = np.unique(np.concatenate([coarse, near, np.array(include_points, dtype=float)]))
    betas = betas[betas > beta_c]
    betas.sort()
    return betas


def analyze_one(A: np.ndarray, name: str, cgm: CGM, stats: Optional[Dict[str, float]] = None) -> None:
    """
    Run the CGM-KMS diagnostics on one adjacency.
    Prints factual values only.
    """
    kms = KMSSystem(A, name)
    n = kms.N
    nnz = int(np.count_nonzero(kms.A))
    total_w = float(np.sum(kms.A))

    print(SEPARATOR)
    print(name)
    print(SUBSEP)
    print(f"N={n} nnz={nnz} total_weight={total_w:.0f} r(A)={kms.r:.6f} beta_c={kms.beta_c:.6f}")

    if stats is not None:
        # Print a few loader stats if present
        extra = []
        for k in ["edges_rows_used", "type_rows_chemical", "type_rows_electrical", "type_weight_chemical", "type_weight_electrical"]:
            if k in stats and not math.isnan(stats[k]):
                extra.append(f"{k}={stats[k]:.0f}")
        if extra:
            print("loader_stats:", ", ".join(extra))

    if kms.beta_c <= 0:
        print("beta_c<=0: KMS β>βc regime not available for this matrix.")
        return

    A_star = cgm.A_star
    beta_star = kms.beta_c / (1.0 - A_star)
    margin_at_star = 1.0 - kms.beta_c / beta_star  # equals A_star algebraically

    # include points: β*, βc*(2.5), and βc*(3.0) boundary already in grid
    include_points = [beta_star, kms.beta_c * 2.5]
    betas = build_beta_grid(kms.beta_c, include_points)

    # Precompute adjacency-normalized target
    P_out = kms.outgoing_normalized_adjacency()

    # Choose similarity computation mode
    # Full exact for smaller N; sampled for larger N.
    max_pairs = None if n <= 220 else 12000

    ps = kms.mixing_distributions()

    sims = np.zeros_like(betas)
    fit = np.zeros_like(betas)
    ent_uniform = np.zeros_like(betas)
    ent_out = np.zeros_like(betas)
    ent_in = np.zeros_like(betas)

    # Main scan (each β requires one solve)
    for i, beta in enumerate(betas):
        X = kms.kms_matrix(float(beta))

        sims[i] = mean_pair_similarity(X, max_pairs=max_pairs, seed=42)
        fit[i] = float(np.linalg.norm(X - P_out, ord="fro"))

        ent_uniform[i] = entropy_of_mixed(X, ps["uniform"])
        ent_out[i] = entropy_of_mixed(X, ps["out_strength"])
        ent_in[i] = entropy_of_mixed(X, ps["in_strength"])

    # Smooth similarity if using sampling (reduces gradient noise)
    sims_s = smooth_1d(sims, w=7) if max_pairs is not None else sims
    ds = np.gradient(sims_s, betas)
    idx_break = int(np.argmax(-ds))
    beta_break = float(betas[idx_break])

    # Structural-fit beta (min Frobenius distance to P_out)
    idx_fit = int(np.argmin(fit))
    beta_fit = float(betas[idx_fit])

    # Entropy peaks and steepest changes (diagnostics)
    def peak_and_slope(ent: np.ndarray) -> Tuple[float, float]:
        i_peak = int(np.argmax(ent))
        d_ent = np.gradient(ent, betas)
        i_slope = int(np.argmax(np.abs(d_ent)))
        return float(betas[i_peak]), float(betas[i_slope])

    beta_ent_peak_u, beta_ent_slope_u = peak_and_slope(ent_uniform)
    beta_ent_peak_out, beta_ent_slope_out = peak_and_slope(ent_out)
    beta_ent_peak_in, beta_ent_slope_in = peak_and_slope(ent_in)

    # Interpolate similarity at β* (nearest grid point)
    idx_star = int(np.argmin(np.abs(betas - beta_star)))
    sim_at_star = float(sims[idx_star])

    # Print summary
    print(f"beta_star={beta_star:.6f} A_KMS(beta_star)={margin_at_star:.6f} A*={A_star:.6f}")
    print(f"beta_break={beta_break:.6f} rel_err_vs_beta_star={(abs(beta_break-beta_star)/beta_star)*100:.2f}% sim_at_beta_star≈{sim_at_star:.6f}")
    print(f"beta_fit={beta_fit:.6f} beta_fit/beta_c={(beta_fit/kms.beta_c):.3f} fit_norm={float(fit[idx_fit]):.4f}")
    print(f"beta_target_2.5bc={(kms.beta_c*2.5):.6f}")

    print(f"entropy_uniform_at_beta_star={float(ent_uniform[idx_star]):.4f} logN={math.log(n):.4f}")
    print(f"entropy_peaks_beta: uniform={beta_ent_peak_u:.6f} out_strength={beta_ent_peak_out:.6f} in_strength={beta_ent_peak_in:.6f}")
    print(f"entropy_max_slope_beta: uniform={beta_ent_slope_u:.6f} out_strength={beta_ent_slope_out:.6f} in_strength={beta_ent_slope_in:.6f}")


def load_all_csvs(data_dir: str = "data") -> List[Tuple[str, np.ndarray, Optional[Dict[str, float]]]]:
    """
    Scan data_dir for .csv files and attempt to load each as either:
    - edge list with Source/Target/Weight/Type
    - adjacency matrix

    Returns a list of (label, A, stats).
    """
    outputs: List[Tuple[str, np.ndarray, Optional[Dict[str, float]]]] = []
    if not os.path.isdir(data_dir):
        print(f"{SEPARATOR}\ndata_dir_not_found: {data_dir}\n{SEPARATOR}")
        return outputs

    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".csv")])
    if not files:
        print(f"{SEPARATOR}\nno_csv_files_found_in: {data_dir}\n{SEPARATOR}")
        return outputs

    for fn in files:
        path = os.path.join(data_dir, fn)
        header = try_read_csv_header(path)

        # Try edge list first if it has the expected fields
        try:
            if {"Source", "Target", "Weight", "Type"}.issubset(set(header)):
                # Build two variants: chemical-only and all-types (directed as given)
                A_chem, _, stats_chem = load_edgelist_csv(path, allowed_types=("chemical",))
                outputs.append((f"{fn} [chemical]", A_chem, stats_chem))

                A_all, _, stats_all = load_edgelist_csv(path, allowed_types=None)
                outputs.append((f"{fn} [all_types]", A_all, stats_all))
                continue
        except Exception:
            # fall through to matrix attempt
            pass

        # Try matrix load
        try:
            A_mat, _, stats = load_numeric_matrix_csv(path)
            outputs.append((f"{fn} [matrix]", A_mat, stats))
        except Exception:
            # Skip silently? We'll print a factual skip line.
            print(f"skip_csv: {fn} (unrecognized format)")
            continue

    return outputs


def main() -> None:
    cgm = CGM()
    print(SEPARATOR)
    print("CGM-KMS RUNNER")
    print(SEPARATOR)
    print(f"A*={cgm.A_star:.6f} ({cgm.A_star*100:.2f}%) closure={cgm.closure:.6f}")
    print()

    datasets = load_all_csvs("data")
    if not datasets:
        print("no_datasets_loaded")
        return

    # Run analysis on all loaded datasets
    for label, A, stats in datasets:
        try:
            analyze_one(A, label, cgm, stats=stats)
        except Exception as e:
            print(SEPARATOR)
            print(label)
            print(SUBSEP)
            print(f"analysis_error: {type(e).__name__}: {e}")

    print(SEPARATOR)
    print("DONE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()