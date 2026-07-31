#!/usr/bin/env python3
"""
hqvm_cgm_allometry_2.py

CGM allometry data engine: log-log OLS/RMA/SMA fits, dual-null + μ-family
compare, bootstrap CI, AIC, null shuffle, M0 intercept, activity-regime,
info conjugacy, and sum-rule audits against frozen catalogs.

No printing. Invoked by hqvm_cgm_allometry_run.py.
Companion: hqvm_cgm_allometry_1.py, hqvm_cgm_allometry_notes.md.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from hqvm_cgm_allometry_1 import (
    A_BULK,
    A_EGRESS,
    A_SURFACE,
    A_TIME,
    EXACT_TOL,
    NEAR_TOL,
    TICKS_PER_OCTAVE,
    a_from_mu,
    channel_rules,
    classify_vs_isometry,
    in_dual_band_a,
    kleiber_absolute_intercept,
    mu_from_a,
)

LOCO_TOL = 5e-2
BOOT_DEFAULT = 400
NULL_PERM_DEFAULT = 200


@dataclass(frozen=True)
class FitResult:
    name: str
    n: int
    a_OLS: float
    a_RMA: float
    a_SMA: float
    a_primary: float
    estimator: str
    log_k_RMA: float
    r: float
    mu_primary: float
    mu_in_band: bool
    resid_2_3: float
    resid_3_4: float
    nearest_null: str
    status: str
    a_lo: float
    a_hi: float
    a_med: float
    a_iso: float
    vs_isometry: str


# Organism traits: directed y|x with x=mass → OLS primary.
# City conjugacy / uncertain dual axes → RMA.
_RMA_TRAIT_CLASSES = frozenset(
    {
        "city_infrastructure",
        "city_socioeconomic",
        "company_scale",
    }
)


def primary_estimator_for(trait_class: str) -> str:
    return "RMA" if trait_class in _RMA_TRAIT_CLASSES else "OLS"


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    trait_class: str
    xs: Tuple[float, ...]
    ys: Tuple[float, ...]
    source: str


@dataclass(frozen=True)
class ModelCompareRow:
    series: str
    model: str
    a: float
    sse: float
    n_params: int
    aic: float
    delta_aic: float


@dataclass(frozen=True)
class NullAuditRow:
    series: str
    n_perm: int
    real_status: str
    real_nearest: str
    frac_near_or_exact: float
    frac_same_nearest: float
    pass_null: bool


def _corr(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx <= 0 or dy <= 0:
        return 0.0
    return num / (dx * dy)


def fit_loglog(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float, float, float]:
    """Return (a_OLS, a_RMA, a_SMA, log_k_RMA, r) on log10 axes."""
    if len(xs) != len(ys) or len(xs) < 3:
        raise ValueError("need >=3 paired positive points")
    lx = [math.log10(v) for v in xs]
    ly = [math.log10(v) for v in ys]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    syy = sum((b - my) ** 2 for b in ly)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    if sxx <= 0 or syy <= 0:
        raise ValueError("degenerate log-log cloud")
    a_ols = sxy / sxx
    r = sxy / math.sqrt(sxx * syy)
    sign = 1.0 if r >= 0 else -1.0
    a_rma = sign * math.sqrt(syy / sxx)
    a_yx = a_ols
    a_xy = sxy / syy
    a_sma = sign * math.sqrt(abs(a_yx / a_xy)) if abs(a_xy) > 0 else a_rma
    log_k = my - a_rma * mx
    return a_ols, a_rma, a_sma, log_k, r


def classify_exponent(a: float, trait_class: str = "") -> Tuple[str, float, float, str]:
    """Classify vs dual nulls, μ-family interior, time ±1/4, city channels, or density."""
    r23 = abs(a - A_SURFACE)
    r34 = abs(a - A_BULK)
    mu = mu_from_a(a)
    if trait_class in ("metabolic_rate_BU", "metabolic_rate"):
        # Resting/basal metabolic rate: null is the μ-band [2/3, 3/4], not the μ=1 endpoint.
        if A_SURFACE <= a <= A_BULK:
            return f"mu_band|mu={mu:.4f}", r23, r34, "EXACT"
        return "outside_mu_band", r23, r34, _status_from_resid(min(r23, r34))
    if trait_class in ("metabolic_rate_mixed",):
        # Aggregated basal/field/maximal across taxa: diagnostic, not Tier A BMR null.
        if A_SURFACE <= a <= A_BULK:
            return f"mixed_regime|mu={mu:.4f}", r23, r34, "SCAN"
        return "mixed_regime_outside_band", r23, r34, "SCAN"
    if trait_class in ("heart_rate", "specific_metabolic") or (
        trait_class == "" and a < 0 and abs(a + 0.25) <= abs(a + A_BULK)
    ):
        target = -0.25
        resid = abs(a - target)
        return "-1/4", r23, r34, _status_from_resid(resid)
    if trait_class in ("population_density",):
        # Primary: ecosystem conservation dens∝1/M. Secondary Damuth −3/4 reported by resid.
        resid_m1 = abs(a + 1.0)
        resid_m34 = abs(a + A_BULK)
        if resid_m1 <= resid_m34:
            return "-1", r23, r34, _status_from_resid(resid_m1)
        return "-3/4", r23, r34, _status_from_resid(resid_m34)
    if trait_class in ("home_range",):
        # Dual nulls: metabolic A∝B → 3/4; conservation A∝M → 1. Forbid city 7/6.
        resid_met = abs(a - A_BULK)
        resid_cons = abs(a - 1.0)
        if resid_cons <= resid_met:
            return "1_cons", r23, r34, _status_from_resid(resid_cons)
        return "3/4_met", r23, r34, _status_from_resid(resid_met)
    if trait_class in ("city_infrastructure", "company_scale"):
        target = 1.0 - 1.0 / 6.0
        return "5/6", r23, r34, _status_from_resid(abs(a - target))
    if trait_class in ("city_socioeconomic",):
        target = 2.0 - (1.0 - 1.0 / 6.0)
        return "7/6", r23, r34, _status_from_resid(abs(a - target))
    if trait_class in ("gestation", "weaning", "development_time"):
        resid = abs(a - A_EGRESS)
        return "3/16", r23, r34, _status_from_resid(resid)
    if trait_class in ("lifespan",):
        if A_EGRESS <= a <= A_TIME:
            return "longevity_composite|[3/16,1/4]", r23, r34, "EXACT"
        if a < A_EGRESS:
            # Max-longevity catalogs mix adult maintenance with early mortality.
            return "longevity_development_failure", r23, r34, "SCAN"
        return "longevity_above_maintenance", r23, r34, "MISS"
    if 0.2 <= a <= 0.3 and r23 > 0.05 and trait_class == "":
        if abs(a - 0.25) < min(r23, r34):
            return "+1/4", r23, r34, _status_from_resid(abs(a - 0.25))

    if r23 <= EXACT_TOL:
        return "2/3|mu=0", r23, r34, "EXACT"
    if r34 <= EXACT_TOL:
        return "3/4|mu=1", r23, r34, "EXACT"
    if A_SURFACE - NEAR_TOL <= a <= A_BULK + NEAR_TOL:
        resid_mid = abs(a - a_from_mu(0.5))
        nearest = f"mu_family|mu={mu:.4f}"
        status = "EXACT" if resid_mid <= EXACT_TOL or min(r23, r34) <= EXACT_TOL else (
            "NEAR" if min(r23, r34, resid_mid) <= NEAR_TOL else "SCAN"
        )
        return nearest, r23, r34, status
    nearest = "2/3" if r23 <= r34 else "3/4"
    return nearest, r23, r34, _status_from_resid(min(r23, r34))


def _status_from_resid(resid: float) -> str:
    if resid <= EXACT_TOL:
        return "EXACT"
    if resid <= NEAR_TOL:
        return "NEAR"
    if resid <= LOCO_TOL:
        return "SCAN"
    return "MISS"


def bootstrap_a_rma(
    xs: Sequence[float],
    ys: Sequence[float],
    n_boot: int = BOOT_DEFAULT,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (median, 2.5%, 97.5%) of resampled a_RMA."""
    if n_boot <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(xs)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [xs[i] for i in idx]
        by = [ys[i] for i in idx]
        try:
            _, a_rma, _, _, _ = fit_loglog(bx, by)
        except ValueError:
            continue
        vals.append(a_rma)
    if len(vals) < 10:
        return float("nan"), float("nan"), float("nan")
    vals.sort()
    m = len(vals)
    med = vals[m // 2]
    lo = vals[int(0.025 * (m - 1))]
    hi = vals[int(0.975 * (m - 1))]
    return med, lo, hi


def bootstrap_a_ols(
    xs: Sequence[float],
    ys: Sequence[float],
    n_boot: int = BOOT_DEFAULT,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (median, 2.5%, 97.5%) of resampled a_OLS."""
    if n_boot <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(xs)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [xs[i] for i in idx]
        by = [ys[i] for i in idx]
        try:
            a_ols, _, _, _, _ = fit_loglog(bx, by)
        except ValueError:
            continue
        vals.append(a_ols)
    if len(vals) < 10:
        return float("nan"), float("nan"), float("nan")
    vals.sort()
    m = len(vals)
    med = vals[m // 2]
    lo = vals[int(0.025 * (m - 1))]
    hi = vals[int(0.975 * (m - 1))]
    return med, lo, hi


def _sse_fixed_a(lx: Sequence[float], ly: Sequence[float], a: float) -> float:
    n = len(lx)
    b = sum(y - a * x for x, y in zip(lx, ly)) / n
    return sum((y - (a * x + b)) ** 2 for x, y in zip(lx, ly))


def _sse_free_a(lx: Sequence[float], ly: Sequence[float]) -> Tuple[float, float]:
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    if sxx <= 0:
        raise ValueError("degenerate")
    a = sxy / sxx
    b = my - a * mx
    sse = sum((y - (a * x + b)) ** 2 for x, y in zip(lx, ly))
    return a, sse


def _sse_mu_family(lx: Sequence[float], ly: Sequence[float]) -> Tuple[float, float]:
    """Constrained OLS: a in [2/3, 3/4] (μ in [0,1])."""
    a_free, sse_free = _sse_free_a(lx, ly)
    if in_dual_band_a(a_free, tol=0.0):
        return a_free, sse_free
    a_lo, sse_lo = A_SURFACE, _sse_fixed_a(lx, ly, A_SURFACE)
    a_hi, sse_hi = A_BULK, _sse_fixed_a(lx, ly, A_BULK)
    if sse_lo <= sse_hi:
        return a_lo, sse_lo
    return a_hi, sse_hi


def model_comparison(spec: SeriesSpec) -> List[ModelCompareRow]:
    """AIC under log10 residual SSE (Gaussian noise in log space)."""
    lx = [math.log10(v) for v in spec.xs]
    ly = [math.log10(v) for v in spec.ys]
    n = len(lx)
    rows_raw: List[Tuple[str, float, float, int]] = []
    rows_raw.append(("fixed_2_3", A_SURFACE, _sse_fixed_a(lx, ly, A_SURFACE), 1))
    rows_raw.append(("fixed_3_4", A_BULK, _sse_fixed_a(lx, ly, A_BULK), 1))
    a_free, sse_free = _sse_free_a(lx, ly)
    rows_raw.append(("free_a", a_free, sse_free, 2))
    a_mu, sse_mu = _sse_mu_family(lx, ly)
    rows_raw.append(("mu_family", a_mu, sse_mu, 2))

    scored: List[Tuple[str, float, float, int, float]] = []
    for model, a, sse, p in rows_raw:
        mse = max(sse / n, 1e-300)
        aic = n * math.log(mse) + 2 * p
        scored.append((model, a, sse, p, aic))
    aic_min = min(s[4] for s in scored)
    return [
        ModelCompareRow(
            series=spec.name,
            model=model,
            a=a,
            sse=sse,
            n_params=p,
            aic=aic,
            delta_aic=aic - aic_min,
        )
        for model, a, sse, p, aic in scored
    ]


def _isometric_null_for_trait(trait_class: str) -> float:
    """Default isometric comparison class for catalog traits."""
    if trait_class in ("metabolic_rate_BU", "metabolic_rate", "metabolic_rate_mixed", "brain_size"):
        return A_SURFACE  # surface isometry for metabolic debate
    if trait_class in (
        "lifespan",
        "circulation_time",
        "heart_rate",
        "specific_metabolic",
        "gestation",
        "weaning",
        "population_density",
    ):
        return 0.0  # size-independent null; West family is allometric
    if trait_class in ("length", "aorta_radius"):
        return 1.0 / 3.0
    if trait_class in ("exchange_surface", "surface_exchange"):
        return A_SURFACE
    if trait_class in ("blood_volume", "volume_mass", "home_range"):
        return 1.0
    if trait_class in ("city_infrastructure", "company_scale"):
        return 1.0  # per-capita linear null; sublinear = economy of scale
    if trait_class in ("city_socioeconomic",):
        return 1.0  # linear null; superlinear = increasing returns
    return A_SURFACE


def analyze_series(
    spec: SeriesSpec,
    *,
    n_boot: int = BOOT_DEFAULT,
    seed: int = 0,
) -> FitResult:
    a_ols, a_rma, a_sma, log_k, r = fit_loglog(spec.xs, spec.ys)
    estimator = primary_estimator_for(spec.trait_class)
    a_pri = a_ols if estimator == "OLS" else a_rma
    nearest, r23, r34, status = classify_exponent(a_pri, spec.trait_class)
    rules = {rule.trait_class: rule for rule in channel_rules()}
    rule = rules.get(spec.trait_class)
    # μ-band metabolic, mixed-regime, and lifespan use classify_exponent status directly.
    _skip_point_rule = spec.trait_class in (
        "metabolic_rate_BU",
        "metabolic_rate",
        "metabolic_rate_mixed",
        "lifespan",
    )
    if not _skip_point_rule and rule is not None:
        rule_resid = abs(a_pri - rule.a_pred)
        status = _status_from_resid(rule_resid)
        if rule_resid <= min(r23, r34) + 1e-15:
            nearest = f"rule:{spec.trait_class}"
    if estimator == "OLS":
        med, lo, hi = bootstrap_a_ols(spec.xs, spec.ys, n_boot=n_boot, seed=seed)
    else:
        med, lo, hi = bootstrap_a_rma(spec.xs, spec.ys, n_boot=n_boot, seed=seed)
    mu = mu_from_a(a_pri)
    a_iso = _isometric_null_for_trait(spec.trait_class)
    return FitResult(
        name=spec.name,
        n=len(spec.xs),
        a_OLS=a_ols,
        a_RMA=a_rma,
        a_SMA=a_sma,
        a_primary=a_pri,
        estimator=estimator,
        log_k_RMA=log_k,
        r=r,
        mu_primary=mu,
        mu_in_band=in_dual_band_a(a_pri),
        resid_2_3=r23,
        resid_3_4=r34,
        nearest_null=nearest,
        status=status,
        a_lo=lo,
        a_hi=hi,
        a_med=med,
        a_iso=a_iso,
        vs_isometry=classify_vs_isometry(a_pri, a_iso),
    )


def null_shuffle_audit(
    spec: SeriesSpec,
    *,
    n_perm: int = NULL_PERM_DEFAULT,
    seed: int = 1,
) -> NullAuditRow:
    """Y-shuffle null using OLS exponents (RMA |a| is σy/σx-invariant under shuffle)."""
    a_ols, _a_rma, _, _, _ = fit_loglog(spec.xs, spec.ys)
    nearest, r23, r34, status = classify_exponent(a_ols, spec.trait_class)
    rules = {rule.trait_class: rule for rule in channel_rules()}
    rule = rules.get(spec.trait_class)
    _skip_point_rule = spec.trait_class in (
        "metabolic_rate_BU",
        "metabolic_rate",
        "metabolic_rate_mixed",
        "lifespan",
    )
    if not _skip_point_rule and rule is not None:
        rule_resid = abs(a_ols - rule.a_pred)
        status = _status_from_resid(rule_resid)
        if rule_resid <= min(r23, r34) + 1e-15:
            nearest = f"rule:{spec.trait_class}"

    rng = random.Random(seed)
    ys = list(spec.ys)
    near_hits = 0
    same_hits = 0
    for _ in range(n_perm):
        shuffled = ys[:]
        rng.shuffle(shuffled)
        try:
            a_s, _, _, _, _ = fit_loglog(spec.xs, shuffled)
        except ValueError:
            continue
        nearest_s, r23_s, r34_s, status_s = classify_exponent(a_s, spec.trait_class)
        if not _skip_point_rule and rule is not None:
            rr = abs(a_s - rule.a_pred)
            status_s = _status_from_resid(rr)
            if rr <= min(r23_s, r34_s) + 1e-15:
                nearest_s = f"rule:{spec.trait_class}"
        if status_s in ("EXACT", "NEAR"):
            near_hits += 1
        if nearest_s == nearest:
            same_hits += 1
    frac_near = near_hits / n_perm if n_perm else float("nan")
    frac_same = same_hits / n_perm if n_perm else float("nan")
    pass_null = frac_near < 0.10
    return NullAuditRow(
        series=spec.name,
        n_perm=n_perm,
        real_status=status,
        real_nearest=nearest,
        frac_near_or_exact=frac_near,
        frac_same_nearest=frac_same,
        pass_null=pass_null,
    )


def synthetic_power_series(
    name: str,
    trait_class: str,
    a_true: float,
    k: float = 1.0,
    n: int = 40,
    m0: float = 0.01,
    m1: float = 1000.0,
    noise: float = 0.0,
    seed: int = 0,
) -> SeriesSpec:
    rng = random.Random(seed)
    xs = []
    ys = []
    for i in range(n):
        t = i / (n - 1)
        m = m0 * (m1 / m0) ** t
        y = k * (m**a_true)
        if noise > 0:
            y *= 10 ** (rng.gauss(0.0, noise))
        xs.append(m)
        ys.append(y)
    return SeriesSpec(name=name, trait_class=trait_class, xs=tuple(xs), ys=tuple(ys), source="synthetic")


_REPO = Path(__file__).resolve().parents[1]
ALLOMETRY_CATALOG = _REPO / "data" / "catalogs" / "allometry"


def default_synthetic_suite() -> List[SeriesSpec]:
    """Self-check suite: recovers 2/3, 3/4, μ=1/2, and time exponents."""
    return [
        synthetic_power_series("synth_surface_2_3", "exchange_surface", A_SURFACE, seed=1),
        synthetic_power_series("synth_bulk_3_4", "metabolic_rate_BU", A_BULK, seed=2),
        synthetic_power_series("synth_mu_half", "mixed_physiology", a_from_mu(0.5), seed=3),
        synthetic_power_series("synth_time_m14", "heart_rate", -0.25, k=200.0, seed=4),
        synthetic_power_series("synth_life_p14", "lifespan", 0.25, k=10.0, seed=5),
        synthetic_power_series("synth_noisy_3_4", "metabolic_rate_BU", A_BULK, noise=0.02, seed=6),
    ]


def default_catalog_suite() -> List[SeriesSpec]:
    """Real catalogs under data/catalogs/allometry/ (run ingest if missing)."""
    suite: List[SeriesSpec] = []

    def maybe(
        path: Path,
        name: str,
        trait_class: str,
        x_col: str,
        y_col: str,
        filters: Optional[Dict[str, str]] = None,
    ) -> None:
        if not path.exists():
            return
        suite.append(
            load_csv_series(
                path,
                name=name,
                trait_class=trait_class,
                x_col=x_col,
                y_col=y_col,
                filters=filters,
            )
        )

    maybe(
        ALLOMETRY_CATALOG / "pantheria_bmr.csv",
        "pantheria_bmr",
        "metabolic_rate_BU",
        "mass_kg",
        "bmr_W",
    )
    maybe(
        ALLOMETRY_CATALOG / "pantheria_bmr.csv",
        "pantheria_specific_bmr",
        "specific_metabolic",
        "mass_kg",
        "specific_bmr_W_kg",
    )
    maybe(
        ALLOMETRY_CATALOG / "pantheria_longevity.csv",
        "pantheria_longevity",
        "lifespan",
        "mass_kg",
        "max_longevity_years",
    )
    maybe(
        ALLOMETRY_CATALOG / "pantheria_gestation.csv",
        "pantheria_gestation",
        "gestation",
        "mass_kg",
        "gestation_days",
    )
    maybe(
        ALLOMETRY_CATALOG / "pantheria_weaning.csv",
        "pantheria_weaning",
        "weaning",
        "mass_kg",
        "weaning_days",
    )
    maybe(
        ALLOMETRY_CATALOG / "pantheria_pop_density.csv",
        "pantheria_pop_density",
        "population_density",
        "mass_kg",
        "pop_density_per_km2",
    )
    maybe(
        ALLOMETRY_CATALOG / "pantheria_home_range.csv",
        "pantheria_home_range",
        "home_range",
        "mass_kg",
        "home_range_km2",
    )
    at = ALLOMETRY_CATALOG / "animaltraits_metabolic.csv"
    maybe(at, "animaltraits_metabolic_all", "metabolic_rate_mixed", "mass_kg", "metabolic_rate")
    maybe(
        at,
        "animaltraits_metabolic_Mammalia",
        "metabolic_rate_BU",
        "mass_kg",
        "metabolic_rate",
        filters={"class": "Mammalia"},
    )
    maybe(
        ALLOMETRY_CATALOG / "animaltraits_specific_mr.csv",
        "animaltraits_specific_mr",
        "specific_metabolic",
        "mass_kg",
        "specific_mr",
    )
    maybe(
        ALLOMETRY_CATALOG / "animaltraits_specific_mr.csv",
        "animaltraits_specific_mr_Mammalia",
        "specific_metabolic",
        "mass_kg",
        "specific_mr",
        filters={"class": "Mammalia"},
    )
    maybe(
        ALLOMETRY_CATALOG / "animaltraits_brain.csv",
        "animaltraits_brain",
        "brain_size",
        "mass_kg",
        "brain_size",
    )
    for order in ("Primates", "Rodentia", "Carnivora"):
        maybe(
            ALLOMETRY_CATALOG / "animaltraits_brain.csv",
            f"animaltraits_brain_{order}",
            "brain_size",
            "mass_kg",
            "brain_size",
            filters={"order": order},
        )
    maybe(
        ALLOMETRY_CATALOG / "anage_metabolic.csv",
        "anage_metabolic",
        "metabolic_rate_BU",
        "mass_kg",
        "metabolic_rate_W",
    )
    maybe(
        ALLOMETRY_CATALOG / "anage_longevity.csv",
        "anage_longevity",
        "lifespan",
        "mass_kg",
        "max_longevity_years",
    )
    maybe(
        ALLOMETRY_CATALOG / "city_wages.csv",
        "city_wages",
        "city_socioeconomic",
        "population",
        "wages_thousands_usd",
    )
    maybe(
        ALLOMETRY_CATALOG / "city_road_length.csv",
        "city_road_length",
        "city_infrastructure",
        "population",
        "road_length_m",
    )
    if not suite:
        raise FileNotFoundError(
            f"no derived catalogs in {ALLOMETRY_CATALOG}; "
            "run experiments/hqvm_cgm_allometry_data_ingest.py"
        )
    return suite


def load_csv_series(
    path: Path,
    name: str,
    trait_class: str,
    x_col: str = "mass",
    y_col: str = "trait",
    filters: Optional[Dict[str, str]] = None,
) -> SeriesSpec:
    xs: List[float] = []
    ys: List[float] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if filters:
                ok = True
                for k, v in filters.items():
                    if (row.get(k) or "").strip() != v:
                        ok = False
                        break
                if not ok:
                    continue
            try:
                x = float(row[x_col])
                y = float(row[y_col])
            except (KeyError, ValueError):
                continue
            if x > 0 and y > 0:
                xs.append(x)
                ys.append(y)
    if len(xs) < 3:
        raise ValueError(f"insufficient rows in {path} name={name}")
    src = str(path) if not filters else f"{path}|filters={filters}"
    return SeriesSpec(name=name, trait_class=trait_class, xs=tuple(xs), ys=tuple(ys), source=src)


def compare_to_channel_rule(fit: FitResult, trait_class: str) -> Dict[str, float]:
    rules = {r.trait_class: r for r in channel_rules()}
    rule = rules.get(trait_class)
    if rule is None:
        return {"has_rule": 0.0}
    out = {
        "has_rule": 1.0,
        "a_pred": rule.a_pred,
        "dual_band": 1.0 if rule.dual_band else 0.0,
        "resid_vs_rule": abs(fit.a_primary - rule.a_pred),
    }
    if rule.dual_band and rule.mu_default is not None:
        out["mu_pred"] = rule.mu_default
        out["mu_resid"] = abs(fit.mu_primary - rule.mu_default)
    else:
        out["mu_pred"] = float("nan")
        out["mu_resid"] = float("nan")
    return out


def run_data_battery(
    series: Optional[Sequence[SeriesSpec]] = None,
    *,
    n_boot: int = BOOT_DEFAULT,
) -> Tuple[List[FitResult], List[Dict[str, float]]]:
    series = list(series or default_synthetic_suite())
    fits = [analyze_series(s, n_boot=n_boot, seed=i) for i, s in enumerate(series)]
    comps = [compare_to_channel_rule(f, s.trait_class) for f, s in zip(fits, series)]
    return fits, comps


def run_model_battery(series: Sequence[SeriesSpec]) -> List[ModelCompareRow]:
    out: List[ModelCompareRow] = []
    for s in series:
        out.extend(model_comparison(s))
    return out


@dataclass(frozen=True)
class KleiberM0Audit:
    series: str
    n: int
    a_fixed: float
    log10_K: float
    K: float
    c_M0: float
    log2_B0_emp: float
    B0_emp: float
    geom_mean_M_kg: float
    M_over_M0: float
    resid_std_log10: float
    note: str


def kleiber_m0_audit(
    spec: SeriesSpec,
    *,
    a_fixed: float = A_BULK,
) -> KleiberM0Audit:
    """Fixed-slope Kleiber intercept about M0; SI B0 from catalog K.

    Model: log2(B) = a log2(M/M0) + c_M0; theory c_M0 = log2(B0) + b_K.
    Empirically B0_emp = 2^(c_M0 − b_K); also K from B = K M^a.
    """
    k = kleiber_absolute_intercept()
    lx10 = [math.log10(v) for v in spec.xs]
    ly10 = [math.log10(v) for v in spec.ys]
    n = len(lx10)
    mx = sum(lx10) / n
    my = sum(ly10) / n
    b10 = my - a_fixed * mx
    log10_K = b10
    K = 10.0**log10_K
    resid = [y - (a_fixed * x + b10) for x, y in zip(lx10, ly10)]
    resid_std = math.sqrt(sum(r * r for r in resid) / n)
    log2_M0 = math.log2(k.M0_kg)
    c_vals = [
        math.log2(y) - a_fixed * (math.log2(x) - log2_M0)
        for x, y in zip(spec.xs, spec.ys)
    ]
    c_M0 = sum(c_vals) / n
    log2_B0_emp = c_M0 - k.b_K
    B0_emp = 2.0**log2_B0_emp
    gmean_M = 10.0 ** (sum(lx10) / n)
    return KleiberM0Audit(
        series=spec.name,
        n=n,
        a_fixed=a_fixed,
        log10_K=log10_K,
        K=K,
        c_M0=c_M0,
        log2_B0_emp=log2_B0_emp,
        B0_emp=B0_emp,
        geom_mean_M_kg=gmean_M,
        M_over_M0=gmean_M / k.M0_kg,
        resid_std_log10=resid_std,
        note="B0_emp=2^(c_M0−b_K); K from fixed-a OLS",
    )


@dataclass(frozen=True)
class ActivityRegimeRow:
    series: str
    a_primary: float
    mu_primary: float
    regime_hyp: str
    a_hyp: float
    resid_hyp: float
    resid_bulk: float
    resid_surface: float
    nearer_hyp_than_alt: bool


_ACTIVITY_REGIME: Dict[str, Tuple[str, float]] = {
    "pantheria_bmr": ("thermal", 1.0),
    "anage_metabolic": ("thermal", 1.0),
    "animaltraits_metabolic_Mammalia": ("intermediate", 0.5),
    "animaltraits_metabolic_all": ("mixed", float("nan")),
}


def activity_regime_audit(
    fits: Sequence[FitResult],
) -> List[ActivityRegimeRow]:
    """Map metabolic catalogs onto QuBEC μ_η regimes; resid vs hyp a(μ)."""
    by_name = {f.name: f for f in fits}
    rows: List[ActivityRegimeRow] = []
    for name, (regime, mu_hyp) in _ACTIVITY_REGIME.items():
        f = by_name.get(name)
        if f is None:
            continue
        if math.isnan(mu_hyp):
            a_hyp = float("nan")
            resid_hyp = float("nan")
            nearer = False
        else:
            a_hyp = a_from_mu(mu_hyp)
            resid_hyp = abs(f.a_primary - a_hyp)
            alt = A_SURFACE if mu_hyp > 0.5 else A_BULK
            nearer = resid_hyp <= abs(f.a_primary - alt) + 1e-15
        rows.append(
            ActivityRegimeRow(
                series=name,
                a_primary=f.a_primary,
                mu_primary=f.mu_primary,
                regime_hyp=regime,
                a_hyp=a_hyp,
                resid_hyp=resid_hyp,
                resid_bulk=f.resid_3_4,
                resid_surface=f.resid_2_3,
                nearer_hyp_than_alt=nearer,
            )
        )
    return rows


def activity_regime_pass(rows: Sequence[ActivityRegimeRow]) -> Dict[str, bool]:
    by = {r.series: r for r in rows}
    bmr = by.get("pantheria_bmr")
    mixed = by.get("animaltraits_metabolic_all")
    mam = by.get("animaltraits_metabolic_Mammalia")
    return {
        "activity_bmr_nearer_bulk_than_surface": bool(
            bmr and bmr.resid_bulk <= bmr.resid_surface + 1e-15
        ),
        "activity_bmr_closer_bulk_than_mixed": bool(
            bmr and mixed and bmr.resid_bulk < mixed.resid_bulk
        ),
        "activity_mammalia_nearer_mid_than_surface": bool(
            mam and mam.nearer_hyp_than_alt
        ),
    }


@dataclass(frozen=True)
class InfoConjugacyRow:
    id: str
    a_left: float
    a_right: float
    a_sum: float
    target: float
    resid: float
    status: str


def info_conjugacy_audit(
    fits: Sequence[FitResult],
) -> List[InfoConjugacyRow]:
    """Empirical conjugacy: city infra+socio → 2; brain order splits vs bulk."""
    by = {f.name: f for f in fits}
    rows: List[InfoConjugacyRow] = []

    def _status(resid: float) -> str:
        if resid <= EXACT_TOL:
            return "EXACT"
        if resid <= NEAR_TOL:
            return "NEAR"
        if resid <= LOCO_TOL:
            return "SCAN"
        return "MISS"

    wages = by.get("city_wages")
    roads = by.get("city_road_length")
    if wages is not None and roads is not None:
        s = wages.a_primary + roads.a_primary
        resid = abs(s - 2.0)
        rows.append(
            InfoConjugacyRow(
                id="city_wages+roads",
                a_left=roads.a_primary,
                a_right=wages.a_primary,
                a_sum=s,
                target=2.0,
                resid=resid,
                status=_status(resid),
            )
        )
    dens = by.get("pantheria_pop_density")
    if dens is not None:
        resid = abs(dens.a_primary + A_BULK)
        rows.append(
            InfoConjugacyRow(
                id="pop_density_OLS_vs_Damuth",
                a_left=dens.a_primary,
                a_right=float("nan"),
                a_sum=dens.a_primary,
                target=-A_BULK,
                resid=resid,
                status=_status(resid),
            )
        )
    hr = by.get("pantheria_home_range")
    if hr is not None:
        resid_cons = abs(hr.a_primary - 1.0)
        resid_met = abs(hr.a_primary - A_BULK)
        rows.append(
            InfoConjugacyRow(
                id="home_range_vs_1_cons",
                a_left=hr.a_primary,
                a_right=float("nan"),
                a_sum=hr.a_primary,
                target=1.0,
                resid=resid_cons,
                status=_status(resid_cons),
            )
        )
        rows.append(
            InfoConjugacyRow(
                id="home_range_vs_3_4_met",
                a_left=hr.a_primary,
                a_right=float("nan"),
                a_sum=hr.a_primary,
                target=A_BULK,
                resid=resid_met,
                status=_status(resid_met),
            )
        )
    for name in (
        "animaltraits_brain",
        "animaltraits_brain_Primates",
        "animaltraits_brain_Rodentia",
        "animaltraits_brain_Carnivora",
    ):
        brain = by.get(name)
        if brain is None:
            continue
        resid = abs(brain.a_primary - A_BULK)
        rows.append(
            InfoConjugacyRow(
                id=f"{name}_vs_bulk",
                a_left=brain.a_primary,
                a_right=float("nan"),
                a_sum=brain.a_primary,
                target=A_BULK,
                resid=resid,
                status=_status(resid),
            )
        )
    return rows


def run_null_battery(
    series: Sequence[SeriesSpec],
    *,
    n_perm: int = NULL_PERM_DEFAULT,
) -> List[NullAuditRow]:
    return [null_shuffle_audit(s, n_perm=n_perm, seed=10 + i) for i, s in enumerate(series)]


def tick_residual(a_fit: float, a_null: float) -> float:
    return (a_fit - a_null) * float(TICKS_PER_OCTAVE)


def ci_contains(lo: float, hi: float, a0: float) -> bool:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return False
    return lo <= a0 <= hi
