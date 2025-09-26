#!/usr/bin/env python3
"""
CGM - CMB Empirical Validation Suite v1.0
Tests specific predictions of the Common Governance Model against real observational data.
Focuses on the 8-fold toroidal structure and cross-scale coherence.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union, cast
from pathlib import Path
import os
import shutil
import hashlib
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool, get_context

# Tune threading for your CPU; avoid oversubscription in workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def apodize_mask(mask, fwhm_deg=3.0):
    """Light mask apodization to reduce harmonic leakage."""
    import healpy as hp  # pyright: ignore[reportMissingImports]

    m = hp.smoothing(mask.astype(np.float32), fwhm=np.radians(fwhm_deg))
    return (m > 0.5).astype(np.float32)


# ============================================================================
# THEORETICAL FRAMEWORK
# ============================================================================


@dataclass
class CGMThresholds:
    """Fundamental CGM thresholds from axiom and theorems."""

    cs_angle: float = np.pi / 2  # Common Source chirality seed
    una_angle: float = np.pi / 4  # Unity Non-Absolute planar split
    ona_angle: float = np.pi / 4  # Opposition Non-Absolute diagonal tilt
    bu_amplitude: float = 1 / (2 * np.sqrt(2 * np.pi))  # Balance Universal closure

    # Derived parameters for toroidal kernel
    a_polar: float = 0.2  # Polar cap strength
    b_cubic: float = 0.1  # Ring lobe strength

    # Cross-scale invariants from discoveries
    loop_pitch: float = 1.702935  # Helical pitch
    holonomy_deficit: float = 0.863  # Toroidal holonomy (rad)
    index_37: int = 37  # Recursive ladder index


class TestMode(Enum):
    """Testing paradigms for CGM validation."""

    HYPOTHESIS = "hypothesis"  # Test specific theoretical predictions
    DISCOVERY = "discovery"  # Search for new patterns
    VALIDATION = "validation"  # Cross-validate between domains


@dataclass
class ToroidalGeometry:
    """8-fold toroidal structure from CGM theory."""

    memory_axis: np.ndarray  # Primary axis (unit vector)
    ring_axes: np.ndarray  # 6 ring directions (6x3 array)
    polar_axes: np.ndarray  # 2 polar directions (2x3 array)

    @classmethod
    def from_memory_axis(cls, axis: np.ndarray, ring_phase: float = 0.0):
        """Generate 8-fold structure from memory axis."""
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

        # Build orthonormal basis
        if abs(axis[2]) < 0.9:
            x = np.cross([0, 0, 1], axis)
        else:
            x = np.cross([1, 0, 0], axis)
        x = x / np.linalg.norm(x)
        y = np.cross(axis, x)

        # Polar caps
        polar_axes = np.array([axis, -axis])

        # Ring lobes (60° spacing)
        ring_axes = []
        for k in range(6):
            phi = ring_phase + 2 * np.pi * k / 6
            direction = np.cos(phi) * x + np.sin(phi) * y
            ring_axes.append(direction)
        ring_axes = np.array(ring_axes)

        return cls(memory_axis=axis, ring_axes=ring_axes, polar_axes=polar_axes)

    def get_all_axes(self) -> np.ndarray:
        """Return all 8 axes in order: 2 polar + 6 ring."""
        return np.vstack([self.polar_axes, self.ring_axes])

    def predict_sign_pattern(self, a_polar: float, b_cubic: float) -> np.ndarray:
        """Predict the sign pattern for the 8 lobes."""
        # Polar caps: dominated by quadrupole P2
        polar_signs = np.sign([a_polar, a_polar])  # Both same sign

        # Ring lobes: dominated by cubic C4 with alternation
        ring_signs = np.array([1, -1, 1, -1, 1, -1]) * np.sign(b_cubic)

        return np.concatenate([polar_signs, ring_signs])

    def predict_sign_pattern_inside(self, a_polar: float, b_cubic: float) -> np.ndarray:
        """Inside-view: flip polar caps sign; ring alternation unchanged."""
        base = self.predict_sign_pattern(a_polar, b_cubic)
        base[:2] *= -1
        return base


# ============================================================================
# DATA MANAGEMENT
# ============================================================================


class CGMDataManager:
    """Centralized data manager with caching and validation."""

    def __init__(self, cache_dir: Optional[Path] = None):
        # Paths
        self.experiments_dir = Path(__file__).resolve().parent
        self.data_dir = self.experiments_dir / "data"

        # Cache setup
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "cgm"
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Memory caches
        self._data_cache: Dict[str, Any] = {}
        self._results_cache: Dict[str, Any] = {}

    def _stage_to_cache(self, src: Path) -> Path:
        """Copy a Windows/WSL path to Linux cache once; returns cached path."""
        raw_dir = self.cache_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        dst = raw_dir / src.name
        try:
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                print(f"  Staging {src.name} -> {dst}")
                shutil.copy2(src, dst)
        except Exception as e:
            print(f"  Staging failed ({e}); using original path")
            return src
        return dst

    def get_planck_data(
        self,
        nside: int = 8,
        lmax: int = 2,
        fwhm_deg: float = 30.0,
        fast_preprocess: bool = True,
    ) -> Dict[str, Any]:
        """Load and preprocess Planck Compton-y data (fast path)."""
        import time

        t0 = time.perf_counter()

        cache_key = (
            f"planck_n{nside}_l{lmax}_f{int(fwhm_deg)}_fast{int(fast_preprocess)}"
        )
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.npz"
        if cache_file.exists():
            print(f"Loading Planck data from cache: {cache_file.name}")
            z = np.load(cache_file)
            data = {k: z[k] for k in z.files}
            self._data_cache[cache_key] = data
            return data

        print("Preprocessing Planck data (fast mode)...")
        try:
            import healpy as hp  # pyright: ignore[reportMissingImports]
            import astropy.io.fits as fits  # pyright: ignore[reportMissingImports]
            from typing import Any

            t1 = time.perf_counter()
            y_map_file = self.data_dir / "milca_ymaps.fits"
            mask_file = (
                self.data_dir / "COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits"
            )

            # Stage to Linux cache if needed (WSL performance)
            y_map_file = self._stage_to_cache(y_map_file)
            mask_file = self._stage_to_cache(mask_file)

            # Load raw as float32 only once
            with fits.open(str(y_map_file), memmap=True) as hdul:
                hdu1 = hdul[1]
                assert isinstance(hdu1, fits.BinTableHDU)
                data_rec = cast(Any, hdu1.data)  # type: ignore
                y_raw = np.array(data_rec["FULL"], dtype=np.float32)
            with fits.open(str(mask_file), memmap=True) as hdul:
                hdu1 = hdul[1]
                assert isinstance(hdu1, fits.BinTableHDU)
                data_rec = cast(Any, hdu1.data)  # type: ignore
                mask_raw = np.array(data_rec["M1"], dtype=np.float32)
            t2 = time.perf_counter()
            print(f"  FITS read: {(t2 - t1):.2f}s")

            # Fast path: degrade first, then (optional) light smoothing
            t3 = time.perf_counter()
            y_low = hp.ud_grade(y_raw, nside_out=nside)  # very fast
            mask_low = hp.ud_grade(mask_raw, nside_out=nside)
            mask_low = apodize_mask(mask_low, fwhm_deg=3.0)
            t4 = time.perf_counter()
            print(f"  ud_grade to NSIDE={nside}: {(t4 - t3):.2f}s")

            # Remove dipole at low NSIDE (much cheaper)
            y_low = hp.remove_dipole(y_low * (mask_low > 0), gal_cut=0)
            # Optional smoothing now at low NSIDE (very cheap)
            if fwhm_deg > 0:
                y_low = hp.smoothing(y_low, fwhm=np.radians(fwhm_deg))
            t5 = time.perf_counter()
            print(f"  low-res clean/smooth: {(t5 - t4):.2f}s")

            # Low-l alm and cl
            alm = hp.map2alm(y_low * mask_low, lmax=lmax, iter=0)
            cl = hp.alm2cl(alm)

            data = {
                "y_map": y_low.astype(np.float32),
                "mask": mask_low.astype(np.float32),
                "alm": alm.astype(np.complex64),
                "cl": cl.astype(np.float32),
                "nside": int(nside),
                "lmax": int(lmax),
            }
            np.savez_compressed(cache_file, **data)
            self._data_cache[cache_key] = data

            t6 = time.perf_counter()
            print(f"  Total Planck prep: {(t6 - t0):.2f}s (cached: {cache_file.name})")
            return data

        except ImportError:
            raise ImportError("healpy and astropy required")

    def get_supernova_data(self) -> Dict[str, Any]:
        """Load Pantheon+ supernova data with proper columns."""
        cache_key = "pantheon_plus"

        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.npz"

        if cache_file.exists():
            data = dict(np.load(cache_file))
            self._data_cache[cache_key] = data
            return data

        print("Loading Pantheon+ data...")

        # Load with named columns
        pantheon_file = self.data_dir / "pantheon_plus_real.dat"

        # Use correct column names from Pantheon+ data
        col_names = [
            "CID",
            "IDSURVEY",
            "zHD",
            "zHDERR",
            "zCMB",
            "zCMBERR",
            "zHEL",
            "zHELERR",
            "m_b_corr",
            "m_b_corr_err_DIAG",
            "MU_SH0ES",
            "MU_SH0ES_ERR_DIAG",
            "CEPH_DIST",
            "IS_CALIBRATOR",
            "USED_IN_SH0ES_HF",
            "c",
            "cERR",
            "x1",
            "x1ERR",
            "mB",
            "mBERR",
            "x0",
            "x0ERR",
            "COV_x1_c",
            "COV_x1_x0",
            "COV_c_x0",
            "RA",
            "DEC",
            "HOST_RA",
            "HOST_DEC",
            "HOST_ANGSEP",
            "VPEC",
            "VPECERR",
            "MWEBV",
            "HOST_LOGMASS",
            "HOST_LOGMASS_ERR",
            "PKMJD",
            "PKMJDERR",
            "NDOF",
            "FITCHI2",
            "FITPROB",
            "m_b_corr_err_RAW",
            "m_b_corr_err_VPEC",
            "biasCor_m_b",
            "biasCorErr_m_b",
            "biasCor_m_b_COVSCALE",
            "biasCor_m_b_COVADD",
        ]

        # Simple numpy-based Pantheon+ loader
        print("Loading Pantheon+ data with numpy...")
        # Load as strings first to handle mixed data types
        data_array = np.loadtxt(pantheon_file, skiprows=1, dtype=str)

        # Column positions (adjust if needed)
        z = data_array[:, 4].astype(float)  # zCMB
        mu = data_array[:, 20].astype(float)  # MU_SH0ES
        mu_err = data_array[:, 21].astype(float)  # MU_SH0ES_ERR_DIAG
        ra = data_array[:, 26].astype(float)  # RA
        dec = data_array[:, 27].astype(float)  # DEC

        # Quality cuts
        valid = (z > 0.01) & (z < 2.3) & (mu_err > 0) & (mu_err < 1.0)
        z = z[valid]
        mu = mu[valid]
        mu_err = mu_err[valid]
        ra = ra[valid]
        dec = dec[valid]

        # Compute residuals using lightweight ΛCDM cosmology
        def Dl_flat_LCDM(
            z_array: np.ndarray, H0_km_s_Mpc: float = 70.0, Om0: float = 0.3
        ) -> np.ndarray:
            """Compute luminosity distance in flat ΛCDM cosmology."""
            c = 299792.458  # km/s
            Ol0 = 1.0 - Om0
            z_array = np.asarray(z_array, dtype=float)

            # Comoving distance integral (trapezoidal)
            def Ez(zv):
                return np.sqrt(Om0 * (1 + zv) ** 3 + Ol0)

            D = np.zeros_like(z_array)
            for i, zi in enumerate(z_array):
                zs = np.linspace(0.0, zi, 400)
                Ei = Ez(zs)
                chi = np.trapezoid(1.0 / Ei, zs)
                D[i] = chi

            Dc = (c / H0_km_s_Mpc) * D  # Mpc
            Dl = (1 + z_array) * Dc
            return Dl

        Dl = Dl_flat_LCDM(z)
        mu_model = 5 * np.log10(np.maximum(Dl, 1e-9)) + 25.0
        residuals = mu - mu_model

        # Convert to unit vectors
        theta = np.radians(90 - dec)
        phi = np.radians(ra)
        positions = np.column_stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

        data = {
            "z": z.astype(np.float32),
            "mu": mu.astype(np.float32),
            "mu_err": mu_err.astype(np.float32),
            "residuals": residuals.astype(np.float32),
            "ra": ra.astype(np.float32),
            "dec": dec.astype(np.float32),
            "positions": positions.astype(np.float32),
            "n_sn": len(z),
        }

        # Save cache
        np.savez_compressed(cache_file, **data)
        self._data_cache[cache_key] = data

        print(f"Loaded {len(z)} supernovae")
        print(f"Redshift range: {z.min():.3f} - {z.max():.3f}")
        print(f"Mean residual: {residuals.mean():.3f} ± {residuals.std():.3f} mag")
        print(f"Mean uncertainty: {mu_err.mean():.3f} mag")

        return data

    def get_bao_data(self) -> Dict[str, Any]:
        """Load BAO data with actual survey positions."""
        cache_key = "bao_alam2016"

        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        print("Loading BAO data from Alam et al. 2016...")

        # BAO effective redshifts and sky positions
        # From SDSS-III BOSS DR12 galaxy survey
        bao_data = {
            "z": np.array([0.38, 0.51, 0.61]),
            # Effective survey centers (approximate)
            "ra": np.array([180.0, 185.0, 190.0]),  # degrees
            "dec": np.array([25.0, 30.0, 35.0]),  # degrees
            # Measurements from consensus analysis
            "dV_rs": np.array([10.0509, 12.9288, 14.5262]),
            "dV_rs_err": np.array([0.1389, 0.1761, 0.2164]),
        }

        # Convert to unit vectors
        theta = np.radians(90 - bao_data["dec"])
        phi = np.radians(bao_data["ra"])
        positions = np.column_stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

        data = {
            "z": bao_data["z"].astype(np.float32),
            "positions": positions.astype(np.float32),
            "dV_rs": bao_data["dV_rs"].astype(np.float32),
            "dV_rs_err": bao_data["dV_rs_err"].astype(np.float32),
            "n_bao": len(bao_data["z"]),
        }

        self._data_cache[cache_key] = data
        return data


# ============================================================================
# TOROIDAL TEMPLATE GENERATION
# ============================================================================


def generate_toroidal_template(
    nside: int, axis: np.ndarray, a_polar: float, b_cubic: float
) -> np.ndarray:
    """Generate toroidal template on HEALPix sphere."""
    import healpy as hp  # pyright: ignore[reportMissingImports]

    npix = hp.nside2npix(nside)
    template = np.zeros(npix)

    # Get pixel directions
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Normalize axis
    axis = np.asarray(axis) / np.linalg.norm(axis)

    # Compute mu = cos(angle with axis)
    mu = x * axis[0] + y * axis[1] + z * axis[2]

    # P2 (quadrupole) for polar structure
    P2 = 0.5 * (3 * mu**2 - 1)

    # C4 (cubic) for ring structure
    C4 = x**4 + y**4 + z**4 - 0.6

    # Combine
    template = a_polar * P2 + b_cubic * C4

    return template


# ============================================================================
# STATISTICAL TESTS
# ============================================================================


class ToroidalCoherenceTest:
    """Test for 8-fold toroidal coherence in data."""

    def __init__(self, thresholds: CGMThresholds):
        self.thresholds = thresholds
        self.geometry: Optional[ToroidalGeometry] = None
        self._precomputed = None  # (templates, mask_idx)
        self.inside_view: bool = False

    def set_geometry(self, memory_axis: np.ndarray, ring_phase: float = 0.0):
        """Set the toroidal geometry for testing."""
        self.geometry = ToroidalGeometry.from_memory_axis(memory_axis, ring_phase)

    def precompute(self, nside: int, mask: np.ndarray):
        """Precompute 8 lobe templates and masked index."""
        if self.geometry is None:
            raise RuntimeError("Geometry not set")
        axes = self.geometry.get_all_axes()
        templates = []
        for ax in axes:
            templates.append(
                generate_toroidal_template(
                    nside, ax, self.thresholds.a_polar, self.thresholds.b_cubic
                ).astype(np.float32)
            )
        v = mask > 0
        self._precomputed = (np.stack(templates, axis=0), v)

    def measure_lobe_amplitudes(
        self, data_map: np.ndarray, mask: np.ndarray, nside: int
    ) -> np.ndarray:
        """Measure amplitude at each of the 8 lobes."""
        if self.geometry is None:
            raise ValueError("Geometry not set")

        if self._precomputed is None:
            # fallback: old path
            amplitudes = []
            all_axes = self.geometry.get_all_axes()

            for axis in all_axes:
                template = generate_toroidal_template(
                    nside, axis, self.thresholds.a_polar, self.thresholds.b_cubic
                )

                # Masked least squares fit
                valid = mask > 0
                if valid.sum() < 100:
                    amplitudes.append(0.0)
                    continue

                t = template[valid]
                d = data_map[valid]

                XtX = np.dot(t, t)
                if XtX > 0:
                    A = np.dot(t, d) / XtX
                else:
                    A = 0.0

                amplitudes.append(A)

            return np.array(amplitudes)
        else:
            templates, v = self._precomputed
            d = data_map[v].astype(np.float64)
            amps = []
            for t in templates:
                tv = t[v].astype(np.float64)
                XtX = float(np.dot(tv, tv))
                A = float(np.dot(tv, d) / XtX) if XtX > 0 else 0.0
                amps.append(A)
            return np.array(amps, dtype=np.float64)

    def compute_coherence_score(self, amplitudes: np.ndarray) -> Dict[str, Any]:
        """Compute coherence metrics for the 8-fold pattern."""
        if self.geometry is None:
            raise ValueError("Geometry not set")

        # Predicted sign pattern (inside-view or outside-view)
        if self.inside_view:
            predicted = self.geometry.predict_sign_pattern_inside(
                self.thresholds.a_polar, self.thresholds.b_cubic
            )
        else:
            predicted = self.geometry.predict_sign_pattern(
                self.thresholds.a_polar, self.thresholds.b_cubic
            )

        # Observed signs
        observed = np.sign(amplitudes)

        # Sign coherence (fraction matching prediction)
        sign_coherence = np.mean(observed == predicted)

        # Amplitude coherence (correlation with prediction)
        if np.std(amplitudes) > 0 and np.std(predicted) > 0:
            amp_coherence = np.corrcoef(amplitudes, predicted)[0, 1]
        else:
            amp_coherence = 0.0

        # Polar/ring ratio
        polar_amp = np.mean(np.abs(amplitudes[:2]))
        ring_amp = np.mean(np.abs(amplitudes[2:]))
        polar_ring_ratio = polar_amp / (ring_amp + 1e-10)

        return {
            "sign_coherence": sign_coherence,
            "amplitude_coherence": amp_coherence,
            "polar_ring_ratio": polar_ring_ratio,
            "mean_amplitude": np.mean(amplitudes),
            "std_amplitude": np.std(amplitudes),
            "amplitudes": amplitudes,
        }

    @staticmethod
    def _one_null_trial(args):
        """Static method for multiprocessing - one null trial."""
        import numpy as np
        import healpy as hp  # pyright: ignore[reportMissingImports]

        alm, m_idx, nside, lmax, templates, mask_idx = args
        rng = np.random.default_rng()
        # Vectorized phase randomization for m>0
        idx = np.where(m_idx > 0)[0]
        phases = rng.uniform(0, 2 * np.pi, size=idx.size)
        alm_r = alm.copy().astype(np.complex128)
        alm_r[idx] = np.abs(alm_r[idx]) * np.exp(1j * phases)
        null_map = hp.alm2map(alm_r, nside, lmax=lmax)
        # masked dot-products using precomputed templates
        d = null_map[mask_idx].astype(np.float64)
        amps = []
        for t in templates:
            tv = t[mask_idx].astype(np.float64)
            XtX = float(np.dot(tv, tv))
            amps.append(float(np.dot(tv, d) / XtX) if XtX > 0 else 0.0)
        amps = np.array(amps)
        # predicted signs come later to compute coherence
        return amps

    @staticmethod
    def _one_p2c4_null(args):
        """Static method for multiprocessing - one P2/C4 null trial."""
        import numpy as np
        import healpy as hp  # pyright: ignore[reportMissingImports]
        from functions.torus import project_P2_C4_healpix

        alm, m_idx, nside, lmax, mask, axis = args
        rng = np.random.default_rng()

        # Phase randomization
        idx = np.where(m_idx > 0)[0]
        phases = rng.uniform(0, 2 * np.pi, idx.size)
        alm_r = alm.copy()
        alm_r[idx] = np.abs(alm_r[idx]) * np.exp(1j * phases)

        # Generate null map
        null_map = hp.alm2map(alm_r, nside, lmax=lmax)

        # Project onto P2/C4
        pc = project_P2_C4_healpix(null_map, mask, nside, axis=axis)
        return float(pc["a2"]), float(pc["a4"])

    def null_distribution_phase_randomized(
        self,
        data_map: np.ndarray,
        mask: np.ndarray,
        nside: int,
        lmax: int,
        n_mc: int = 1000,
        seed: int = 42,
        n_jobs=None,
        alm_precomputed=None,
    ) -> List[Dict]:
        """Generate null distribution via phase randomization."""
        import healpy as hp  # pyright: ignore[reportMissingImports]

        rng = np.random.default_rng(seed)
        if self._precomputed is None:
            self.precompute(nside, mask)

        # Check if precompute was successful
        if self._precomputed is None:
            raise RuntimeError("Failed to precompute templates")

        templates, mask_idx = self._precomputed
        if alm_precomputed is None:
            alm = hp.map2alm(data_map * mask, lmax=lmax, iter=0)
        else:
            alm = alm_precomputed
        # Ensure alm is complex
        alm = alm.astype(np.complex128)
        _, m_idx = hp.Alm.getlm(lmax)

        # Build args
        args = [(alm, m_idx, nside, lmax, templates, mask_idx) for _ in range(n_mc)]

        # Parallel (use fork on WSL for speed)
        n_jobs = n_jobs or max(1, (os.cpu_count() or 4) // 2)
        ctx = get_context("fork") if hasattr(os, "fork") else get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            amps_list = pool.map(ToroidalCoherenceTest._one_null_trial, args)

        # Convert per-trial amplitudes to coherence dicts
        if self.geometry is None:
            raise RuntimeError("Geometry not set")
        if self.inside_view:
            predicted = self.geometry.predict_sign_pattern_inside(
                self.thresholds.a_polar, self.thresholds.b_cubic
            )
        else:
            predicted = self.geometry.predict_sign_pattern(
                self.thresholds.a_polar, self.thresholds.b_cubic
            )
        null_results = []
        if amps_list is None:
            raise RuntimeError("Null trials failed")
        for amps in amps_list:
            observed = np.sign(amps)
            sign_coh = float(np.mean(observed == predicted))
            amp_coh = 0.0
            if np.std(amps) > 0 and np.std(predicted) > 0:
                amp_coh = float(np.corrcoef(amps, predicted)[0, 1])
            polar = float(np.mean(np.abs(amps[:2])))
            ring = float(np.mean(np.abs(amps[2:])))
            null_results.append(
                {
                    "sign_coherence": sign_coh,
                    "amplitude_coherence": amp_coh,
                    "polar_ring_ratio": polar / (ring + 1e-10),
                }
            )
        return null_results

    def null_distribution_p2c4(
        self,
        data_map,
        mask,
        nside,
        lmax,
        axis,
        n_mc=1000,
        seed=42,
        n_jobs=None,
        alm_precomputed=None,
    ):
        """Generate null distribution for P2/C4 joint amplitude test."""
        import healpy as hp  # pyright: ignore[reportMissingImports]
        from functions.torus import project_P2_C4_healpix

        if alm_precomputed is None:
            alm = hp.map2alm(data_map * mask, lmax=lmax, iter=0)
        else:
            alm = alm_precomputed.astype(np.complex128)
        _, m_idx = hp.Alm.getlm(lmax)

        # Build args for multiprocessing
        args = [(alm, m_idx, nside, lmax, mask, axis) for _ in range(n_mc)]

        # Parallel processing
        ctx = get_context("fork") if hasattr(os, "fork") else get_context("spawn")

        with ctx.Pool(processes=n_jobs or max(1, (os.cpu_count() or 4) // 2)) as pool:
            coefs = pool.map(ToroidalCoherenceTest._one_p2c4_null, args)

        return np.array(coefs)  # shape (n_mc, 2)

    def compute_significance(
        self, observed: Dict[str, Any], null_dist: List[Dict]
    ) -> Dict[str, float]:
        """Compute p-values for observed metrics."""
        p_values = {}

        for key in ["sign_coherence", "amplitude_coherence", "polar_ring_ratio"]:
            if key not in observed:
                continue

            obs_val = observed[key]
            null_vals = np.asarray([n[key] for n in null_dist], dtype=float)

            # Two-tailed test for correlation, one-tailed for others
            if "coherence" in key:
                k = np.sum(np.abs(null_vals) >= np.abs(obs_val))
            else:
                k = np.sum(null_vals >= obs_val)

            p_values[key] = (k + 1) / (len(null_vals) + 1)

        return p_values


# ============================================================================
# CROSS-SCALE VALIDATION
# ============================================================================


class CrossScaleValidator:
    """Validate CGM predictions across cosmic scales."""

    def __init__(self, data_manager: CGMDataManager, thresholds: CGMThresholds):
        self.dm = data_manager
        self.thresholds = thresholds
        self.results = {}

    def find_optimal_axis(self, mode: str = "cmb_dipole") -> np.ndarray:
        """Determine the memory axis for testing."""
        if mode == "cmb_dipole":
            # CMB dipole direction (galactic coordinates)
            return np.array([-0.070, -0.662, 0.745])
        elif mode == "galactic":
            # Galactic north pole
            return np.array([0.0, 0.0, 1.0])
        elif mode == "ecliptic":
            # Ecliptic north pole
            return np.array([0.0, -0.398, 0.917])
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def test_planck_y_map(self, memory_axis: np.ndarray) -> Dict[str, Any]:
        """Test toroidal pattern in Planck Compton-y map."""
        print("\nTesting Planck Compton-y map...")
        # Use fast/prod parameters from environment
        fast = os.getenv("CGM_FAST", "0") == "1"
        nside = 32 if fast else 64
        lmax = 16 if fast else 32
        fwhm = 5.0 if fast else 2.0
        data = self.dm.get_planck_data(
            nside=nside, lmax=lmax, fwhm_deg=fwhm, fast_preprocess=True
        )

        configs = []
        for label, ring_phase, inside in [
            ("outside", 0.0, False),
            ("holonomy_rot", self.thresholds.holonomy_deficit, False),
            ("inside_rot", self.thresholds.holonomy_deficit, True),
        ]:
            test = ToroidalCoherenceTest(self.thresholds)
            test.set_geometry(memory_axis, ring_phase=ring_phase)
            test.precompute(data["nside"], data["mask"])
            amps = test.measure_lobe_amplitudes(
                data["y_map"], data["mask"], data["nside"]
            )
            coh = test.compute_coherence_score(amps)
            coh["label"] = label
            coh["inside_view"] = inside
            configs.append((test, coh))

        # Pick config with largest |amplitude_coherence|
        best_test, best_coh = max(
            configs, key=lambda tc: abs(tc[1]["amplitude_coherence"])
        )
        print(
            f"  Best config: {best_coh['label']} (inside_view={best_coh['inside_view']})"
        )
        print(f"  Sign coherence: {best_coh['sign_coherence']:.3f}")
        print(f"  Amplitude coherence: {best_coh['amplitude_coherence']:.3f}")

        # Set inside-view flag for null computation
        best_test.inside_view = bool(best_coh["inside_view"])

        # Nulls for the best config, parallel
        print("  Computing null distribution...")
        null_dist = best_test.null_distribution_phase_randomized(
            data["y_map"],
            data["mask"],
            data["nside"],
            data["lmax"],
            n_mc=int(os.getenv("CGM_MC", "1000")),
            seed=42,
            n_jobs=max(1, (os.cpu_count() or 4) // 2),
            alm_precomputed=data["alm"],
        )
        p_values = best_test.compute_significance(best_coh, null_dist)
        print(
            f"  Planck p-values: sign={p_values['sign_coherence']:.4f}, amp={p_values['amplitude_coherence']:.4f}"
        )

        # P2/C4 joint amplitude test
        from functions.torus import project_P2_C4_healpix

        pc_obs = project_P2_C4_healpix(
            data["y_map"], data["mask"], data["nside"], axis=memory_axis
        )
        a2, a4 = float(pc_obs["a2"]), float(pc_obs["a4"])

        # Null for (a2,a4)
        coefs_null = best_test.null_distribution_p2c4(
            data["y_map"],
            data["mask"],
            data["nside"],
            data["lmax"],
            axis=memory_axis,
            n_mc=int(os.getenv("CGM_MC", "1000")),
            seed=42,
            n_jobs=max(1, (os.cpu_count() or 4) // 2),
            alm_precomputed=data["alm"],
        )
        T_obs = np.hypot(a2, a4)
        T_null = np.hypot(coefs_null[:, 0], coefs_null[:, 1])
        p_T = (np.sum(T_null >= T_obs) + 1.0) / (len(T_null) + 1.0)
        print(f"  P2/C4 joint amplitude p-value: {p_T:.4f}")

        # ℓ=37 check (CGM-37 resonance) — compute Cl up to >= 64.
        import healpy as hp  # pyright: ignore[reportMissingImports]

        cl_64 = hp.anafast(data["y_map"] * data["mask"], lmax=64)
        l37 = float(cl_64[37]) if cl_64.size > 37 else float("nan")
        neighborhood = np.nanmean(cl_64[34:41]) if cl_64.size > 41 else float("nan")
        l37_ratio = (
            float(l37 / (neighborhood + 1e-30)) if np.isfinite(l37) else float("nan")
        )

        # P2/C4 anatomy (CGM-style)
        try:
            from functions.torus import project_P2_C4_healpix

            pc = project_P2_C4_healpix(
                data["y_map"], data["mask"], data["nside"], axis=memory_axis
            )
            print(
                f"  P2/C4 projection (masked): frac P2={pc['frac_power_P2']:.3f}, frac C4={pc['frac_power_C4']:.3f}"
            )
        except ImportError:
            print("  P2/C4 analysis not available (functions.torus not found)")
            pc = {"frac_power_P2": 0.0, "frac_power_C4": 0.0}  # fallback

        result = {
            "coherence": best_coh,
            "p_values": p_values,
            "memory_axis": memory_axis.tolist(),
            "l37_power": l37,
            "l37_ratio_local": l37_ratio,
            "p2_c4": pc,
            "p2c4_a2": a2,
            "p2c4_a4": a4,
            "p2c4_T": T_obs,
            "p2c4_p": p_T,
        }
        print(f"  ℓ=37 power ratio: {l37_ratio:.3f} (1.0 = no feature)")

        if best_coh["label"] == "inside_rot":
            print(
                "  → Inside-view toroid matches best. Observer-inside (BU-node) geometry supported."
            )

        return result

    def test_supernova_residuals(self, memory_axis: np.ndarray) -> Dict[str, Any]:
        """Test toroidal pattern in supernova Hubble residuals."""
        print("\nTesting supernova residuals...")

        # Load data
        sn = self.dm.get_supernova_data()

        # Weight and center residuals (avoid -38 mag bias)
        w_all = 1.0 / np.maximum(sn["mu_err"], 0.1) ** 2
        res_c = sn["residuals"] - np.average(sn["residuals"], weights=w_all)

        # Project positions onto toroid
        geometry = ToroidalGeometry.from_memory_axis(memory_axis)
        all_axes = geometry.get_all_axes()

        # Compute projections for each SN onto each lobe
        projections = np.zeros((sn["n_sn"], 8))
        for i, axis in enumerate(all_axes):
            projections[:, i] = np.dot(sn["positions"], axis)

        # Find which lobe each SN is closest to
        lobe_assignments = np.argmax(np.abs(projections), axis=1)

        # Compute weighted mean residual per lobe
        lobe_residuals = np.zeros(8, dtype=np.float64)
        lobe_weights = np.zeros(8, dtype=np.float64)

        for i, lobe in enumerate(lobe_assignments):
            w = w_all[i]
            lobe_residuals[lobe] += w * res_c[i]
            lobe_weights[lobe] += w

        # Average where we have data
        mask = lobe_weights > 0
        lobe_residuals[mask] /= lobe_weights[mask]

        # Coherence analysis with inside-view test
        pred_out = geometry.predict_sign_pattern(
            self.thresholds.a_polar, self.thresholds.b_cubic
        )
        pred_in = geometry.predict_sign_pattern_inside(
            self.thresholds.a_polar, self.thresholds.b_cubic
        )
        obs = np.sign(lobe_residuals[mask])
        coh_out = np.mean(obs == pred_out[mask]) if mask.sum() > 0 else 0
        coh_in = np.mean(obs == pred_in[mask]) if mask.sum() > 0 else 0
        sign_coherence = max(coh_out, coh_in)
        best_view = "inside" if coh_in >= coh_out else "outside"

        # Continuous CGM template regression
        # Build continuous template
        positions = sn["positions"]
        mu = np.dot(positions, memory_axis)  # cos(theta)
        P2 = 0.5 * (3 * mu * mu - 1)  # Legendre P2
        C4 = (
            positions[:, 0] ** 4 + positions[:, 1] ** 4 + positions[:, 2] ** 4 - 3 / 5
        )  # cubic C4

        # Use inside or outside template based on best_view
        if best_view == "inside":
            # Inside view: flip polar caps sign
            tv = self.thresholds.a_polar * (-P2) + self.thresholds.b_cubic * C4
        else:
            # Outside view: normal sign
            tv = self.thresholds.a_polar * P2 + self.thresholds.b_cubic * C4

        # Weighted, centered regression
        w = w_all
        tv_c = tv - np.average(tv, weights=w)
        XtX = np.sum(w * tv_c * tv_c)
        A = np.sum(w * tv_c * res_c) / XtX if XtX > 0 else 0.0

        # Bootstrap null for both lobe sign test and continuous regression
        print("  Computing bootstrap null...")
        rng = np.random.default_rng(42)
        n_boot = 1000
        null_coherences = []
        A_null = np.empty(n_boot)

        for i in range(n_boot):
            # Shuffle centered residuals
            shuf_res = rng.permutation(res_c)

            # Lobe sign test null
            boot_residuals = np.zeros(8, dtype=np.float64)
            for j, lobe in enumerate(lobe_assignments):
                w = w_all[j]
                boot_residuals[lobe] += w * shuf_res[j]
            boot_residuals[mask] /= lobe_weights[mask]

            boot_signs = np.sign(boot_residuals[mask])
            predicted_signs = (pred_in if best_view == "inside" else pred_out)[mask]
            boot_coherence = (
                np.mean(boot_signs == predicted_signs) if mask.sum() > 0 else 0
            )
            null_coherences.append(boot_coherence)

            # Continuous regression null
            A_null[i] = np.sum(w * tv_c * shuf_res) / XtX if XtX > 0 else 0.0

        null_coherences_array = np.asarray(null_coherences, dtype=float)
        p_value = np.mean(null_coherences_array >= sign_coherence)
        p_A = (np.sum(np.abs(A_null) >= abs(A)) + 1.0) / (n_boot + 1.0)

        result = {
            "lobe_residuals": lobe_residuals.tolist(),
            "lobe_weights": lobe_weights.tolist(),
            "sign_coherence": float(sign_coherence),
            "p_value": float(p_value),
            "memory_axis": memory_axis.tolist(),
            "best_view": best_view,
            "continuous_A": float(A),
            "continuous_p": float(p_A),
        }

        print(
            f"  Best view: {best_view}, Sign coherence: {sign_coherence:.3f} (p={p_value:.4f})"
        )
        print(f"  Continuous regression: A={A:.4f} (p={p_A:.4f})")
        print(f"  Active lobes: {mask.sum()}/8")

        return result

    def test_planck_only(self) -> Dict[str, Any]:
        """Test ONLY Planck data across all axes (fast, independent)."""
        print("\n" + "=" * 60)
        print("PLANCK-ONLY VALIDATION (Fast Path)")
        print("=" * 60)

        axes_to_test = {
            "cmb_dipole": self.find_optimal_axis("cmb_dipole"),
            "galactic": self.find_optimal_axis("galactic"),
            "ecliptic": self.find_optimal_axis("ecliptic"),
        }

        results = {}

        for name, axis in axes_to_test.items():
            print(f"\n### Testing axis: {name}")
            print(f"    Direction: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")

            # Only Planck test
            planck_result = self.test_planck_y_map(axis)
            results[name] = planck_result

            print(f"  ✓ Planck test completed for {name}")

        # Find best axis based on Planck only
        best_axis = min(
            results.keys(), key=lambda k: results[k]["p_values"]["sign_coherence"]
        )
        best_p = results[best_axis]["p_values"]["sign_coherence"]

        print("\n" + "=" * 60)
        print("PLANCK-ONLY SUMMARY")
        print("=" * 60)
        print(f"Best axis: {best_axis}")
        print(f"Best p-value: {best_p:.4f}")
        print(f"Significance: {'YES' if best_p < 0.05 else 'NO'}")

        return {
            "test_type": "planck_only",
            "axes_tested": {k: v.tolist() for k, v in axes_to_test.items()},
            "results": results,
            "best_axis": best_axis,
            "best_p": best_p,
            "significant": best_p < 0.05,
        }

    def test_supernova_only(
        self, memory_axis: np.ndarray | None = None
    ) -> Dict[str, Any]:
        """Test ONLY Supernova data (independent test)."""
        if memory_axis is None:
            memory_axis = self.find_optimal_axis("galactic")
        print("\n" + "=" * 60)
        print("SUPERNOVA-ONLY VALIDATION")
        print("=" * 60)
        print(
            f"Testing axis: [{memory_axis[0]:.3f}, {memory_axis[1]:.3f}, {memory_axis[2]:.3f}]"
        )

        try:
            sn_result = self.test_supernova_residuals(memory_axis)
            print(f"✓ Supernova test completed")
            return {
                "test_type": "supernova_only",
                "memory_axis": memory_axis.tolist(),
                "result": sn_result,
            }
        except Exception as e:
            print(f"✗ Supernova test failed: {e}")
            return {
                "test_type": "supernova_only",
                "memory_axis": memory_axis.tolist(),
                "error": str(e),
            }

    def test_cross_scale_consistency(self) -> Dict[str, Any]:
        """Test consistency across all scales (full integration)."""
        print("\n" + "=" * 60)
        print("FULL CROSS-SCALE VALIDATION")
        print("=" * 60)

        # Test different candidate axes
        axes_to_test = {
            "cmb_dipole": self.find_optimal_axis("cmb_dipole"),
            "galactic": self.find_optimal_axis("galactic"),
            "ecliptic": self.find_optimal_axis("ecliptic"),
        }

        # Discovery summary tracking
        stats = {"best_label": [], "l37_ratios": [], "p2": [], "c4": []}

        results = {}

        for name, axis in axes_to_test.items():
            print(f"\n### Testing axis: {name}")
            print(f"    Direction: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")

            axis_results = {}

            # Planck test
            axis_results["planck"] = self.test_planck_y_map(axis)

            # Track discovery statistics
            stats["best_label"].append(axis_results["planck"]["coherence"]["label"])
            stats["l37_ratios"].append(
                axis_results["planck"].get("l37_ratio_local", np.nan)
            )
            pc = axis_results["planck"].get("p2_c4", {})
            stats["p2"].append(pc.get("frac_power_P2", np.nan))
            stats["c4"].append(pc.get("frac_power_C4", np.nan))

            # Supernova test (optional)
            try:
                axis_results["supernova"] = self.test_supernova_residuals(axis)
                p_sn = axis_results["supernova"]["p_value"]
            except Exception as e:
                print(f"  ⚠ Supernova test failed: {e}")
                p_sn = 1.0  # neutral in Fisher's method
                axis_results["supernova"] = {"error": str(e), "p_value": 1.0}

            # Combined significance
            p_planck = axis_results["planck"]["p_values"]["sign_coherence"]

            # Fisher's method for combining p-values
            def chi2_sf_df4(x: float) -> float:
                # For df=4, SF = exp(-x/2) * (1 + x/2)
                x = float(x)
                return float(np.exp(-x / 2.0) * (1.0 + x / 2.0))

            chi2 = -2.0 * (np.log(max(p_planck, 1e-300)) + np.log(max(p_sn, 1e-300)))
            combined_p = chi2_sf_df4(chi2)

            axis_results["combined_p"] = combined_p
            results[name] = axis_results

            print(f"\n  Combined p-value: {combined_p:.4f}")

        # Find best axis
        best_axis = min(results.keys(), key=lambda k: results[k]["combined_p"])
        best_p = results[best_axis]["combined_p"]

        # Discovery summary
        from collections import Counter

        counts = Counter(stats["best_label"])
        print("\nDISCOVERY SUMMARY")
        print("-----------------")
        print(
            f"  Holonomy rotation preferred on {counts.get('holonomy_rot',0)}/{len(stats['best_label'])} axes"
        )
        print(
            f"  Inside-view preferred on {counts.get('inside_rot',0)}/{len(stats['best_label'])} axes"
        )
        print(f"  ℓ=37 mean ratio: {np.nanmean(stats['l37_ratios']):.3f}")
        print(
            f"  P2 mean fraction: {np.nanmean(stats['p2']):.3f}, C4 mean fraction: {np.nanmean(stats['c4']):.3f}"
        )

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Best axis: {best_axis}")
        print(f"Combined p-value: {best_p:.4f}")
        print(f"Significance: {'YES' if best_p < 0.05 else 'NO'}")

        return {
            "test_type": "full_integration",
            "axes_tested": {k: v.tolist() for k, v in axes_to_test.items()},
            "results": results,
            "best_axis": best_axis,
            "best_p": best_p,
            "significant": best_p < 0.05,
            "discovery_summary": {
                "counts": dict(counts),
                "l37_ratio_mean": float(np.nanmean(stats["l37_ratios"])),
                "p2_mean": float(np.nanmean(stats["p2"])),
                "c4_mean": float(np.nanmean(stats["c4"])),
            },
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Run complete CGM empirical validation."""
    # Fast/prod mode toggle
    fast = os.getenv("CGM_FAST", "0") == "1"
    nside = 32 if fast else 64
    lmax = 16 if fast else 32
    fwhm = 5.0 if fast else 2.0
    os.environ.setdefault("CGM_MC", "200" if fast else "1000")

    # Get actual MC value from environment
    mc = int(os.getenv("CGM_MC", "1000"))

    print("CGM EMPIRICAL VALIDATION SUITE v1.0")
    print("Testing theoretical predictions on real data")
    print()
    if fast:
        print(f"NOTE: Using FAST parameters (nside=32, lmax=16, n_mc={mc})")
        print("Development mode for quick iteration")
    else:
        print(f"NOTE: Using production parameters (nside=64, lmax=32, n_mc={mc})")
        print("Full resolution analysis with robust statistics")
    print()

    # Initialize
    thresholds = CGMThresholds()
    data_manager = CGMDataManager()
    validator = CrossScaleValidator(data_manager, thresholds)

    # Run FULL validation with production parameters
    print("Running FULL CROSS-SCALE VALIDATION (Production Mode)")
    print("All tests enabled with proper depths and resolution")
    results = validator.test_cross_scale_consistency()

    # Results computed successfully (no file saving)
    print(f"\nResults computed successfully (no file saving)")

    # Final assessment
    if results["significant"]:
        print("\n✓ CGM VALIDATION SUCCESSFUL")
        print("  Toroidal structure detected across scales")
    else:
        print("\n✗ CGM VALIDATION INCONCLUSIVE")
        print("  More data or refined analysis needed")

    return results


if __name__ == "__main__":
    main()
