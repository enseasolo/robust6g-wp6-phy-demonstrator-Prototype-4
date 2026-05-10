"""
AoA-based spoofing detector.

Wraps the pipeline from `aoajamming.py` so that user/jammer/spoofer positions
come from grid placements instead of hardcoded constants. First-launch
calibration and clean-AoA computation are cached to disk; subsequent launches
load the cached values immediately. The mitigated-AoA map is also cached per
(jammer xy, Pj, SNR) tuple so that repeat configurations are instantaneous.

The public entry point is `SpoofDetector.compute(...)` which is safe to call
from a worker thread - it does only numpy/scipy work and returns a dict that
the GUI thread can render directly.
"""

from __future__ import annotations

import os
import pickle
import numpy as np
from scipy.optimize import least_squares

# ----------------------------------------------------------------------------
# Physical / system constants (matching aoajamming.py)
# ----------------------------------------------------------------------------
CARRIER_FREQ = 2.61e9               # Hz
C = 299_792_458                     # m/s
WAVELENGTH = C / CARRIER_FREQ
ANTENNA_SPACING = 0.07              # m (verified from antenna_positions.npy)
TX_POWER_DBM = 15.0
P_TX_MW = 10 ** (TX_POWER_DBM / 10)
DETECTION_THRESHOLD = 0.3           # degrees


# ----------------------------------------------------------------------------
# Dataset loading (npz format used elsewhere in the demonstrator)
# ----------------------------------------------------------------------------
def _load_dataset(npz_path: str, ant_pos_path: str | None):
    """Load CSI / positions from .npz and antenna positions from .npy.

    Returns the same shape conventions used by aoajamming.py:
      csi : (N, Na, Ns) sorted by (y, x)
      xy  : (N, 2) sorted to match csi
      side : grid side length (24 here)
      spacing : grid spacing (m)
      ant_center : (3,) antenna array centre in metres
    """
    with np.load(npz_path) as data:
        raw_csi = np.asarray(data['csi_UEs_all'])           # (Na, Ns, N)
        pos = np.asarray(data['UEs_positions'])             # (N, 3)

    if raw_csi.shape[0] == 64 and raw_csi.shape[1] == 100:
        csi = raw_csi.transpose(2, 0, 1)
    else:
        csi = raw_csi
    n = csi.shape[0]
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError(f"Dataset has {n} points which is not a perfect square.")

    xy = pos[:, :2].astype(float)
    sort_idx = np.lexsort((xy[:, 0], xy[:, 1]))
    xy = xy[sort_idx]
    csi = csi[sort_idx]
    x_uniq = np.sort(np.unique(np.round(xy[:, 0], 6)))
    spacing = float(np.mean(np.diff(x_uniq)))

    if ant_pos_path and os.path.isfile(ant_pos_path):
        ant_mm = np.load(ant_pos_path)                      # (Na, 3) mm
        ant_center = np.mean(ant_mm, axis=0) / 1000.0       # -> m
    else:
        ant_center = np.array([0.0, 0.0, 1.0])
        print("[SpoofDetector] WARNING: antenna_positions.npy not found, "
              "using fallback centre [0, 0, 1].")

    return csi, xy, side, spacing, ant_center


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _ground_truth_aoa(positions: np.ndarray, ant_center: np.ndarray) -> np.ndarray:
    """AoA = arctan2(dx, dy) relative to antenna array centre, in degrees."""
    vec = positions[:, :2] - ant_center[:2]
    return np.degrees(np.arctan2(vec[:, 0], vec[:, 1]))


def _snap_to_grid(xy: np.ndarray, coord):
    d = np.sqrt((xy[:, 0] - coord[0]) ** 2 + (xy[:, 1] - coord[1]) ** 2)
    idx = int(np.argmin(d))
    return idx, (float(xy[idx, 0]), float(xy[idx, 1]))


# ----------------------------------------------------------------------------
# Calibration (Li et al. 2022) - identical math to aoajamming.compute_calibration
# ----------------------------------------------------------------------------
def _compute_calibration(csi_cal, pos_cal, ant_center):
    n_cal, na, ns = csi_cal.shape
    gt_rad = np.radians(_ground_truth_aoa(pos_cal, ant_center))

    res_phases = np.zeros((n_cal, ns))
    for s in range(n_cal):
        for nk in range(ns):
            r = [np.angle(csi_cal[s, ant, nk])
                 - (-2 * np.pi * ANTENNA_SPACING
                    * np.sin(gt_rad[s]) * ant / WAVELENGTH)
                 for ant in range(na)]
            res_phases[s, nk] = np.angle(np.mean(np.exp(1j * np.array(r))))
    mean_res = np.mean(res_phases, axis=0)

    def res_func(p):
        eg, vt, ep, zs, ec = p
        pred = np.array([
            np.arctan(eg * np.sin(nk * vt + ep)
                       / (np.cos(nk * vt) + 1e-10))
            + nk * zs + ec for nk in range(ns)
        ])
        return mean_res - pred

    sol = least_squares(res_func, [0., 0.01, 0., 0.001, 0.],
                         method='lm', max_nfev=1000)
    eg, vt, ep, zs, ec = sol.x

    xi_ant = np.zeros(na)
    for ant in range(na):
        ph = []
        for s in range(n_cal):
            ideal = -2 * np.pi * ANTENNA_SPACING * np.sin(gt_rad[s]) * ant / WAVELENGTH
            h_c = sum(
                csi_cal[s, ant, nk] * np.exp(
                    -1j * (np.arctan(eg * np.sin(nk * vt + ep)
                                      / (np.cos(nk * vt) + 1e-10))
                           + nk * zs + ec))
                for nk in range(ns)
            ) / ns
            ph.append(np.angle(np.exp(1j * (np.angle(h_c) - ideal))))
        xi_ant[ant] = np.angle(np.mean(np.exp(1j * np.array(ph))))

    return dict(eg=eg, vt=vt, ep=ep, zs=zs, ec=ec, xi_ant=xi_ant)


def _apply_calibration(H, cal):
    na, ns = H.shape
    eg, vt, ep, zs, ec, xi = (cal['eg'], cal['vt'], cal['ep'],
                               cal['zs'], cal['ec'], cal['xi_ant'])
    out = H.copy()
    for ant in range(na):
        for sub in range(ns):
            zIQ = np.arctan(eg * np.sin(sub * vt + ep)
                             / (np.cos(sub * vt) + 1e-10))
            out[ant, sub] *= np.exp(-1j * (zIQ + sub * zs + ec + xi[ant]))
    return out


# ----------------------------------------------------------------------------
# Root-MUSIC AoA estimator
# ----------------------------------------------------------------------------
def _root_music(H, subarray_size=16, num_sources=1):
    na, ns = H.shape
    n_sub = na - subarray_size + 1
    R = np.zeros((subarray_size, subarray_size), dtype=complex)
    J = np.eye(subarray_size)[::-1]
    for i in range(n_sub):
        Hs = H[i:i + subarray_size, :]
        Rs = Hs @ Hs.conj().T / ns
        R += Rs + J @ Rs.conj() @ J
    R /= 2 * n_sub
    _, vecs = np.linalg.eig(R)
    vecs = vecs[:, np.argsort(np.linalg.eigvals(R).real)[::-1]]
    noise = vecs[:, num_sources:]
    Cm = noise @ noise.conj().T
    M = subarray_size
    coeffs = np.zeros(2 * M - 1, dtype=complex)
    for k in range(-(M - 1), M):
        coeffs[k + (M - 1)] = np.sum(np.diag(Cm, k=abs(k)))
    roots = np.roots(coeffs[::-1])
    z = roots[np.argsort(np.abs(np.abs(roots) - 1))[0]]
    sin_t = np.clip(np.angle(z) * WAVELENGTH / (2 * np.pi * ANTENNA_SPACING), -1, 1)
    return float(np.degrees(np.arcsin(sin_t)))


def _aoa_map(csi_array, calib, progress_cb=None):
    n = csi_array.shape[0]
    aoas = np.full(n, np.nan)
    for k in range(n):
        try:
            aoas[k] = _root_music(_apply_calibration(csi_array[k], calib))
        except Exception:
            pass
        if progress_cb is not None and k % 32 == 0:
            progress_cb(k, n)
    if progress_cb is not None:
        progress_cb(n, n)
    return aoas


# ----------------------------------------------------------------------------
# Jammer + mitigation
# ----------------------------------------------------------------------------
def _dist(positions, xy_pt):
    x, y = xy_pt
    return np.sqrt((positions[:, 0] - x) ** 2 + (positions[:, 1] - y) ** 2)


def _signal_power_norm(csi):
    na, ns = csi.shape[1], csi.shape[2]
    return P_TX_MW * np.mean(np.sum(np.abs(csi) ** 2, axis=1), axis=1) / (na * ns)


def _jammer_rx_power(positions, jam_xy, p_j_mw, n_exp, d0):
    d = np.maximum(_dist(positions, jam_xy), d0)
    pl0 = (4 * np.pi * d0 * CARRIER_FREQ / C) ** 2
    return p_j_mw / (pl0 * (d / d0) ** n_exp)


def _add_jammer(csi_clean, positions, jam_xy, p_j_mw, n_exp, d0, snr_lin, rng):
    n, na, ns = csi_clean.shape
    p_sig = _signal_power_norm(csi_clean)
    p_jam = _jammer_rx_power(positions, jam_xy, p_j_mw, n_exp, d0)
    H = csi_clean.copy().astype(complex)
    for k in range(n):
        sj = np.sqrt(p_jam[k] / (na * ns) / 2)
        sn = (np.sqrt(p_sig[k] / snr_lin / (na * ns) / 2)
              if np.isfinite(snr_lin) else 0.0)
        H[k] += (rng.standard_normal((na, ns))
                 + 1j * rng.standard_normal((na, ns))) * sj
        if sn > 0:
            H[k] += (rng.standard_normal((na, ns))
                     + 1j * rng.standard_normal((na, ns))) * sn
    return H, p_jam


def _mitigate(H_jammed, positions, jam_xy, sinr_j_lin, snr_lin, n_exp, d0, eps=1e-6):
    n, na, ns = H_jammed.shape
    h_sq = np.mean(np.sum(np.abs(H_jammed) ** 2, axis=1), axis=1) / (na * ns)
    j_idx = int(np.argmin(_dist(positions, jam_xy)))
    p_sig_j = h_sq[j_idx]
    coeff = 1.0 / sinr_j_lin - (1.0 / snr_lin if np.isfinite(snr_lin) else 0.0)
    if coeff <= 0:
        return H_jammed.copy()
    p_jam_j = p_sig_j * coeff
    pl0 = (4 * np.pi * d0 * CARRIER_FREQ / C) ** 2
    p_j_mw = p_jam_j * na * ns * pl0
    p_jam_all = _jammer_rx_power(positions, jam_xy, p_j_mw, n_exp, d0)
    ratio = (p_jam_all / (na * ns)) / np.maximum(h_sq, 1e-30)
    alpha = np.sqrt(np.clip(1.0 - ratio, eps, 1.0))
    return H_jammed * alpha[:, np.newaxis, np.newaxis]


# ----------------------------------------------------------------------------
# Spoof evaluation
# ----------------------------------------------------------------------------
def _evaluate_spoof_map(xy, user_xy, gt_aoa, aoa_mit, mae_clean,
                         medae_clean, threshold):
    user_idx, user_snapped = _snap_to_grid(xy, user_xy)
    aoa_user_mit = aoa_mit[user_idx]
    gt_aoa_user = gt_aoa[user_idx]
    delta_mit = np.abs(aoa_mit - aoa_user_mit)
    delta_gt = np.abs(gt_aoa - gt_aoa_user)
    detected = delta_mit > threshold
    ambiguous = delta_gt < medae_clean
    return dict(
        user_idx=user_idx, user_snapped=user_snapped,
        aoa_user_mit=float(aoa_user_mit), gt_aoa_user=float(gt_aoa_user),
        delta_mit=delta_mit, delta_gt=delta_gt,
        detected=detected, ambiguous=ambiguous,
        medae_clean=medae_clean,
        n_detected=int(np.nansum(detected)),
        n_total=int(np.sum(~np.isnan(delta_mit))),
        n_detected_out=int(np.nansum(detected & ~ambiguous)),
        n_total_out=int(np.sum(~np.isnan(delta_mit) & ~ambiguous)),
    )


def _evaluate_single_spoofer(spoofer_xy, spoof_map, xy, medae_clean):
    sp_idx, sp_snapped = _snap_to_grid(xy, spoofer_xy)
    detected = bool(spoof_map['detected'][sp_idx])
    delta = float(spoof_map['delta_mit'][sp_idx])
    delta_gt_sp = float(spoof_map['delta_gt'][sp_idx])
    ambiguous = delta_gt_sp < medae_clean
    if ambiguous:
        verdict = 'UNDEFINED (inside ambiguous zone)'
    elif detected:
        verdict = 'SPOOF FAIL (detected)'
    else:
        verdict = 'SPOOF SUCCESS (missed)'
    return dict(sp_idx=sp_idx, sp_snapped=sp_snapped,
                detected=detected, delta=delta,
                ambiguous=ambiguous, verdict=verdict)


# ----------------------------------------------------------------------------
# Public detector class
# ----------------------------------------------------------------------------
class SpoofDetector:
    """AoA-based spoofing detector with on-disk caching.

    Parameters
    ----------
    dataset_path : path to the .npz with csi_UEs_all + UEs_positions
    ant_pos_path : path to antenna_positions.npy
    cache_dir : where to store calibration + clean AoA caches
    n_cal : number of calibration samples to use (matches aoajamming default)
    """

    def __init__(self, dataset_path: str, ant_pos_path: str | None = None,
                 cache_dir: str = 'cache', n_cal: int = 200):
        self.dataset_path = dataset_path
        self.ant_pos_path = ant_pos_path
        self.cache_dir = cache_dir
        self.n_cal = n_cal
        os.makedirs(self.cache_dir, exist_ok=True)

        # These are populated lazily by `prepare()`.
        self.csi = self.xy = None
        self.side = self.spacing = self.d0 = None
        self.ant_center = None
        self.calib = None
        self.aoa_clean = None
        self.gt_aoa = None
        self.mae_clean = None
        self.medae_clean = None
        self._prepared = False

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------
    def _cache(self, name):
        return os.path.join(self.cache_dir, name)

    def _load_or_compute_calibration(self, progress_cb=None):
        fpath = self._cache('calibration.pkl')
        if os.path.isfile(fpath):
            with open(fpath, 'rb') as f:
                return pickle.load(f)
        if progress_cb is not None:
            progress_cb('Computing calibration (one-time, ~1-2 min)...')
        calib = _compute_calibration(self.csi[:self.n_cal],
                                      self.xy[:self.n_cal],
                                      self.ant_center)
        with open(fpath, 'wb') as f:
            pickle.dump(calib, f)
        return calib

    def _load_or_compute_aoa_clean(self, progress_cb=None):
        fpath = self._cache('aoa_clean.npy')
        if os.path.isfile(fpath):
            return np.load(fpath)
        if progress_cb is not None:
            progress_cb('Computing clean AoA map (one-time, ~30 s)...')
        aoa = _aoa_map(self.csi, self.calib,
                        progress_cb=lambda k, n: progress_cb(
                            f'Clean AoA: {k}/{n}') if progress_cb else None)
        np.save(fpath, aoa)
        return aoa

    def _load_or_compute_aoa_mitigated(self, H_mit, jam_xy, pj_dbm, snr_db_str,
                                        progress_cb=None):
        # Cache key mirrors aoajamming.py's key.
        key = (f"aoa_mit_x{jam_xy[0]:+.4f}_y{jam_xy[1]:+.4f}"
               f"_pj{pj_dbm:.1f}_snr{snr_db_str}.npy")
        fpath = self._cache(key)
        if os.path.isfile(fpath):
            return np.load(fpath)
        if progress_cb is not None:
            progress_cb('Computing mitigated AoA map (~30 s)...')
        aoa = _aoa_map(H_mit, self.calib,
                        progress_cb=lambda k, n: progress_cb(
                            f'Mitigated AoA: {k}/{n}') if progress_cb else None)
        np.save(fpath, aoa)
        return aoa

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def prepare(self, progress_cb=None):
        """One-time heavy setup. Idempotent. Safe to call from a worker."""
        if self._prepared:
            return
        self.csi, self.xy, self.side, self.spacing, self.ant_center = \
            _load_dataset(self.dataset_path, self.ant_pos_path)
        self.d0 = self.spacing
        self.gt_aoa = _ground_truth_aoa(self.xy, self.ant_center)
        self.calib = self._load_or_compute_calibration(progress_cb)
        self.aoa_clean = self._load_or_compute_aoa_clean(progress_cb)
        valid = ~np.isnan(self.aoa_clean) & ~np.isnan(self.gt_aoa)
        errs = np.abs(self.aoa_clean[valid] - self.gt_aoa[valid])
        self.mae_clean = float(np.mean(errs))
        self.medae_clean = float(np.median(errs))
        self._prepared = True

    # ------------------------------------------------------------------
    # Per-run computation
    # ------------------------------------------------------------------
    def compute(self, user_xy, spoofer_xy,
                jammer_xy=None, pj_dbm=15.0, snr_db=15.0,
                n_exp=2.5, threshold=DETECTION_THRESHOLD,
                progress_cb=None):
        """Run the spoof-detection pipeline.

        Returns a dict containing everything the visualization panel needs:
          xy, side
          user_snapped, spoofer_snapped, jammer_snapped (or None)
          spoof_map (per-cell arrays for colour-coding)
          spoofer_result (single-cell verdict for the placed spoofer)
          metrics (mae_clean, medae_clean, threshold)
        """
        if not self._prepared:
            self.prepare(progress_cb)

        snr_lin = float('inf') if not np.isfinite(snr_db) else 10 ** (snr_db / 10)
        snr_db_str = 'inf' if not np.isfinite(snr_db) else f'{int(snr_db)}'
        p_j_mw = 10 ** (pj_dbm / 10)

        # Snap inputs to grid.
        _, user_snapped = _snap_to_grid(self.xy, user_xy)
        _, spoofer_snapped = _snap_to_grid(self.xy, spoofer_xy)
        if jammer_xy is not None:
            _, jam_snapped = _snap_to_grid(self.xy, jammer_xy)
        else:
            # No jammer: pick a "neutral" position for the simulator and use
            # very low Pj so the channel is essentially clean.
            jam_snapped = (float(self.xy[0, 0]), float(self.xy[0, 1]))
            p_j_mw = 1e-12

        # Simulate jammer effect (matches aoajamming.add_jammer).
        rng = np.random.default_rng(42)
        H_jammed, p_jam = _add_jammer(self.csi, self.xy, jam_snapped,
                                       p_j_mw, n_exp, self.d0, snr_lin, rng)

        # SINR at jammer cell -> feeds mitigation.
        j_idx = int(np.argmin(_dist(self.xy, jam_snapped)))
        na, ns = self.csi.shape[1], self.csi.shape[2]
        p_sig_j = _signal_power_norm(H_jammed[[j_idx]])[0]
        p_noise_j = p_sig_j / snr_lin if np.isfinite(snr_lin) else 0.0
        sinr_j_lin = p_sig_j / (p_jam[j_idx] / (na * ns) + p_noise_j + 1e-30)

        # Mitigate.
        H_mit = _mitigate(H_jammed, self.xy, jam_snapped, sinr_j_lin,
                           snr_lin, n_exp, self.d0)

        # Mitigated AoA map (cached per jammer config).
        aoa_mit = self._load_or_compute_aoa_mitigated(
            H_mit, jam_snapped, pj_dbm, snr_db_str, progress_cb)

        # Spoof evaluation.
        spoof_map = _evaluate_spoof_map(
            self.xy, user_snapped, self.gt_aoa, aoa_mit,
            self.mae_clean, self.medae_clean, threshold)
        spoofer_result = _evaluate_single_spoofer(
            spoofer_snapped, spoof_map, self.xy, self.medae_clean)

        return dict(
            xy=self.xy, side=self.side,
            ant_center=self.ant_center,
            user_snapped=user_snapped,
            spoofer_snapped=spoofer_snapped,
            jam_snapped=jam_snapped if jammer_xy is not None else None,
            spoof_map=spoof_map,
            spoofer_result=spoofer_result,
            mae_clean=self.mae_clean,
            medae_clean=self.medae_clean,
            threshold=threshold,
        )
