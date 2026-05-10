"""
Spatial GLRT + temporal WL-CUSUM jamming detector.

Python port of `generate_realizations_baseline_jammed_full.m`. The detector
operates on a 2D Psignal grid derived from the measured CSI dataset and
returns, per snapshot:

    - alarm      : True if jamming is detected this step
    - peak_r/c   : grid indices of the estimated jammer location (None if no alarm)
    - delta_dB   : estimated SINR drop at the detected peak (dB), 0 if no alarm
    - peak_val   : raw GLRT peak score (always returned, for monitoring)

Convention: P_signal grid is indexed [r, c] where r corresponds to ascending
y-values and c to ascending x-values, matching the MATLAB npz loader.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from scipy.special import erfcinv


# -----------------------------------------------------------------------------
# Helpers to build the Psignal grid from the npz dataset
# -----------------------------------------------------------------------------

def build_psignal_grid(csi_all: np.ndarray, ue_positions: np.ndarray,
                       ptx_dbm: float = 15.0):
    """Build the 24x24 mean-|H|^2 grid in mW from the dataset.

    Parameters
    ----------
    csi_all : (Nant, Nsub, Ns) complex array
    ue_positions : (Ns, 3) float array, columns (x, y, z)
    ptx_dbm : transmit power of the legitimate UE in dBm.

    Returns
    -------
    Ps : (R, L) float array, P_signal in mW. R = number of unique y values,
         L = number of unique x values. Indexed Ps[r=y_idx, c=x_idx].
    x_vals, y_vals : 1D arrays of unique x/y coords (ascending), in metres.
    """
    p_per_ue = np.mean(np.abs(csi_all) ** 2, axis=(0, 1))  # (Ns,)
    ptx_mw = 10.0 ** (ptx_dbm / 10.0)
    p_per_ue = ptx_mw * p_per_ue

    x = ue_positions[:, 0]
    y = ue_positions[:, 1]
    x_vals = np.unique(x)
    y_vals = np.unique(y)

    ps = np.full((len(y_vals), len(x_vals)), np.nan)
    for k in range(len(p_per_ue)):
        c_idx = int(np.argmin(np.abs(x_vals - x[k])))
        r_idx = int(np.argmin(np.abs(y_vals - y[k])))
        ps[r_idx, c_idx] = p_per_ue[k]

    if np.any(~np.isfinite(ps)):
        raise ValueError("Could not map all UE positions to a complete 2D grid.")
    return ps, x_vals, y_vals


def xy_to_grid_index(x_m: float, y_m: float,
                     x_vals: np.ndarray, y_vals: np.ndarray):
    """Snap a metric (x, y) position to its (r, c) grid indices."""
    c_idx = int(np.argmin(np.abs(x_vals - x_m)))
    r_idx = int(np.argmin(np.abs(y_vals - y_m)))
    return r_idx, c_idx


# -----------------------------------------------------------------------------
# Detector
# -----------------------------------------------------------------------------

@dataclass
class GLRTDetectorConfig:
    alpha: float = 2.5            # path-loss exponent (per PDF defaults)
    pfa: float = 0.01             # spatial false-alarm probability
    W_temporal: int = 5           # WL-CUSUM window
    use_temporal: bool = True
    sigma_floor: float = 0.35     # noise floor in z-map units
    eps0: float = 1e-12
    delta_m: float = 0.1215       # grid spacing (m)
    f_hz: float = 2.61e9          # carrier frequency
    temporal_tau: float = 4.0     # WL-CUSUM threshold


class GLRTJammingDetector:
    """Spatial GLRT detector with optional WL-CUSUM temporal smoothing.

    Mirrors `run_glrt_template_detector_pure` + `wl_cusum_stat` from the
    reference MATLAB implementation.
    """

    def __init__(self, ps_grid: np.ndarray, cfg: Optional[GLRTDetectorConfig] = None):
        self.cfg = cfg or GLRTDetectorConfig()
        self.ps = np.asarray(ps_grid, dtype=float)
        self.R, self.L = self.ps.shape

        # Geometry & frequency
        c0 = 299_792_458.0
        d0 = self.cfg.delta_m
        self.pl0_db = 20 * np.log10(4 * np.pi * d0 * self.cfg.f_hz / c0)
        rr, cc = np.meshgrid(np.arange(self.R), np.arange(self.L), indexing='ij')
        self.RR = rr
        self.CC = cc

        # Spatial threshold (Bonferroni-corrected)
        m = self.R * self.L
        z_thr = -np.sqrt(2.0) * erfcinv(2.0 * (1.0 - self.cfg.pfa / m))
        self.tau_detector = z_thr ** 2

        # Pre-compute the unit-norm template for every candidate (r0, c0).
        # Cost is R*L*R*L = ~330k cells * 576 templates = ~190M ops, but each
        # template just needs the distance map - fast in numpy.
        self._templates = self._precompute_templates()

        # Reset state
        self.peak_history: List[float] = []

    def _precompute_templates(self) -> np.ndarray:
        """Pre-compute (R*L, R*L) matrix of unit-norm path-loss templates.

        Row index = candidate (r0, c0) flattened; columns = grid cells flattened.
        This makes per-step inference one matrix-vector product.
        """
        R, L = self.R, self.L
        d0 = self.cfg.delta_m
        alpha = self.cfg.alpha
        templates = np.empty((R * L, R * L), dtype=float)
        for r0 in range(R):
            for c0 in range(L):
                d = np.sqrt(((self.RR - r0) * d0) ** 2 +
                            ((self.CC - c0) * d0) ** 2)
                d = np.maximum(d, d0)
                t = d ** (-alpha)
                t = t.reshape(-1)
                t = t / max(np.linalg.norm(t), self.cfg.eps0)
                templates[r0 * L + c0] = t
        return templates

    # ------------------------------------------------------------------
    # Reference (clean) model
    # ------------------------------------------------------------------
    def learn_normal_model(self, snr_db: float):
        """Clean reference P_obs map under the chosen SNR setting."""
        ps = self.ps
        if np.isinf(snr_db):
            pn = np.zeros_like(ps)
        else:
            pn = ps / (10.0 ** (snr_db / 10.0))
        mu = 10.0 * np.log10((ps + self.cfg.eps0) / (pn + self.cfg.eps0))
        sig = self.cfg.sigma_floor * np.ones_like(ps)
        return mu, sig

    # ------------------------------------------------------------------
    # Build the observed P map (jammed or baseline)
    # ------------------------------------------------------------------
    def observed_p_map(self, snr_db: float, jammer_rc: Optional[Tuple[int, int]],
                      pj_dbm: float):
        """Return P_obs (dB) for one snapshot.

        If jammer_rc is None or pj_dbm is -inf -> baseline (no jamming).
        """
        ps = self.ps
        if np.isinf(snr_db):
            pn = np.zeros_like(ps)
        else:
            pn = ps / (10.0 ** (snr_db / 10.0))

        if jammer_rc is None or not np.isfinite(pj_dbm):
            p_jam = np.zeros_like(ps)
        else:
            rJ, cJ = jammer_rc
            d = np.sqrt(((self.RR - rJ) * self.cfg.delta_m) ** 2 +
                        ((self.CC - cJ) * self.cfg.delta_m) ** 2)
            d = np.maximum(d, self.cfg.delta_m)
            pl_db = self.pl0_db + 10 * self.cfg.alpha * np.log10(d / self.cfg.delta_m)
            p_jam = 10.0 ** ((pj_dbm - pl_db) / 10.0)

        p_obs_db = 10.0 * np.log10((ps + self.cfg.eps0) /
                                    (pn + p_jam + self.cfg.eps0))
        return p_obs_db, p_jam, pn

    # ------------------------------------------------------------------
    # GLRT step
    # ------------------------------------------------------------------
    def _glrt(self, z_map: np.ndarray, sig_used: np.ndarray):
        a = (-z_map).reshape(-1)
        a[~np.isfinite(a)] = 0.0
        # Score for every candidate location: max(0, t·a)^2
        scores = self._templates @ a   # (R*L,)
        beta_hat = np.maximum(scores, 0.0)
        score_map = (beta_hat ** 2).reshape(self.R, self.L)

        idx = int(np.argmax(score_map))
        peak_r, peak_c = divmod(idx, self.L)
        peak_val = float(score_map[peak_r, peak_c])
        is_detected = peak_val > self.tau_detector

        # Reconstruct delta in dB at the peak
        t_best = self._templates[idx].reshape(self.R, self.L)
        beta_best = float(beta_hat[idx])
        delta_z = beta_best * t_best
        delta_db = delta_z * sig_used

        return {
            'score_map': score_map,
            'peak_r': peak_r,
            'peak_c': peak_c,
            'peak_val': peak_val,
            'is_detected': is_detected,
            'delta_db_peak': float(delta_db[peak_r, peak_c]),
        }

    # ------------------------------------------------------------------
    # WL-CUSUM
    # ------------------------------------------------------------------
    def _wl_cusum(self):
        x = np.asarray(self.peak_history, dtype=float)
        t = len(x)
        w_max = min(self.cfg.W_temporal, t)
        g_t = 0.0
        for w in range(1, w_max + 1):
            s = float(x[t - w:t].sum())
            g_w = (max(0.0, s) ** 2) / (2.0 * w)
            if g_w > g_t:
                g_t = g_w
        return g_t

    # ------------------------------------------------------------------
    # Per-step entry point
    # ------------------------------------------------------------------
    def step(self, snr_db: float, jammer_rc: Optional[Tuple[int, int]],
             pj_dbm: float):
        """Run one detection step. Returns a dict of results.

        Note: we operate on the SAME generated P_obs (jammed or not) that the
        attacker would produce, then ask the detector "do you see jamming?".
        """
        mu, sig = self.learn_normal_model(snr_db)
        p_obs, p_jam, p_noise = self.observed_p_map(snr_db, jammer_rc, pj_dbm)

        z = (p_obs - mu) / (sig + self.cfg.eps0)
        z[~np.isfinite(z)] = 0.0

        det = self._glrt(z, sig)
        self.peak_history.append(det['peak_val'])

        temporal_alarm = False
        g_t = float('nan')
        if self.cfg.use_temporal:
            g_t = self._wl_cusum()
            temporal_alarm = g_t > self.cfg.temporal_tau

        alarm = bool(det['is_detected'] or temporal_alarm)
        return {
            'alarm': alarm,
            'spatial_alarm': det['is_detected'],
            'temporal_alarm': temporal_alarm,
            'peak_r': det['peak_r'] if alarm else None,
            'peak_c': det['peak_c'] if alarm else None,
            'peak_val': det['peak_val'],
            'tau': self.tau_detector,
            'g_t': g_t,
            'sinr_drop_db': det['delta_db_peak'] if alarm else 0.0,
            'p_jam_grid_mW': p_jam,
            'p_noise_grid_mW': p_noise,
        }

    def reset(self):
        self.peak_history = []
