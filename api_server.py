"""ROBUST-6G WP6 PHY Demonstrator — HTTP/JSON API server.

Thin FastAPI layer over the existing detection/SKG engines. Endpoints map
1-to-1 to those declared in `openapi.yaml`:

    GET  /api/v1/health
    GET  /api/v1/grid
    POST /api/v1/jamming/detect
    POST /api/v1/spoofing/detect
    POST /api/v1/skg/generate

Engines are loaded once at startup. The SKG endpoint serialises calls
behind a lock (the underlying reconciliation pipeline is not reentrant
because it chdir's and writes a `bit_channel.npz` cache file).
"""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# --- Engine imports --------------------------------------------------------
# These mirror what the NiceGUI demonstrator imports. Keep paths in sync if
# the package layout changes.
from models.jamming_detector_glrt import (  # type: ignore
    GLRTJammingDetector,
    GLRTDetectorConfig,
    build_psignal_grid,
    xy_to_grid_index,
)
from models.spoof_detector import SpoofDetector  # type: ignore
from models.skg_engine import SKGEngine  # type: ignore


# --- Configuration ---------------------------------------------------------

API_VERSION = "0.1.0"
API_PREFIX = "/api/v1"

# Two distinct datasets are required:
#   * SKG_DATASET_PATH  — used by the SKG engine for Alice/Bob reciprocity.
#                         Contains csi_up/csi_dw and UE_positions_up/dw.
#   * AOA_DATASET_PATH  — used by jamming + spoofing detectors. Contains
#                         csi_UEs_all and UEs_positions.
SKG_DATASET_PATH = os.environ.get(
    "ROBUST6G_DATASET",
    os.path.join(os.path.dirname(__file__), "dataset", "data_ULA_skg.npz"),
)
AOA_DATASET_PATH = os.environ.get(
    "ROBUST6G_AOA_DATASET",
    os.path.join(os.path.dirname(__file__), "data_ULA_all.npz"),
)
SKG_PKG_DIR = os.environ.get(
    "ROBUST6G_SKG_PKG",
    os.path.join(os.path.dirname(__file__), "skg_robust6G"),
)
def _find_antenna_positions() -> str | None:
    """Locate antenna_positions.npy without scanning the whole filesystem.

    Resolution order:
      1. $ROBUST6G_ANT_POS (explicit override) — even if it doesn't exist,
         we honour the env var and let SpoofDetector report the miss.
      2. Conventional locations next to the project root and AoA dataset.

    Returns the first existing path, or None if nothing was found.
    """
    explicit = os.environ.get("ROBUST6G_ANT_POS")
    if explicit:
        return explicit

    here = os.path.dirname(__file__)
    aoa_dir = os.path.dirname(AOA_DATASET_PATH)
    candidates = [
        os.path.join(aoa_dir, "antenna_positions.npy"),
        os.path.join(here, "antenna_positions.npy"),
        os.path.join(here, "dataset", "antenna_positions.npy"),
        os.path.join(here, "skg_robust6G", "antenna_positions.npy"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


ANT_POS_PATH = _find_antenna_positions()

logger = logging.getLogger("robust6g.api")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


# --- Engine container ------------------------------------------------------

class Engines:
    """Holds the long-lived engine instances loaded at startup."""

    def __init__(self) -> None:
        self.glrt: GLRTJammingDetector | None = None
        self.spoof: SpoofDetector | None = None
        self.skg: SKGEngine | None = None
        self.x_vals: np.ndarray | None = None
        self.y_vals: np.ndarray | None = None
        self.dataset_positions: np.ndarray | None = None
        # SKG is not reentrant — serialise.
        self.skg_lock = threading.Lock()
        self.ready = False

    def load(self) -> None:
        # ---- AoA dataset (jamming + spoofing) ----
        logger.info("Loading AoA dataset from %s", AOA_DATASET_PATH)
        aoa = np.load(AOA_DATASET_PATH)
        if "csi_UEs_all" in aoa.files:
            csi_all = aoa["csi_UEs_all"]
        elif "csi_up" in aoa.files:          # fall back if user pointed both vars at the same file
            csi_all = aoa["csi_up"]
        else:
            raise KeyError(
                f"AoA dataset {AOA_DATASET_PATH!r} has no recognised CSI key "
                f"(expected 'csi_UEs_all' or 'csi_up'). Keys present: {aoa.files}"
            )
        if "UEs_positions" in aoa.files:
            ue_pos = aoa["UEs_positions"]
        elif "UE_positions_up" in aoa.files:
            ue_pos = aoa["UE_positions_up"]
        else:
            raise KeyError(
                f"AoA dataset {AOA_DATASET_PATH!r} has no UE position key "
                f"(expected 'UEs_positions' or 'UE_positions_up'). Keys: {aoa.files}"
            )

        # GLRT engine
        ps_grid, x_vals, y_vals = build_psignal_grid(csi_all, ue_pos)
        self.glrt = GLRTJammingDetector(ps_grid, GLRTDetectorConfig())
        self.x_vals, self.y_vals = x_vals, y_vals
        self.dataset_positions = ue_pos[:, :2].astype(float)

        # Spoofing engine — runs an expensive prepare() once.
        logger.info("Preparing SpoofDetector (calibration + clean AoA map)...")
        if ANT_POS_PATH:
            logger.info("Using antenna_positions.npy at %s", ANT_POS_PATH)
        else:
            logger.warning(
                "antenna_positions.npy not found anywhere on the search path; "
                "SpoofDetector will fall back to a default array geometry. "
                "Set ROBUST6G_ANT_POS to silence this."
            )
        self.spoof = SpoofDetector(
            dataset_path=AOA_DATASET_PATH, ant_pos_path=ANT_POS_PATH
        )
        self.spoof.prepare()

        # ---- SKG dataset ----
        logger.info("Wiring SKGEngine to %s (dataset %s)", SKG_PKG_DIR, SKG_DATASET_PATH)
        self.skg = SKGEngine(skg_pkg_dir=SKG_PKG_DIR, dataset_path=SKG_DATASET_PATH)

        self.ready = True
        logger.info("All engines loaded.")


engines = Engines()


# --- Lifespan --------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        engines.load()
    except Exception:
        # Don't crash the server — /health will report degraded so the
        # orchestrator can detect the problem instead of getting connection
        # refused.
        logger.exception("Engine initialisation failed; running in degraded mode.")
    yield


app = FastAPI(
    title="ROBUST-6G WP6 PHY Demonstrator API",
    version=API_VERSION,
    lifespan=lifespan,
)


# --- Pydantic models matching openapi.yaml ---------------------------------

class Position2D(BaseModel):
    x: float
    y: float


class OperatingPoint(BaseModel):
    snr_db: float = 20.0
    pj_dbm: float = 15.0


class JammingRequest(BaseModel):
    user: Position2D
    jammer: Position2D | None = None
    operating_point: OperatingPoint = Field(default_factory=OperatingPoint)
    n_steps: int = Field(default=30, ge=1, le=200)


class JammingResponse(BaseModel):
    alarm: bool
    spatial_alarm: bool
    temporal_alarm: bool
    peak_score: float
    threshold: float
    cusum_g_t: float
    jammer_estimated: Position2D | None = None
    sinr_user_db: float
    jn_user_db: float
    sinr_drop_db: float
    confidence: str


class SpoofingRequest(BaseModel):
    user: Position2D
    spoofer: Position2D
    jammer: Position2D | None = None
    operating_point: OperatingPoint = Field(default_factory=OperatingPoint)


class SpoofingResponse(BaseModel):
    verdict: str  # SPOOF_FAIL | SPOOF_SUCCESS | AMBIGUOUS
    detected: bool
    delta_aoa_deg: float
    med_ae_deg: float
    mae_deg: float
    grid_success_rate_pct: float


class SkgRequest(BaseModel):
    user: Position2D
    eavesdropper: Position2D | None = None
    operating_point: OperatingPoint = Field(default_factory=OperatingPoint)
    seed: int = 42


class SkgEveBlock(BaseModel):
    key_hex: str | None = None
    pre_hash_match_pct: float | None = None
    post_hash_match_pct: float | None = None


class SkgResponse(BaseModel):
    reconciliation_pct: float
    error_prob: float
    n_recon_bits: int
    alice_key_hex: str
    alice_bob_match: bool
    eve: SkgEveBlock | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    engines: dict


class GridInfoResponse(BaseModel):
    bbox_m: dict
    n_dataset_points: int
    delta_m: float
    ula: dict


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: dict | None = None


# --- Error handling --------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc)
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message=str(exc) or exc.__class__.__name__,
        ).model_dump(),
    )


def _bad_request(error: str, message: str, **detail) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=ErrorResponse(error=error, message=message, detail=detail or None).model_dump(),
    )


def _require_engines() -> None:
    if not engines.ready or engines.glrt is None or engines.spoof is None or engines.skg is None:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error="ENGINES_NOT_READY",
                message="Engine initialisation failed or has not completed.",
            ).model_dump(),
        )


def _validate_in_bbox(p: Position2D) -> None:
    """Reject positions wildly outside the dataset bounding box.

    Positions slightly outside still get snapped by the engines, but a
    request like (1000, 1000) is almost certainly a caller bug.
    """
    assert engines.dataset_positions is not None
    pos = engines.dataset_positions
    pad_x = (pos[:, 0].max() - pos[:, 0].min()) * 0.5
    pad_y = (pos[:, 1].max() - pos[:, 1].min()) * 0.5
    if not (
        pos[:, 0].min() - pad_x <= p.x <= pos[:, 0].max() + pad_x
        and pos[:, 1].min() - pad_y <= p.y <= pos[:, 1].max() + pad_y
    ):
        raise _bad_request(
            "POSITION_OUT_OF_BBOX",
            f"Position ({p.x}, {p.y}) is far outside the placement region.",
            x=p.x,
            y=p.y,
        )


# --- Routes: meta ----------------------------------------------------------

@app.get(API_PREFIX + "/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if engines.ready else "degraded",
        version=API_VERSION,
        engines={
            "jamming_loaded": engines.glrt is not None,
            "spoofing_loaded": engines.spoof is not None,
            "skg_loaded": engines.skg is not None,
            "dataset_path": AOA_DATASET_PATH,
            "skg_dataset_path": SKG_DATASET_PATH,
        },
    )


@app.get(API_PREFIX + "/grid", response_model=GridInfoResponse, tags=["meta"])
def grid() -> GridInfoResponse:
    _require_engines()
    pos = engines.dataset_positions
    assert pos is not None and engines.glrt is not None
    return GridInfoResponse(
        bbox_m={
            "x_min": float(pos[:, 0].min()),
            "x_max": float(pos[:, 0].max()),
            "y_min": float(pos[:, 1].min()),
            "y_max": float(pos[:, 1].max()),
        },
        n_dataset_points=int(pos.shape[0]),
        delta_m=float(engines.glrt.cfg.delta_m),
        ula={
            # The visualization places the ULA to the right of the data area
            # at x = x_max + 1.5*(x_max - x_min). Report that to callers.
            "n_elements": 64,
            "x_m": float(pos[:, 0].max() + 1.5 * (pos[:, 0].max() - pos[:, 0].min())),
            "y_m": np.linspace(
                float(pos[:, 1].min()), float(pos[:, 1].max()), 6
            ).tolist(),
        },
    )


# --- Routes: jamming -------------------------------------------------------

def _confidence(peak: float, tau: float) -> str:
    if peak <= tau:
        return "None"
    ratio = peak / tau
    if ratio > 20:
        return "Very High"
    if ratio > 10:
        return "High"
    if ratio > 4:
        return "Medium"
    return "Low"


@app.post(
    API_PREFIX + "/jamming/detect",
    response_model=JammingResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["jamming"],
)
def jamming_detect(req: JammingRequest) -> JammingResponse:
    _require_engines()
    _validate_in_bbox(req.user)
    if req.jammer is not None:
        _validate_in_bbox(req.jammer)

    glrt = engines.glrt
    assert glrt is not None and engines.x_vals is not None and engines.y_vals is not None

    glrt.reset()

    snr_db = req.operating_point.snr_db
    pj_dbm = req.operating_point.pj_dbm

    if req.jammer is not None:
        jammer_rc = xy_to_grid_index(
            req.jammer.x, req.jammer.y, engines.x_vals, engines.y_vals
        )
    else:
        jammer_rc = None
        pj_dbm = float("-inf")

    last: dict[str, Any] | None = None
    for _ in range(req.n_steps):
        last = glrt.step(snr_db=snr_db, jammer_rc=jammer_rc, pj_dbm=pj_dbm)
    assert last is not None

    # Per-cell SINR / J/N at the user
    user_rc = xy_to_grid_index(req.user.x, req.user.y, engines.x_vals, engines.y_vals)
    ur, uc = user_rc
    p_sig = float(glrt.ps[ur, uc])
    p_jam_user = float(last["p_jam_grid_mW"][ur, uc])
    p_n_user = float(last["p_noise_grid_mW"][ur, uc])
    sinr_user_db = 10.0 * np.log10(
        max(p_sig, 1e-30) / max(p_jam_user + p_n_user, 1e-30)
    )
    jn_user_db = 10.0 * np.log10(max(p_jam_user, 1e-30) / max(p_n_user, 1e-30))

    jammer_est: Position2D | None = None
    if last["alarm"] and last["peak_r"] is not None:
        jammer_est = Position2D(
            x=float(engines.x_vals[last["peak_c"]]),
            y=float(engines.y_vals[last["peak_r"]]),
        )

    return JammingResponse(
        alarm=bool(last["alarm"]),
        spatial_alarm=bool(last["spatial_alarm"]),
        temporal_alarm=bool(last["temporal_alarm"]),
        peak_score=float(last["peak_val"]),
        threshold=float(last["tau"]),
        cusum_g_t=float(last["g_t"]) if np.isfinite(last["g_t"]) else 0.0,
        jammer_estimated=jammer_est,
        sinr_user_db=float(sinr_user_db),
        jn_user_db=float(jn_user_db),
        sinr_drop_db=float(last["sinr_drop_db"]),
        confidence=_confidence(float(last["peak_val"]), float(last["tau"])),
    )


# --- Routes: spoofing ------------------------------------------------------

def _verdict_to_enum(verdict_str: str) -> str:
    if "AMBIGUOUS" in verdict_str.upper() or "UNDEFINED" in verdict_str.upper():
        return "AMBIGUOUS"
    if "FAIL" in verdict_str.upper():
        return "SPOOF_FAIL"
    return "SPOOF_SUCCESS"


@app.post(
    API_PREFIX + "/spoofing/detect",
    response_model=SpoofingResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["spoofing"],
)
def spoofing_detect(req: SpoofingRequest) -> SpoofingResponse:
    _require_engines()
    _validate_in_bbox(req.user)
    _validate_in_bbox(req.spoofer)
    if req.jammer is not None:
        _validate_in_bbox(req.jammer)

    spoof = engines.spoof
    assert spoof is not None

    res = spoof.compute(
        user_xy=(req.user.x, req.user.y),
        spoofer_xy=(req.spoofer.x, req.spoofer.y),
        jammer_xy=(req.jammer.x, req.jammer.y) if req.jammer is not None else None,
        pj_dbm=req.operating_point.pj_dbm,
        snr_db=req.operating_point.snr_db,
    )

    sp = res["spoofer_result"]
    sm = res["spoof_map"]

    # defender's success rate = fraction of non-ambiguous cells where
    # detected=True (matches the corrected metric in the GUI).
    n_out = int(sm["n_total_out"])
    n_det = int(sm["n_detected_out"])
    success_pct = 100.0 * (n_det / n_out) if n_out > 0 else 0.0

    return SpoofingResponse(
        verdict=_verdict_to_enum(sp["verdict"]),
        detected=bool(sp["detected"]),
        delta_aoa_deg=float(sp["delta"]),
        med_ae_deg=float(res["medae_clean"]),
        mae_deg=float(res["mae_clean"]),
        grid_success_rate_pct=float(success_pct),
    )


# --- Routes: SKG -----------------------------------------------------------

@app.post(
    API_PREFIX + "/skg/generate",
    response_model=SkgResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["skg"],
)
def skg_generate(req: SkgRequest) -> SkgResponse:
    _require_engines()
    _validate_in_bbox(req.user)
    if req.eavesdropper is not None:
        _validate_in_bbox(req.eavesdropper)

    skg = engines.skg
    assert skg is not None

    eve_xy = (
        (req.eavesdropper.x, req.eavesdropper.y)
        if req.eavesdropper is not None
        else None
    )

    t0 = time.time()
    with engines.skg_lock:
        result = skg.run(
            alice_xy=(req.user.x, req.user.y),
            eve_xy=eve_xy,
            snr_db=req.operating_point.snr_db,
            seed=req.seed,
        )
    logger.info("SKG run completed in %.1f s", time.time() - t0)

    eve_block: SkgEveBlock | None = None
    if eve_xy is not None and result.get("eve_key_hex"):
        eve_block = SkgEveBlock(
            key_hex=result["eve_key_hex"],
            pre_hash_match_pct=result.get("eve_pre_recon_match_pct"),
            post_hash_match_pct=result.get("eve_key_match_pct"),
        )

    return SkgResponse(
        reconciliation_pct=float(result["reconciliation_pct"]),
        error_prob=float(result["error_prob"]),
        n_recon_bits=int(result["n_recon_bits"]),
        alice_key_hex=result["alice_key_hex"],
        alice_bob_match=bool(result["alice_bob_match"]),
        eve=eve_block,
    )


# --- Entry point -----------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
