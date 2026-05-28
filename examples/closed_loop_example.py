"""
ROBUST-6G WP6 — Closed-loop integration example.

Implements a MAPE (Monitor / Analyse / Plan / Execute) security loop against
the ROBUST-6G WP6 PHY Demonstrator API. The loop:

  1. Queries the grid bounding box once (so we know where nodes can sit).
  2. MONITOR — runs jamming + spoofing detection at a baseline scene.
  3. ANALYSE — if jamming detected, re-runs with the suspected position
                to verify localisation; if spoofing detected, examines
                ΔAoA against the clean-baseline error floor.
  4. PLAN/EXECUTE — placeholder for the partner's RAN / network action,
                then rotates the secret key with /skg/generate.

API base URL — pick one of:

  Local Docker:    http://localhost:8000/api/v1
  Live deployment: https://robust6g-demo.etis-lab.fr/api/v1
                   (during v0.2 rollout the prefix may still be /api/;
                    consult the README for the current path.)

Operating-point presets (must match the GUI's Low/Medium/High):

  Low     snr_db = 20    pj_dbm = 7
  Medium  snr_db = 25    pj_dbm = 15
  High    snr_db = 30    pj_dbm = 24

Run:
    python3 closed_loop_example.py
"""

import time
import requests

# Choose your base URL --------------------------------------------------------
BASE = "http://localhost:8000/api/v1"
# BASE = "https://robust6g-demo.etis-lab.fr/api/v1"

TIMEOUT = 90               # seconds; covers /skg/generate cold call
LOOP_INTERVAL = 30         # seconds between monitoring cycles
OPERATING_POINT = {"snr_db": 25, "pj_dbm": 15}     # "Medium" preset


# --------------------------------------------------------------------------- #
# Low-level helpers                                                           #
# --------------------------------------------------------------------------- #
def call(method, path, **kwargs):
    """Thin wrapper: hits the API and returns parsed JSON, with sensible errors."""
    url = f"{BASE}{path}"
    r = requests.request(method, url, timeout=TIMEOUT, **kwargs)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# Stage 0 — Discover the grid                                                 #
# --------------------------------------------------------------------------- #
def discover_grid():
    """Fetch the placement bounding box once at startup."""
    print("[setup] fetching grid metadata...")
    g = call("GET", "/grid")
    bbox = g["bbox_m"]
    print(f"  X in [{bbox['x_min']:+.3f}, {bbox['x_max']:+.3f}] m")
    print(f"  Y in [{bbox['y_min']:+.3f}, {bbox['y_max']:+.3f}] m")
    print(f"  {g['n_dataset_points']} dataset points; "
          f"ULA: {g['ula']['n_elements']} antennas at x={g['ula']['x_m']} m")
    return g


# --------------------------------------------------------------------------- #
# Stage 1 — MONITOR                                                           #
# --------------------------------------------------------------------------- #
def monitor(user_xy, jammer_xy_guess, spoofer_xy_guess):
    """Run jamming + spoofing detection at the current best-guess scene."""
    print("[monitor] checking for jamming and spoofing...")

    jam = call("POST", "/jamming/detect", json={
        "user":   {"x": user_xy[0],         "y": user_xy[1]},
        "jammer": {"x": jammer_xy_guess[0], "y": jammer_xy_guess[1]},
        "operating_point": OPERATING_POINT,
    })

    spoof = call("POST", "/spoofing/detect", json={
        "user":    {"x": user_xy[0],          "y": user_xy[1]},
        "spoofer": {"x": spoofer_xy_guess[0], "y": spoofer_xy_guess[1]},
        "jammer":  {"x": jammer_xy_guess[0],  "y": jammer_xy_guess[1]},
        "operating_point": OPERATING_POINT,
    })

    print(f"  jamming  : alarm={jam['alarm']}, "
          f"peak={jam['peak_score']:.2f}, "
          f"SINR_user={jam['sinr_user_db']:+.2f} dB, "
          f"confidence={jam.get('confidence', '?')}")
    print(f"  spoofing : verdict={spoof['verdict']}, "
          f"dAoA={spoof['delta_aoa_deg']:.2f}° "
          f"(clean MedAE {spoof.get('med_ae_deg', float('nan')):.2f}°)")
    return jam, spoof


# --------------------------------------------------------------------------- #
# Stage 2 — ANALYSE jamming                                                   #
# --------------------------------------------------------------------------- #
def analyse_jamming(user_xy, jam):
    """If alarm fired, evaluate localisation quality."""
    if not jam["alarm"] or jam["jammer_estimated"] is None:
        return False

    est = jam["jammer_estimated"]
    print(f"[analyse:jam] GLRT estimate at ({est['x']:+.3f}, {est['y']:+.3f}); "
          f"peak {jam['peak_score']:.2f} vs threshold {jam['threshold']:.2f}")

    # A confident localisation means peak >> threshold and the spatial alarm
    # agrees with the temporal alarm.
    confirmed = (
        jam["spatial_alarm"]
        and jam.get("temporal_alarm", True)
        and jam["peak_score"] > 5.0 * jam["threshold"]
    )
    print(f"  confirmed={confirmed}")
    return confirmed


# --------------------------------------------------------------------------- #
# Stage 3 — ANALYSE spoofing                                                  #
# --------------------------------------------------------------------------- #
def analyse_spoofing(spoof):
    """Check whether the spoof is strong enough to act on."""
    delta = spoof["delta_aoa_deg"]
    medae = spoof.get("med_ae_deg", 1.0)
    confirmed = spoof["verdict"] == "SPOOF_FAIL" and delta > 0.3
    print(f"[analyse:spoof] dAoA={delta:.2f}° vs clean MedAE {medae:.2f}° "
          f"-> confirmed={confirmed}")
    return confirmed


# --------------------------------------------------------------------------- #
# Stage 4 — PLAN & EXECUTE                                                    #
# --------------------------------------------------------------------------- #
def execute_mitigation(jam_confirmed, spoof_confirmed):
    """Placeholder for the partner's own RAN / network actions."""
    actions = []
    if jam_confirmed:
        actions.append("notify_RAN(handover_or_beamsteer)")
    if spoof_confirmed:
        actions.append("drop_session(suspected_spoofer)")
    if actions:
        print("[execute] mitigation actions:", ", ".join(actions))
    return bool(actions)


def rotate_key(user_xy, eve_xy):
    """Re-run SKG and commit the new key if quality is sufficient."""
    print("[execute] rotating secret key (this takes ~30 s)...")
    skg = call("POST", "/skg/generate", json={
        "user":         {"x": user_xy[0], "y": user_xy[1]},
        "eavesdropper": {"x": eve_xy[0],  "y": eve_xy[1]},
        "operating_point": {"snr_db": OPERATING_POINT["snr_db"]},
        "seed": 42,
    })
    rate = skg["reconciliation_pct"]
    match = skg["alice_bob_match"]
    print(f"  reconciliation_pct={rate:.2f}%, alice_bob_match={match}")
    print(f"  alice_key={skg['alice_key_hex']}")
    if rate >= 99.0 and match:
        return True
    print("  key rotation result below threshold; deferring commit")
    return False


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #
def main():
    # Static topology for this example. In your prototype these come from
    # your radio / orchestrator / threat-intelligence feed.
    user_xy    = (-0.7,  3.6)
    jammer_xy  = (-0.3,  2.9)
    spoofer_xy = ( 0.7,  3.5)
    eve_xy     = (-1.0,  2.1)

    discover_grid()

    while True:
        try:
            jam, spoof = monitor(user_xy, jammer_xy, spoofer_xy)
            jam_conf = analyse_jamming(user_xy, jam)
            spoof_conf = analyse_spoofing(spoof)

            if execute_mitigation(jam_conf, spoof_conf):
                rotate_key(user_xy, eve_xy)

        except requests.HTTPError as e:
            print(f"[error] API returned {e.response.status_code}: "
                  f"{e.response.text[:200]}")
        except requests.RequestException as e:
            print(f"[error] connectivity issue: {e}")

        print(f"[idle] sleeping {LOOP_INTERVAL}s...\n")
        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
