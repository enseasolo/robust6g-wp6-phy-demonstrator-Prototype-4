# ROBUST-6G WP6 — PHY Demonstrator

Physical-layer security demonstrator for the ROBUST-6G project, Work
Package 6. Provides three independent detection capabilities operating on
a measured-CSI dataset (24 × 24 grid, 64-element ULA, 2.61 GHz):

| Capability                | Method                                           |
|---------------------------|--------------------------------------------------|
| Jamming detection         | Spatial GLRT + temporal WL-CUSUM                 |
| Spoofing detection        | Root-MUSIC AoA + jammer-mitigation calibration   |
| Secret-key generation     | Polar-CRC reconciliation + Davies-Meyer/AES-128  |

This repository ships two front-ends to the same underlying engines:

* **HTTP/JSON API** (`api_server.py`) — for orchestrator / OpenC2 use.
  Specified by [`openapi.yaml`](./openapi.yaml).
* **Interactive UI** (`main.py`) — NiceGUI single-page demonstrator
  (optional; not required for integration).

---

## Quick start (Docker)

```bash
git clone https://github.com/<org>/robust6g-wp6-phy-demonstrator
cd robust6g-wp6-phy-demonstrator
docker compose up -d --build
curl -s http://localhost:8000/api/v1/health | jq .
```

Expected output:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "engines": {
    "jamming_loaded": true,
    "spoofing_loaded": true,
    "skg_loaded": true,
    "dataset_path": "/app/dataset/data_ULA_skg.npz"
  }
}
```

First call to `/skg/generate` takes 25–40 s (real Polar-CRC reconciliation
running). Subsequent calls hit the same hot path; there is no per-request
warm-up.

---

## API

See [`openapi.yaml`](./openapi.yaml) for the full schema. Five endpoints:

```
GET  /api/v1/health
GET  /api/v1/grid
POST /api/v1/jamming/detect
POST /api/v1/spoofing/detect
POST /api/v1/skg/generate
```

Interactive docs (Swagger UI) are served at `http://localhost:8000/docs`
when the container is running.

### Example: jamming detection

```bash
curl -s http://localhost:8000/api/v1/jamming/detect \
  -H 'Content-Type: application/json' \
  -d '{
    "user":   {"x": -0.7, "y": 3.6},
    "jammer": {"x": -0.2, "y": 2.9},
    "operating_point": {"snr_db": 20, "pj_dbm": 15}
  }' | jq .
```

### Example: secret-key generation

```bash
curl -s --max-time 90 http://localhost:8000/api/v1/skg/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "user":         {"x": -0.7, "y": 3.6},
    "eavesdropper": {"x": -1.0, "y": 2.1},
    "operating_point": {"snr_db": 20}
  }' | jq .
```

---

## Repository layout

```
.
├── api_server.py            # FastAPI server (entry point)
├── main.py                  # NiceGUI interactive demonstrator (optional)
├── models/
│   ├── jamming_detector_glrt.py
│   ├── spoof_detector.py
│   └── skg_engine.py
├── skg_robust6G/            # Upstream Polar-CRC reconciliation package
├── dataset/
│   └── data_ULA_skg.npz     # CSI dataset (≈ 80 MB, included)
├── assets/                  # Logos for the NiceGUI front-end
├── openapi.yaml             # OpenAPI 3.1 specification
├── INTEGRATION.md           # Integration notes for partners
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Configuration

Environment variables (all optional):

| Variable                | Default                                    | Purpose                              |
|-------------------------|--------------------------------------------|--------------------------------------|
| `PORT`                  | `8000`                                     | Listen port                          |
| `LOG_LEVEL`             | `INFO`                                     | `DEBUG` for verbose engine logs      |
| `ROBUST6G_DATASET`      | `/app/dataset/data_ULA_skg.npz`            | SKG dataset (Alice/Bob CSI)          |
| `ROBUST6G_AOA_DATASET`  | `/app/data_ULA_all.npz`                    | AoA dataset (jamming + spoofing CSI) |
| `ROBUST6G_SKG_PKG`      | `/app/skg_robust6G`                        | SKG package directory                |
| `ROBUST6G_ANT_POS`      | (unset)                                    | Optional antenna geometry override   |

---

## Running without Docker (development)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python api_server.py            # API on :8000
# in another shell
python main.py                  # NiceGUI demonstrator on :8080
```

Python 3.11 is the supported version.

---

## Concurrency

The SKG endpoint serialises behind a process-wide lock — the underlying
reconciliation pipeline writes a `bit_channel.npz` cache and `chdir`s into
its package directory, so concurrent calls would corrupt each other. Two
clients can issue `/jamming/detect` or `/spoofing/detect` concurrently
without serialisation.

---

## Licence / attribution

Project deliverable for ROBUST-6G WP6. Final licence to be confirmed.

---

## Contact

ROBUST-6G WP6 PHY Demonstrator team — `<fill in>`
