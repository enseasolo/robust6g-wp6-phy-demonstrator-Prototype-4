# ROBUST-6G WP6 PHY Demonstrator — Integration Notes

> **Live preview:** <https://robust6g-demo.etis-lab.fr/> (GUI + API).
> The live API path prefix is currently `/api/` during the v0.2 rollout;
> the canonical `/api/v1/` paths in this document are guaranteed inside
> the Docker container shipped here.

This document accompanies `openapi.yaml` 



## 1. What the component does

A physical-layer security demonstrator for a 6G uplink scenario with a
64-element ULA. Built on a measured CSI dataset (24 × 24 placement grid,
spacing ≈ 0.12 m, carrier 2.61 GHz). Three independent capabilities, each
exposed as one HTTP endpoint:

| Capability            | Method                                          | Output                     |
| --------------------- | ----------------------------------------------- | -------------------------- |
| Jamming detection     | Spatial GLRT + temporal WL-CUSUM                | alarm + estimated location |
| Spoofing detection    | Root-MUSIC AoA + jammer-mitigation calibration  | verdict + ΔAoA + grid map  |
| Secret-key generation | Polar-CRC reconciliation + Davies-Meyer/AES-128 | 128-bit key + match stats  |

The endpoints are stateless: each request carries the full scene
(positions and operating point), the engine runs once, and the response
contains the result. 

## 2. Interface

### 2.1 Specification

The full machine-readable interface is in `openapi.yaml` (OpenAPI 3.1).
Validated with `openapi-spec-validator 0.8.x`.

Endpoints:

```
GET  /api/v1/health
GET  /api/v1/grid
POST /api/v1/jamming/detect
POST /api/v1/spoofing/detect
POST /api/v1/skg/generate
```

### 2.2 Request shape

All POST bodies share a position model:

```json
{
  "user":    { "x": -0.7, "y": 3.6 },
  "jammer":  { "x": -0.2, "y": 2.9 },
  "operating_point": { "snr_db": 25, "pj_dbm": 15 }
}
```

Coordinates are metres in the dataset frame. The valid bounding box is
returned by `GET /grid` and should be queried once at startup. Out-of-box
positions are snapped to the nearest dataset measurement point but a
warning is logged.

### 2.3 Operating-point presets

The interactive GUI exposes three SNR / jamming-intensity levels. For
results numerically comparable with the GUI, use these values:

| Level  |
| ------ | 
| Low    |
| Medium | 
| High   | 

The API will accept any numeric value, but values below 20 dB SNR have
not been validated against the demonstrator's headline KPIs.

### 2.4 Response shape (jamming example)

```json
{
  "alarm": true,
  "spatial_alarm": true,
  "temporal_alarm": true,
  "peak_score": 636.85,
  "threshold": 17.14,
  "cusum_g_t": 1013957.66,
  "jammer_estimated": { "x": -0.402, "y": 3.009 },
  "sinr_user_db": 19.94,
  "jn_user_db": -18.44,
  "sinr_drop_db": 0.06,
  "confidence": "Very High"
}
```

Spoofing and SKG response schemas are likewise in the OpenAPI document.

### 2.5 Latency

| Endpoint           | Typical wall-clock |
| ------------------ | ------------------ |
| `/health`, `/grid` | < 50 ms            |
| `/jamming/detect`  | 0.5–2 s            |
| `/spoofing/detect` | 1–3 s              |
| `/skg/generate`    | 25–40 s            |

Clients should set request timeouts of at least 60 s on `/skg/generate`.

### 2.6 Authentication

For lab integration: none (HTTP, plain JSON). If required, a static bearer-token header (`Authorization: Bearer <token>`)
can be added without changing the schema.

---

## 3. Delivery & Deployment

The demonstrator is delivered as a **public GitHub repository plus a
Dockerfile**. The partner builds the image locally; no outbound
dependencies at runtime, no VPN required.

### 3.1 Repository

```
https://github.com/enseasolo/robust6g-wp6-phy-demonstrator-Prototype-4
```

Public repository. The repository contains:

```
api_server.py            # FastAPI server (entry point)
main.py                  # Optional NiceGUI interactive demonstrator
models/                  # Detection & SKG engines
skg_robust6G/            # Upstream Polar-CRC reconciliation package
dataset/data_ULA_skg.npz # CSI dataset, ≈ 80 MB
examples/                # Closed-loop integration example
openapi.yaml             # This API's specification
Dockerfile
docker-compose.yml
requirements.txt
README.md / README.fr.md
INTEGRATION.md           # This document
AUTHORS.md / CHANGELOG.md
```

### 3.2 Build & run

```bash
git clone https://github.com/enseasolo/robust6g-wp6-phy-demonstrator-Prototype-4.git
cd robust6g-wp6-phy-demonstrator-Prototype-4
docker compose up -d --build
curl http://localhost:8000/api/v1/health
```

Interactive Swagger UI at `http://localhost:8000/docs`.

### 3.3 Resource requirements

| Resource          | Minimum | Recommended |
| ----------------- | ------- | ----------- |
| CPU cores         | 2       | 4           |
| RAM               | 4 GB    | 8 GB        |
| Disk              | 2 GB    | 5 GB        |
| Python (in image) | 3.11    | 3.11        |

GPU not required.

### 3.4 Configuration

Override via environment variables on the container:

| Variable               | Default                         | Purpose                               |
| ---------------------- | ------------------------------- | ------------------------------------- |
| `PORT`                 | `8000`                          | Listen port                           |
| `LOG_LEVEL`            | `INFO`                          | `DEBUG` for verbose engine logs       |
| `ROBUST6G_DATASET`     | `/app/dataset/data_ULA_skg.npz` | SKG dataset path (Alice/Bob CSI)      |
| `ROBUST6G_AOA_DATASET` | `/app/data_ULA_all.npz`         | AoA dataset path (jamming + spoofing) |
| `ROBUST6G_SKG_PKG`     | `/app/skg_robust6G`             | SKG package directory                 |

### 3.5 Datasets

The component requires **two** measured-CSI files:

- `data_ULA_skg.npz` — uplink + downlink CSI for Alice/Bob channel
  reciprocity. Contains arrays `csi_up`, `csi_dw`, `UE_positions_up`,
  `UE_positions_dw`.
- `data_ULA_all.npz` — single-direction CSI used by the AoA-based
  jamming and spoofing detectors. Contains `csi_UEs_all`,
  `UEs_positions`.

Both files are bundled inside the image at the paths shown above. Mount
your own files via the env vars if you need to swap them out.

---

## 4. Modelling guidance for the Security Ontology

Suggested mapping for any ontology. Refine as needed.

### 4.1 Asset class

`PhysicalLayerSecurityComponent` — composite component providing detection
and key-establishment capabilities at the OSI L1/L2 boundary.

### 4.2 Capabilities

| Capability            | Asset role         | OWL class hint             |
| --------------------- | ------------------ | -------------------------- |
| Jamming detection     | Detective control  | `JammingDetector`          |
| Spoofing detection    | Detective control  | `IdentitySpoofingDetector` |
| Secret-key generation | Preventive control | `KeyAgreementService`      |

### 4.3 Threats addressed

- **T-JAM** — RF/PHY jamming attack on the uplink.
- **T-SPF** — AoA-based identity spoofing of a legitimate UE.
- **T-EAV** — Passive eavesdropping during key establishment.

### 4.4 Operating-point parameters worth modelling

`snr_db`, `pj_dbm`, `n_steps` (jamming), `seed` (SKG).

---

## 5. OpenC2 mapping

Suggested actuator profile for any ad-hoc OpenC2 actuator.

### 5.1 Action verbs

| OpenC2 action | Target                            | Maps to                 |
| ------------- | --------------------------------- | ----------------------- |
| `query`       | `phy_demonstrator:health`         | `GET /health`           |
| `query`       | `phy_demonstrator:grid`           | `GET /grid`             |
| `scan`        | `phy_demonstrator:jamming`        | `POST /jamming/detect`  |
| `scan`        | `phy_demonstrator:spoofing`       | `POST /spoofing/detect` |
| `start`       | `phy_demonstrator:key_generation` | `POST /skg/generate`    |

### 5.2 Example command (jamming scan)

```json
{
  "action": "scan",
  "target": {
    "phy_demonstrator:jamming": {
      "user":   { "x": -0.7, "y": 3.6 },
      "jammer": { "x": -0.2, "y": 2.9 },
      "operating_point": { "snr_db": 25, "pj_dbm": 15 }
    }
  },
  "args": { "response_requested": "complete" }
}
```

The actuator translates this 1-to-1 to the JSON body of
`POST /api/v1/jamming/detect` and returns the response under
`results.phy_demonstrator:jamming`.

---


## 6. Contact

**Solomon Yese** — `solomon.yese@ensea.fr`
ENSEA / ETIS Laboratory, CY Cergy Paris Université

ROBUST-6G WP6 PHY Demonstrator team.
