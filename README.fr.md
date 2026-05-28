# ROBUST-6G WP6 — Démonstrateur PHY

> **Déploiement en direct**
> · Interface graphique : <https://robust6g-demo.etis-lab.fr/>
> · URL de base de l'API : <https://robust6g-demo.etis-lab.fr/api/v1/>
> · Documentation API (Swagger UI) : <https://robust6g-demo.etis-lab.fr/api/v1/docs>
>
> L'API en direct sert les chemins canoniques `/api/v1/` décrits dans ce
> dépôt. Le conteneur Docker expose les mêmes chemins en local.

Démonstrateur de sécurité à la couche physique pour le projet ROBUST-6G,
Work Package 6. Fournit trois capacités de détection indépendantes
opérant sur un jeu de données de CSI mesurées (grille 24 × 24, ULA à
64 antennes, 2,61 GHz) :

| Capacité                       | Méthode                                          |
| ------------------------------ | ------------------------------------------------ |
| Détection de brouillage        | GLRT spatial + WL-CUSUM temporel                 |
| Détection d'usurpation         | AoA par Root-MUSIC + calibration anti-brouillage |
| Génération de clé secrète      | Réconciliation Polar-CRC + Davies-Meyer/AES-128  |

Ce dépôt fournit deux interfaces vers les mêmes moteurs sous-jacents :

- **API HTTP/JSON** (`api_server.py`) — pour l'usage orchestrateur /
  OpenC2. Spécifiée par [`openapi.yaml`](openapi.yaml).
- **Interface graphique** (`main.py`) — démonstrateur monopage NiceGUI
  (optionnel ; non requis pour l'intégration).

Un démonstrateur interactif distinct basé sur Tkinter est également
déployé à l'URL en direct ci-dessus pour l'exploration visuelle.

---

## Démarrage rapide (Docker)

```bash
git clone https://github.com/enseasolo/robust6g-wp6-phy-demonstrator-Prototype-4.git
cd robust6g-wp6-phy-demonstrator-Prototype-4
docker compose up -d --build
curl -s http://localhost:8000/api/v1/health | jq .
```

Sortie attendue :

```json
{
  "status": "ok",
  "version": "0.2.0",
  "engines": {
    "jamming_loaded": true,
    "spoofing_loaded": true,
    "skg_loaded": true,
    "dataset_path": "/app/dataset/data_ULA_skg.npz"
  }
}
```

Le premier appel à `/skg/generate` prend 25 à 40 s (réelle réconciliation
Polar-CRC en cours d'exécution). Les appels suivants empruntent le même
chemin chaud ; il n'y a pas de mise en chauffe par requête.

---

## API

Voir [`openapi.yaml`](openapi.yaml) pour le schéma complet. Cinq endpoints :

```
GET  /api/v1/health
GET  /api/v1/grid
POST /api/v1/jamming/detect
POST /api/v1/spoofing/detect
POST /api/v1/skg/generate
```

La documentation interactive (Swagger UI) est servie à
`http://localhost:8000/docs` quand le conteneur tourne, ou à
<https://robust6g-demo.etis-lab.fr/api/docs> sur le déploiement en direct.

### Préréglages du point de fonctionnement

L'interface graphique propose trois niveaux de SNR / intensité de
brouillage. Utilisez les mêmes valeurs numériques dans les appels API
pour que les résultats GUI et API soient comparables.

### Exemple — détection de brouillage

```bash
curl -s http://localhost:8000/api/v1/jamming/detect \
  -H 'Content-Type: application/json' \
  -d '{
    "user":   {"x": -0.7, "y": 3.6},
    "jammer": {"x": -0.2, "y": 2.9},
    "operating_point": {"snr_db": 25, "pj_dbm": 15}
  }' | jq .
```

### Exemple — génération de clé secrète

```bash
curl -s --max-time 90 http://localhost:8000/api/v1/skg/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "user":         {"x": -0.7, "y": 3.6},
    "eavesdropper": {"x": -1.0, "y": 2.1},
    "operating_point": {"snr_db": 25}
  }' | jq .
```

### Intégration en boucle fermée

Un exemple exécutable de boucle fermée MAPE-K utilisant les trois
endpoints est dans
[`examples/closed_loop_example.py`](examples/closed_loop_example.py).

---

## Organisation du dépôt

```
.
├── api_server.py            # Serveur FastAPI (point d'entrée)
├── main.py                  # Démonstrateur interactif NiceGUI (optionnel)
├── models/
│   ├── jamming_detector_glrt.py
│   ├── spoof_detector.py
│   └── skg_engine.py
├── skg_robust6G/            # Paquet de réconciliation Polar-CRC en amont
├── dataset/
│   └── data_ULA_skg.npz     # Jeu de données CSI (≈ 80 Mo, inclus)
├── assets/                  # Logos pour le front-end NiceGUI
├── examples/
│   └── closed_loop_example.py  # Exemple d'intégration MAPE-K
├── openapi.yaml             # Spécification OpenAPI 3.1
├── INTEGRATION.md           # Notes d'intégration pour les partenaires
├── AUTHORS.md               # Auteurs et contributeurs
├── CHANGELOG.md             # Historique des versions
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Configuration

Variables d'environnement (toutes optionnelles) :

| Variable               | Défaut                          | Rôle                                  |
| ---------------------- | ------------------------------- | ------------------------------------- |
| `PORT`                 | `8000`                          | Port d'écoute                         |
| `LOG_LEVEL`            | `INFO`                          | `DEBUG` pour journalisation verbeuse  |
| `ROBUST6G_DATASET`     | `/app/dataset/data_ULA_skg.npz` | Jeu de données SKG (Alice/Bob CSI)    |
| `ROBUST6G_AOA_DATASET` | `/app/data_ULA_all.npz`         | Jeu de données AoA (brouillage + usurpation) |
| `ROBUST6G_SKG_PKG`     | `/app/skg_robust6G`             | Répertoire du paquet SKG              |
| `ROBUST6G_ANT_POS`     | (non défini)                    | Géométrie d'antenne optionnelle       |

---

## Exécution hors Docker (développement)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python api_server.py            # API sur :8000
# dans un autre shell
python main.py                  # Démonstrateur NiceGUI sur :8080
```

Python 3.11 est la version supportée.

---

## Concurrence

L'endpoint SKG sérialise les requêtes derrière un verrou global — le
pipeline de réconciliation sous-jacent écrit un cache `bit_channel.npz`
et exécute un `chdir` dans son répertoire de paquet, donc les appels
concurrents se corromperaient mutuellement. Deux clients peuvent émettre
`/jamming/detect` ou `/spoofing/detect` en parallèle sans sérialisation.

---

## Citation

Une description des méthodes et du démonstrateur sera présentée à
**EuCNC & 6G Summit 2026, Málaga, Espagne**. Une fois publié, les
partenaires utilisant ce démonstrateur ou ses méthodes dans leurs propres
travaux sont invités à citer l'article correspondant. Une entrée BibTeX
sera ajoutée ici.

```bibtex
@inproceedings{robust6g_wp6_demo_2026,
  title     = {ROBUST-6G WP6: Physical-Layer Security Demonstrator
               for AoA Authentication, Jamming Detection and Secret Key Generation},
  author    = {Solomon Yese, Arsenia Chorti, Sara Berri,  Luan Chen, Luzzi, Laura,  Linda Senigagliesi, Sotiris Skaperas, Mamady Delamou,
                 and Passah, Angelo},
  booktitle = {Proc. EuCNC \& 6G Summit},
  address   = {M\'alaga, Spain},
  year      = {2026},
  note      = {to appear}
}
```

---

## Licence / attribution

Livrable du projet ROBUST-6G WP6. Licence finale à confirmer.

Voir [AUTHORS.md](AUTHORS.md) pour la liste complète des contributeurs.

---

## Contact

> **Solomon Yese** — `solomon.yese@ensea.fr`
> ENSEA / Laboratoire ETIS, CY Cergy Paris Université

Équipe du démonstrateur PHY ROBUST-6G WP6.
