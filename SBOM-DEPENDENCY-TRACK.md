# SBOM + Dependency-Track Integration

## Prezentare Generală

DeepGuard Pro folosește **CycloneDX** pentru generarea Software Bill of Materials (SBOM) și **OWASP Dependency-Track** pentru analiza automată a vulnerabilităților din dependențele proiectului.

## Arhitectură

```
DeepGuard Pro
     │
     ├── sbom.json (CycloneDX format)
     │        │
     │        ▼
     │   upload_sbom.py
     │        │
     │        ▼
     │   Dependency-Track API (port 8081)
     │        │
     │        ├── NVD CVE Database (auto-sync)
     │        ├── Vulnerability Analysis
     │        └── Risk Scoring
     │
     └── Dependency-Track UI (port 8082)
```

## Servicii Docker

| Serviciu | Container | Port | Rol |
|---|---|---|---|
| `dtrack-apiserver` | `deepguard_dtrack_api` | `8081` | API REST + analiză vulnerabilități |
| `dtrack-frontend` | `deepguard_dtrack_ui` | `8082` | Dashboard web |

## Pornire

```bash
docker compose up -d dtrack-apiserver dtrack-frontend
```

Prima pornire durează **5-15 minute** (descarcă baza de date NVD cu CVE-uri din 2002–prezent).

## Credențiale

| Câmp | Valoare |
|---|---|
| URL | http://localhost:8082 |
| Utilizator | `admin` |
| Parolă | `DeepGuard2025!` |

## Generare și Upload SBOM

### Pas 1: Generare SBOM (CycloneDX)

```bash
# Folosind cyclonedx-py
pip install cyclonedx-bom
cyclonedx-py requirements agents/coordinator/requirements.txt -o sbom.json --schema-version 1.4

# SAU cu syft (mai complet)
syft dir:. -o cyclonedx-json > sbom.json
```

### Pas 2: Upload și Analiză

```bash
python upload_sbom.py
```

**Output așteptat:**

```
[1] Logging in...          Logged in OK
[2] Getting API key...     Created new API key
[3] Uploading SBOM...      SBOM uploaded — processing token: <uuid>
[4] Waiting for analysis...  Waiting for vulnerability analysis. done!
[5] Fetching results...

=======================================================
  PROJECT: DeepGuard Pro v1.0.0
=======================================================
  Components:      133
  Vulnerabilities: 0
  Critical:        0
  High:            0
  Medium:          0
  Low:             0
=======================================================
  View at: http://localhost:8082/projects/<uuid>
```

## Configurare `upload_sbom.py`

```python
DTRACK_URL     = "http://127.0.0.1:8081"
PROJECT_NAME   = "DeepGuard Pro"
PROJECT_VERSION = "1.0.0"
SBOM_FILE      = "sbom.json"
DTRACK_USER    = "admin"
DTRACK_PASS    = "DeepGuard2025!"
```

## Rezultate Curent

Analiză din **11 Martie 2026**:

| Metric | Valoare |
|---|---|
| Componente analizate | 133 |
| Vulnerabilități critice | 0 |
| Vulnerabilități high | 0 |
| Vulnerabilități medium | 0 |
| Vulnerabilități low | 0 |
| Status | ✅ Curat |

## Integrare CI/CD

Pentru automatizare în pipeline:

```bash
# Re-generare SBOM la fiecare build
cyclonedx-py requirements requirements_master.txt -o sbom.json --schema-version 1.4

# Upload automat
docker run --rm --network host --workdir //app \
  -v "$(pwd)://app" \
  python:3.11-slim \
  sh -c "pip install requests -q && python upload_sbom.py"
```

## Format SBOM (CycloneDX 1.4)

```json
{
  "components": [
    {
      "bom-ref": "requirements-L22",
      "description": "requirements line 22: Flask==3.0.3",
      "externalReferences": [...],
      "name": "Flask",
      "version": "3.0.3",
      "type": "library"
    }
  ]
}
```

## Referințe

- [OWASP Dependency-Track](https://dependencytrack.org/)
- [CycloneDX Standard](https://cyclonedx.org/)
- [NVD CVE Database](https://nvd.nist.gov/)
