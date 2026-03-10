#!/usr/bin/env python3
"""Upload SBOM to Dependency-Track and show vulnerability summary."""

import requests
import base64
import json
import sys
import time

DTRACK_URL = "http://127.0.0.1:8081"
USERNAME = "admin"
PASSWORD = "DeepGuard2025!"
PROJECT_NAME = "DeepGuard Pro"
PROJECT_VERSION = "1.0.0"
SBOM_FILE = "sbom.json"

def login():
    r = requests.post(f"{DTRACK_URL}/api/v1/user/login",
                      data={"username": USERNAME, "password": PASSWORD},
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
    if r.status_code == 200 and r.text not in ("FORCE_PASSWORD_CHANGE", "INVALID_CREDENTIALS"):
        print(f"  Logged in OK")
        return r.text.strip()
    print(f"  Login failed: {r.text}")
    sys.exit(1)

def get_or_create_api_key(token):
    headers = {"Authorization": f"Bearer {token}"}
    # Get existing teams
    r = requests.get(f"{DTRACK_URL}/api/v1/team", headers=headers)
    teams = r.json()
    # Find Administrators team
    admin_team = next((t for t in teams if t["name"] == "Administrators"), teams[0])
    team_uuid = admin_team["uuid"]

    # Check existing keys (v4.13+ returns only maskedKey for existing keys)
    existing = admin_team.get("apiKeys", [])
    if existing and existing[0].get("key"):
        print(f"  Using existing API key")
        return existing[0]["key"]

    # Create new key (full key returned only on creation)
    r = requests.put(f"{DTRACK_URL}/api/v1/team/{team_uuid}/key", headers=headers)
    key = r.json()["key"]
    print(f"  Created new API key")
    return key

def upload_sbom(api_key):
    with open(SBOM_FILE, "rb") as f:
        bom_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "projectName": PROJECT_NAME,
        "projectVersion": PROJECT_VERSION,
        "autoCreate": True,
        "bom": bom_b64
    }

    r = requests.put(f"{DTRACK_URL}/api/v1/bom",
                     headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
                     json=payload)
    if r.status_code in (200, 201):
        token = r.json().get("token")
        print(f"  SBOM uploaded — processing token: {token}")
        return token
    print(f"  Upload failed: {r.status_code} {r.text}")
    sys.exit(1)

def wait_for_processing(api_key, token, timeout=120):
    print("  Waiting for vulnerability analysis", end="", flush=True)
    headers = {"X-Api-Key": api_key}
    for _ in range(timeout // 5):
        r = requests.get(f"{DTRACK_URL}/api/v1/bom/token/{token}", headers=headers)
        if r.json().get("processing") == False:
            print(" done!")
            return
        print(".", end="", flush=True)
        time.sleep(5)
    print(" timeout!")

def wait_for_metrics(api_key, uuid, timeout=300):
    """Wait until Dependency Track has computed project metrics (NVD sync may be needed)."""
    headers = {"X-Api-Key": api_key}
    print("  Waiting for metrics to be computed", end="", flush=True)
    for _ in range(timeout // 10):
        r = requests.get(f"{DTRACK_URL}/api/v1/metrics/project/{uuid}/current", headers=headers)
        if r.status_code == 200 and r.text.strip():
            data = r.json()
            # metrics are ready when 'components' is populated
            if data.get("components", 0) > 0:
                print(" done!")
                return data
        print(".", end="", flush=True)
        time.sleep(10)
    print(" timeout! (NVD sync may still be in progress — re-run in 10 min)")
    return {}

def get_summary(api_key, project_name, version):
    headers = {"X-Api-Key": api_key}
    # Find project
    r = requests.get(f"{DTRACK_URL}/api/v1/project",
                     headers=headers,
                     params={"name": project_name, "version": version})
    projects = r.json()
    if not projects:
        print("Project not found")
        return
    project = projects[0]
    uuid = project["uuid"]

    # Use dedicated metrics endpoint (more reliable than project.metrics)
    metrics = wait_for_metrics(api_key, uuid)

    print("\n" + "="*55)
    print(f"  PROJECT: {project_name} v{version}")
    print("="*55)
    print(f"  Components:      {metrics.get('components', 'N/A')}")
    print(f"  Vulnerabilities: {metrics.get('vulnerabilities', 'N/A')}")
    print(f"  Critical:        {metrics.get('critical', 'N/A')}")
    print(f"  High:            {metrics.get('high', 'N/A')}")
    print(f"  Medium:          {metrics.get('medium', 'N/A')}")
    print(f"  Low:             {metrics.get('low', 'N/A')}")
    print(f"  Unassigned:      {metrics.get('unassigned', 'N/A')}")
    print(f"  Last updated:    {metrics.get('lastOccurrence', 'N/A')}")
    print("="*55)
    print(f"\n  View at: http://localhost:8082/projects/{uuid}")

    # Get top vulnerabilities
    r2 = requests.get(f"{DTRACK_URL}/api/v1/vulnerability/project/{uuid}",
                      headers=headers,
                      params={"pageSize": 10, "sortName": "severity", "sortOrder": "desc"})
    if r2.status_code == 200:
        vulns = r2.json()
        if vulns:
            print(f"\n  TOP VULNERABILITIES:")
            print(f"  {'CVE':<22} {'Severity':<10} {'Package'}")
            print(f"  {'-'*60}")
            for v in vulns[:10]:
                vid = v.get("vulnId", v.get("id", "N/A"))
                severity = v.get("severity", "N/A")
                components = v.get("components", [])
                pkg = components[0].get("name", "N/A") if components else "N/A"
                print(f"  {vid:<22} {severity:<10} {pkg}")

if __name__ == "__main__":
    print("\n[1] Logging in...")
    token = login()

    print("[2] Getting API key...")
    api_key = get_or_create_api_key(token)

    print("[3] Uploading SBOM...")
    bom_token = upload_sbom(api_key)

    print("[4] Waiting for analysis...")
    wait_for_processing(api_key, bom_token)

    print("[5] Fetching results...")
    get_summary(api_key, PROJECT_NAME, PROJECT_VERSION)
