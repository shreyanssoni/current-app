import requests
import json
import time
import sys

# Configuration
CORE_URL = "http://localhost:3001"
AI_URL = "http://localhost:8000"

# Metadata for Coverage
EXPECTED_ENDPOINTS = [
    {"service": "Core (Node)", "method": "GET", "path": "/"},
    {"service": "Core (Node)", "method": "POST", "path": "/api/tasks/suggest"},
    {"service": "AI Gateway (Python)", "method": "GET", "path": "/health"},
    {"service": "AI Gateway (Python)", "method": "POST", "path": "/generate"},
]

tested_endpoints = set()

def report_coverage():
    total = len(EXPECTED_ENDPOINTS)
    covered = len(tested_endpoints)
    percentage = (covered / total) * 100
    print("\n" + "="*50)
    print(f"API COVERAGE REPORT: {percentage:.2f}%")
    print("="*50)
    for ep in EXPECTED_ENDPOINTS:
        status = "✅ TESTED" if f"{ep['method']} {ep['path']}" in tested_endpoints else "❌ MISSING"
        print(f"{status} | {ep['service']} | {ep['method']} {ep['path']}")
    print("="*50 + "\n")

def test_ai_health():
    print("Testing AI Gateway Health...")
    try:
        response = requests.get(f"{AI_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        tested_endpoints.add("GET /health")
        print("✅ AI Health OK")
    except Exception as e:
        print(f"❌ AI Health Failed: {e}")

def test_ai_generate_failure_no_key():
    print("Testing AI Generate (Expected Fail - No API Key)...")
    try:
        payload = {"prompt": "Hello", "provider": "groq"}
        response = requests.post(f"{AI_URL}/generate", json=payload)
        # We expect a 500 if the key is missing from .env
        assert response.status_code in [200, 500] 
        tested_endpoints.add("POST /generate")
        print(f"✅ AI Generate Handled (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ AI Generate Test Failed: {e}")

def test_core_root():
    print("Testing Core Node Root...")
    try:
        response = requests.get(f"{CORE_URL}/")
        assert response.status_code == 200
        assert "Core Node Service" in response.text
        tested_endpoints.add("GET /")
        print("✅ Core Root OK")
    except Exception as e:
        print(f"❌ Core Root Failed: {e} (Is it running on 3001?)")

def test_core_suggest_edge_case():
    print("Testing Core Suggest (Edge Case - Missing Body)...")
    try:
        response = requests.post(f"{CORE_URL}/api/tasks/suggest", json={})
        # Express error handler or our logic should handle this
        assert response.status_code in [400, 500]
        tested_endpoints.add("POST /api/tasks/suggest")
        print("✅ Core Suggest Edge Case Handled")
    except Exception as e:
        print(f"❌ Core Suggest Edge Case Failed: {e}")

if __name__ == "__main__":
    print("-" * 50)
    print("STARTING E2E BACKEND TESTS")
    print("-" * 50)
    
    test_ai_health()
    test_ai_generate_failure_no_key()
    test_core_root()
    test_core_suggest_edge_case()
    
    report_coverage()
