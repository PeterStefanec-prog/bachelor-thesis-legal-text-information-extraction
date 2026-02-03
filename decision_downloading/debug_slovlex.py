import requests

SLOVLEX_API_URL = "https://www.slov-lex.sk/server/api/judicature"
TEST_CASE_ID = "I. ÚS 266/06"


HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

print(f"--- DEBUGGING SLOV-LEX for: {TEST_CASE_ID} ---")

# test 1: GET na /search
url = f"{SLOVLEX_API_URL}/search"
params = {"caseNumber": TEST_CASE_ID}

print(f"\n1. Trying GET request to: {url}")
try:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
    print(f"Status: {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('Content-Type')}")
    print("RESPONSE TEXT (First 300 chars):")
    print(resp.text[:300])
except Exception as e:
    print(f"Crash: {e}")

# test 2: POST na /search with another headers
print(f"\n2. Trying POST request to: {url}")
payload = {"searchParameters": {"caseNumber": TEST_CASE_ID}}
try:
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=10)
    print(f"Status: {resp.status_code}")
    print("RESPONSE TEXT (First 300 chars):")
    print(resp.text[:300])
except Exception as e:
    print(f"Crash: {e}")