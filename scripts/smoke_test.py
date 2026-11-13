"""End-to-end check against a RUNNING backend (stdlib only).
    python scripts/smoke_test.py                      # echo provider
    python scripts/smoke_test.py http://localhost:8000 gemini
"""
import json
import sys
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
provider = sys.argv[2] if len(sys.argv) > 2 else "echo"


def call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


print("health   ", call("/api/health"))
print("providers", call("/api/providers"))
res = call("/api/chat", {"message": "Say hello in five words.", "provider": provider})
print("chat     ", res)
hist = call("/api/history?limit=1")
assert hist and hist[0]["reply"] == res["reply"], "reply was not saved to the database"
print("database  ok (reply stored)")
print("ALL GOOD")
