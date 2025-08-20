import urllib.request
import sys

try:
    with urllib.request.urlopen("http://127.0.0.1:8080/healthz", timeout=3) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
