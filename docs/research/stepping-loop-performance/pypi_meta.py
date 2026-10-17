"""Print PyPI metadata (version, requires-python, cp313 macOS arm64 wheels)."""

import json
import sys
import urllib.request

for pkg in sys.argv[1:]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json") as r:
        d = json.load(r)
    info, files = d["info"], d["urls"]
    wheels = [
        f["filename"]
        for f in files
        if "cp313-" in f["filename"]
        and "macosx" in f["filename"]
        and "arm64" in f["filename"]
    ]
    up = files[0]["upload_time"][:10] if files else "?"
    print(pkg, info["version"], up, info["requires_python"], wheels[:2])
