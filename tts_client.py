#!/usr/bin/env python3
"""
Minimal client for tts_server.py (stdlib only, runs on any machine with Python).

  python3 tts_client.py "Hallo Welt"                       # -> ausgabe.mp3
  python3 tts_client.py -f synchrotron.txt -o synchro.mp3
  python3 tts_client.py --host 192.168.1.50 --play "Hallo"  # play right away
  python3 tts_client.py --api-key geheim "Hallo"
"""
import argparse
import base64
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("-f", "--file")
    ap.add_argument("-o", "--out", default="ausgabe.mp3")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8887)
    ap.add_argument("--api-key")
    ap.add_argument("--format", default="mp3", choices=["mp3", "wav"])
    ap.add_argument("--play", action="store_true", help="play with ffplay/mpg123 after download")
    a = ap.parse_args()

    text = Path(a.file).read_text(encoding="utf-8") if a.file else " ".join(a.text)
    if not text.strip():
        sys.exit("no text given")

    url = f"http://{a.host}:{a.port}/speech/synthesizeRT"
    headers = {"Content-Type": "application/json"}
    if a.api_key:
        headers["X-API-Key"] = a.api_key
    req = urllib.request.Request(url, json.dumps({"text": text, "format": a.format}).encode(),
                                 headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"server not reachable at {url}: {e.reason}")
    total = time.perf_counter() - t0

    audio = base64.b64decode(resp["audio"])
    out = a.out if a.format == "mp3" or not a.out.endswith(".mp3") else a.out[:-4] + ".wav"
    Path(out).write_bytes(audio)
    print(f"{out}: {resp['duration_s']}s audio, synthesis {resp['synthesis_s']}s, "
          f"round trip {total:.2f}s, {len(audio) // 1024} KB")
    if a.play:
        for cmd in (["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", out],
                    ["mpg123", "-q", out], ["paplay", out]):
            if shutil.which(cmd[0]):
                subprocess.run(cmd)
                break


if __name__ == "__main__":
    main()
