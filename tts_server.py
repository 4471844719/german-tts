#!/usr/bin/env python3
"""
German TTS REST server: the model stays loaded on the GPU, every request returns
an MP3 as a base64 string.

  POST http://<ip>:8887/speech/synthesizeRT
       Content-Type: application/json
       {"text": "Willkommen in der Welt der Synchrotrons!"}

  -> 200 {"audio": "<base64 mp3>", "format": "mp3", "sample_rate": 24000,
          "duration_s": 2.84, "synthesis_s": 0.61, "sentences": 1}

  GET  http://<ip>:8887/health  -> {"status": "ok", "backend": ..., "gpus": ...}

Optional JSON fields: "format": "mp3" | "wav", "bitrate": "128k" (MP3 bitrate)

Start (via ./serve, which picks the right venv):
  ./serve --clone voices/meine_stimme.wav
  ./serve --clone voices/meine_stimme.wav --gpu 0,1 --api-key geheim
Uses the model code from tts_gpu.py (must be in the same folder; it is not changed).
"""
import argparse
import base64
import json
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tts_gpu  # noqa: E402  (Engine + backends, unchanged)

ENDPOINT = "/speech/synthesizeRT"
MAX_TEXT = 20000  # characters per request


# --------------------------------------------------------------------------
# MP3 encoding: ffmpeg (installed by setup_gpu.sh) or the lameenc pip package
# --------------------------------------------------------------------------
def encode_mp3(pcm: bytes, sr: int, bitrate: str = "128k") -> bytes:
    if shutil.which("ffmpeg"):
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
             "-codec:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3", "pipe:1"],
            input=pcm, capture_output=True, check=False)
        if p.returncode == 0:
            return p.stdout
        raise RuntimeError("ffmpeg failed: " + p.stderr.decode(errors="replace")[-300:])
    try:
        import lameenc
    except ImportError:
        raise RuntimeError("no MP3 encoder: sudo apt install ffmpeg  (or pip install lameenc)")
    enc = lameenc.Encoder()
    enc.set_bit_rate(int(bitrate.rstrip("k")))
    enc.set_in_sample_rate(sr)
    enc.set_channels(1)
    enc.set_quality(2)
    return enc.encode(pcm) + enc.flush()


def encode_wav(pcm: bytes, sr: int) -> bytes:
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


# --------------------------------------------------------------------------
class TTSService:
    def __init__(self, a):
        self.a = a
        self.engine = tts_gpu.Engine(a)          # loads + warms up the model(s) once
        # One request at a time on the GPUs (CUDA graphs are not thread-safe).
        # With --gpu 0,1 the sentences of a request are split across both cards.
        self.lock = threading.Lock()
        self.served = 0

    def synthesize(self, text: str):
        sentences = tts_gpu.split_sentences(text)
        if not sentences:
            raise ValueError("text contains nothing to speak")
        chunks = []
        with self.lock:
            t0 = time.perf_counter()
            self.engine._run_jobs(sentences, chunks.append)
            dt = time.perf_counter() - t0
            self.served += 1
        pcm = b"".join(chunks)
        # drop the pause the engine appends after the last sentence
        tail = int(self.engine.sr * tts_gpu.PAUSE_S) * 2
        if len(pcm) > tail and not pcm[-tail:].strip(b"\x00"):
            pcm = pcm[:-tail]
        return pcm, dt, len(sentences)


def make_handler(svc: TTSService, api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GermanTTS/1.0"

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

        def do_OPTIONS(self):  # browser preflight
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                return self._send(200, {"status": "ok", "backend": svc.a.backend,
                                        "gpus": svc.a.gpu, "sample_rate": svc.engine.sr,
                                        "voice": svc.a.clone or svc.a.speaker or svc.a.design,
                                        "requests_served": svc.served,
                                        "busy": svc.lock.locked()})
            self._send(404, {"error": f"use POST {ENDPOINT}"})

        def do_POST(self):
            if self.path.split("?")[0].rstrip("/") != ENDPOINT:
                return self._send(404, {"error": f"unknown path, use POST {ENDPOINT}"})
            if api_key and self.headers.get("X-API-Key") != api_key:
                return self._send(401, {"error": "missing or wrong X-API-Key"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n).decode("utf-8", "replace")
                ctype = self.headers.get("Content-Type", "")
                if "json" in ctype or raw.lstrip().startswith("{"):
                    req = json.loads(raw or "{}")
                else:                              # plain text body is accepted too
                    req = {"text": raw}
                text = str(req.get("text", "")).strip()
                fmt = str(req.get("format", "mp3")).lower()
                bitrate = str(req.get("bitrate", svc.a.bitrate))
                if not text:
                    return self._send(400, {"error": "field 'text' is required"})
                if len(text) > MAX_TEXT:
                    return self._send(413, {"error": f"text longer than {MAX_TEXT} characters"})
                if fmt not in ("mp3", "wav"):
                    return self._send(400, {"error": "format must be 'mp3' or 'wav'"})
            except (json.JSONDecodeError, ValueError) as e:
                return self._send(400, {"error": f"invalid request: {e}"})

            try:
                pcm, synth_s, n_sent = svc.synthesize(text)
                sr = svc.engine.sr
                audio = encode_mp3(pcm, sr, bitrate) if fmt == "mp3" else encode_wav(pcm, sr)
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:
                print(f"[server] error: {e}", file=sys.stderr)
                return self._send(500, {"error": str(e)})

            duration = len(pcm) / 2 / sr
            print(f"[server] {self.client_address[0]}  {len(text)} chars  {n_sent} sentences  "
                  f"{duration:.1f}s audio in {synth_s:.2f}s  (x{duration / max(synth_s, 1e-6):.1f})",
                  file=sys.stderr)
            self._send(200, {"audio": base64.b64encode(audio).decode("ascii"),
                             "format": fmt, "sample_rate": sr,
                             "duration_s": round(duration, 2),
                             "synthesis_s": round(synth_s, 2),
                             "sentences": n_sent})

        def log_message(self, *args):  # requests are logged above
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description="German TTS REST server (model stays loaded)")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable on all network interfaces")
    ap.add_argument("--port", type=int, default=8887)
    ap.add_argument("--api-key", help="if set, clients must send header X-API-Key")
    ap.add_argument("--bitrate", default="128k", help="default MP3 bitrate")
    # same model/voice options as tts_gpu.py
    ap.add_argument("-b", "--backend", choices=tts_gpu.BACKENDS, default="fastqwen")
    ap.add_argument("--gpu", default="0", help="0 | 1 | 0,1 (numbering as in nvidia-smi)")
    ap.add_argument("--speaker")
    ap.add_argument("--clone", metavar="WAV")
    ap.add_argument("--ref-text")
    ap.add_argument("--design")
    ap.add_argument("--instruct")
    ap.add_argument("--size", default="1.7B", choices=["1.7B", "0.6B"])
    ap.add_argument("--model")
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--exaggeration", type=float, default=0.5)
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()

    if a.clone:
        if not Path(a.clone).is_file():
            sys.exit(f"[server] file not found: {a.clone}")
        if not a.ref_text:
            t = Path(a.clone).with_suffix(".txt")
            if t.is_file():
                a.ref_text = t.read_text(encoding="utf-8").strip()
    if not shutil.which("ffmpeg"):
        try:
            import lameenc  # noqa: F401
        except ImportError:
            print("[server] WARNING: no MP3 encoder found (sudo apt install ffmpeg)", file=sys.stderr)

    svc = TTSService(a)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(svc, a.api_key))
    shown = a.host if a.host != "0.0.0.0" else "<this-machine-ip>"
    print(f"[server] ready: POST http://{shown}:{a.port}{ENDPOINT}", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
