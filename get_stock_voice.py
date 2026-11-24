#!/usr/bin/env python3
"""
Create a German reference voice for cloning WITHOUT a microphone.

Source 1 (default): Thorsten-Voice dataset (Thorsten Müller, native German male
speaker, studio-quality, license CC0 = free for research and commercial use).
A few consecutive clips are downloaded (only those, not the 8 GB dataset),
joined into one 10-15 s reference and saved with the exact transcript.

Source 2 (--design): a voice invented by Qwen3-TTS VoiceDesign from a text
description (e.g. a female voice), saved as a reference so it stays identical.

  python3 get_stock_voice.py                         # -> voices/thorsten.wav + .txt
  python3 get_stock_voice.py --subset emotional --style amused --name thorsten_happy
  python3 get_stock_voice.py --subset hessisch --name thorsten_hessisch
  python3 get_stock_voice.py --list-styles
  python3 get_stock_voice.py --design "Deutsche Muttersprachlerin, Mitte 30, warme, klare Stimme, akzentfreies Hochdeutsch" --name sprecherin
  then:  ./say --clone voices/thorsten.wav "Willkommen in der Welt der Synchrotrons!"

Needs internet once, plus ffmpeg (installed by setup_gpu.sh). Python stdlib only.
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOICES = HERE / "voices"
API = "https://datasets-server.huggingface.co/rows"
DATASET = "Thorsten-Voice/TV-44kHz-Full"
SUBSETS = {  # name -> (config, approx. rows)
    "neutral": ("TV-2022.10-Neutral", 12000),      # newest, best microphone
    "neutral2021": ("TV-2021.02-Neutral", 22000),
    "emotional": ("TV-2021.06-Emotional", 2000),
    "hessisch": ("TV-2023.09-Hessisch", 2000),
}
SR = 24000          # Qwen3-TTS works at 24 kHz
GAP_S = 0.35        # pause between joined clips
DESIGN_TEXT = ("Guten Tag! Ich lese diesen kurzen Text ganz natürlich vor, in meinem normalen "
               "Sprechtempo. Die Sonne scheint, und im Garten blühen schon die ersten Tulpen.")


def get_json(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  retry ({e})", file=sys.stderr)
            time.sleep(2 * (i + 1))


def fetch_rows(config, offset, length=100):
    q = urllib.parse.urlencode({"dataset": DATASET, "config": config, "split": "train",
                                "offset": offset, "length": length})
    return [r["row"] for r in get_json(f"{API}?{q}").get("rows", [])]


def to_pcm(url, tmp: Path) -> bytes:
    """Download one clip and convert to 24 kHz mono 16-bit PCM."""
    src = tmp / "clip"
    urllib.request.urlretrieve(url, src)
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src),
                        "-ar", str(SR), "-ac", "1", "-f", "s16le", "pipe:1"],
                       capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace")[-200:])
    return p.stdout


def write_wav(path: Path, pcm: bytes):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


def tidy(text: str) -> str:
    t = " ".join(text.split())
    return t[:1].upper() + t[1:] if t else t


def from_thorsten(a):
    config, n_rows = SUBSETS[a.subset]
    rng = random.Random(a.seed)
    chosen, total = [], 0.0
    for attempt in range(8):
        offset = rng.randrange(0, max(1, n_rows - 100))
        rows = fetch_rows(config, offset)
        if not rows:
            continue
        chosen, total = [], 0.0
        for r in rows:  # consecutive clips -> consistent recording session
            d = float(r.get("durationSeconds") or 0)
            ok = (2.5 <= d <= 8.0 and not (r.get("comment") or "").strip()
                  and (not a.style or (r.get("style") or "").lower() == a.style.lower()))
            if not ok:
                if chosen and total < a.seconds:   # gap in the sequence: restart
                    chosen, total = [], 0.0
                continue
            chosen.append(r)
            total += d + GAP_S
            if total >= a.seconds:
                break
        if total >= a.seconds:
            break
    if total < a.seconds * 0.6:
        sys.exit("[voice] could not find enough matching clips (try another --style/--seed)")

    print(f"[voice] {len(chosen)} clips from {config}, ~{total:.1f}s:")
    silence = b"\x00\x00" * int(SR * GAP_S)
    pcm, texts = b"", []
    with tempfile.TemporaryDirectory() as td:
        for r in chosen:
            print(f"   {float(r['durationSeconds']):4.1f}s  {r['text']}")
            pcm += (silence if pcm else b"") + to_pcm(r["audio"][0]["src"], Path(td))
            texts.append(tidy(r["text"]))
    pcm = silence[: len(silence) // 3] + pcm + silence  # short lead-in, clean tail

    wav = VOICES / f"{a.name}.wav"
    write_wav(wav, pcm)
    (VOICES / f"{a.name}.txt").write_text(" ".join(texts) + "\n", encoding="utf-8")
    (VOICES / f"{a.name}.source.txt").write_text(
        f"Source: {DATASET} ({config}), Hugging Face, DOI 10.57967/hf/3290\n"
        f"Speaker: Thorsten Müller (Thorsten-Voice), license CC0-1.0\n"
        f"Clip ids: {', '.join(r['id'] for r in chosen)}\n", encoding="utf-8")
    return wav


def from_design(a):
    say = HERE / "say"
    if not say.exists():
        sys.exit("[voice] ./say not found (needs the GPU setup)")
    wav = VOICES / f"{a.name}.wav"
    print(f"[voice] designing voice: {a.design}")
    cmd = [str(say), "--design", a.design, DESIGN_TEXT, "-o", str(wav)]
    if a.gpu is not None:
        cmd += ["--gpu", a.gpu]
    if subprocess.run(cmd).returncode:
        sys.exit("[voice] VoiceDesign failed (first run downloads the 1.7B VoiceDesign model)")
    (VOICES / f"{a.name}.txt").write_text(DESIGN_TEXT + "\n", encoding="utf-8")
    (VOICES / f"{a.name}.source.txt").write_text(
        f"Generated with Qwen3-TTS-12Hz-1.7B-VoiceDesign (Apache-2.0)\nDescription: {a.design}\n",
        encoding="utf-8")
    return wav


def list_styles():
    seen = {}
    for off in range(0, 2000, 400):
        for r in fetch_rows(SUBSETS["emotional"][0], off):
            seen[r.get("style")] = seen.get(r.get("style"), 0) + 1
    print("Styles in --subset emotional:", ", ".join(f"{k} ({v})" for k, v in sorted(seen.items())))


def main():
    ap = argparse.ArgumentParser(description="German reference voice without a microphone")
    ap.add_argument("--name", help="output name in voices/ (default: thorsten / designed)")
    ap.add_argument("--subset", choices=SUBSETS, default="neutral")
    ap.add_argument("--style", help="only clips of this style (see --list-styles)")
    ap.add_argument("--seconds", type=float, default=11.0, help="target length (8-15 s is ideal)")
    ap.add_argument("--seed", type=int, help="pick other clips (any number)")
    ap.add_argument("--design", metavar="DESCRIPTION", help="invent a voice with Qwen VoiceDesign")
    ap.add_argument("--gpu", help="GPU for --design")
    ap.add_argument("--list-styles", action="store_true")
    a = ap.parse_args()

    if a.list_styles:
        return list_styles()
    if not a.design and not shutil.which("ffmpeg"):
        sys.exit("[voice] ffmpeg missing: sudo apt install ffmpeg")
    VOICES.mkdir(exist_ok=True)
    a.name = a.name or ("designed" if a.design else "thorsten")
    wav = from_design(a) if a.design else from_thorsten(a)

    print(f"\n[voice] saved {wav.relative_to(HERE)} + {a.name}.txt (transcript)")
    for player in ("aplay", "paplay", "pw-play"):
        if shutil.which(player):
            print("[voice] playing it back ...")
            subprocess.run([player, str(wav)], stderr=subprocess.DEVNULL)
            break
    print(f'\nUse it:  ./say --clone voices/{a.name}.wav "Willkommen in der Welt der Synchrotrons!"')
    print(f"Server:  ./serve --clone voices/{a.name}.wav")


if __name__ == "__main__":
    main()
