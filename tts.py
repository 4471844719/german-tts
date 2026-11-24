#!/usr/bin/env python3
"""
Real-time German TTS (local, offline) using Piper.

Text is split into sentences; sentence N+1 is synthesized while sentence N is
playing, so speech starts after the first sentence (~100-300 ms on a normal CPU)
instead of after the whole text.

Usage:
  ./tts.py "Hallo Welt."                 # speak arguments
  echo "Text" | ./tts.py                 # speak stdin (streams line by line)
  ./tts.py -f buch.txt                   # speak a file
  ./tts.py -i                            # interactive: type, Enter = speak
  ./tts.py --serve 5005                  # HTTP: curl -d "Hallo" localhost:5005
  ./tts.py "Hallo" -o out.wav            # write WAV instead of playing
  ./tts.py --gpu 0 "Hallo"               # run on NVIDIA GPU 0 (see nvidia-smi)
Options: -v voice, --speed 1.2 (faster), --list-voices
"""
import argparse
import http.server
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOICE_DIR = HERE / "voices"
DEFAULT_VOICE = "de_DE-thorsten-high"

# --------------------------------------------------------------------------
# German sentence splitting (doesn't break on "z. B.", "Dr.", "3. Mai", ...)
# --------------------------------------------------------------------------
ABBREV = {
    "z.b", "z. b", "d.h", "d. h", "u.a", "u. a", "usw", "bzw", "ca", "dr", "prof",
    "nr", "str", "vgl", "evtl", "ggf", "inkl", "etc", "bspw", "sog", "u.ä",
    "hr", "fr", "jh", "mio", "mrd", "abs", "s", "st", "geb", "tel", "zb", "dh",
    "jan", "feb", "mär", "apr", "jun", "jul", "aug", "sep", "sept", "okt", "nov", "dez",
}
MAX_CHUNK = 240  # characters (XTTS German limit is ~250); longer sentences split at commas


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts, start = [], 0
    for m in re.finditer(r"[.!?…]+[\"'»«”“‟›‹)]*\s+", text):
        end = m.end()
        before = text[start:m.start()]
        last_word = re.split(r"\s", before)[-1].lower() if before else ""
        tail = text[max(start, m.start() - 4):m.start()].lower().strip()
        # skip: abbreviations, ordinals like "3. Mai", single initials "A. Merkel"
        if m.group().startswith(".") and (
            last_word.rstrip(".") in ABBREV
            or tail.rstrip(".") in ABBREV
            or re.fullmatch(r"\d+", last_word)
            or re.fullmatch(r"\w", last_word)
        ):
            continue
        parts.append(text[start:end].strip())
        start = end
    if start < len(text):
        parts.append(text[start:].strip())

    out = []
    for p in parts:  # cap chunk length for low first-audio latency
        while len(p) > MAX_CHUNK:
            cut = max(p.rfind(c, 0, MAX_CHUNK) for c in ",;:–")
            cut = cut if cut > 40 else p.rfind(" ", 0, MAX_CHUNK)
            cut = cut if cut > 0 else MAX_CHUNK
            out.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            out.append(p)
    return out


# --------------------------------------------------------------------------
# Audio output: sounddevice (PortAudio) or fallback to pw-play/paplay/aplay
# --------------------------------------------------------------------------
class Player:
    def __init__(self, rate: int):
        self.rate, self.stream, self.proc = rate, None, None
        try:
            import sounddevice as sd
            self.stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16")
            self.stream.start()
            return
        except Exception as e:  # no PortAudio / no device
            print(f"[tts] sounddevice unavailable ({e}), using CLI player", file=sys.stderr)
        for cmd in (
            ["pw-play", "--rate", str(rate), "--channels", "1", "--format", "s16", "-"],
            ["paplay", "--raw", f"--rate={rate}", "--channels=1", "--format=s16le"],
            ["aplay", "-q", "-r", str(rate), "-c", "1", "-f", "S16_LE", "-t", "raw"],
        ):
            if shutil.which(cmd[0]):
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                return
        sys.exit("[tts] no audio output found (install libportaudio2 or alsa-utils)")

    def write(self, pcm: bytes):
        if self.stream:
            self.stream.write(pcm)
        else:
            self.proc.stdin.write(pcm)
            self.proc.stdin.flush()

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        elif self.proc:
            self.proc.stdin.close()
            self.proc.wait()


# --------------------------------------------------------------------------
# TTS engine: synth thread -> queue -> playback thread
# --------------------------------------------------------------------------
class GermanTTS:
    def __init__(self, voice: str, speed: float = 1.0, use_cuda: bool = False):
        if use_cuda:
            try:  # load CUDA/cuDNN libs shipped by onnxruntime-gpu[cuda,cudnn]
                import onnxruntime as ort
                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()
            except ImportError:
                pass
        from piper import PiperVoice

        model = Path(voice)
        if not model.exists():
            model = VOICE_DIR / f"{voice}.onnx"
        if not model.exists():
            sys.exit(f"[tts] voice not found: {model}  (run setup.sh or --list-voices)")
        self.voice = PiperVoice.load(str(model), use_cuda=use_cuda)
        self.rate = self.voice.config.sample_rate
        self.syn_config = None
        try:
            from piper import SynthesisConfig
            self.syn_config = SynthesisConfig(length_scale=1.0 / speed)
        except ImportError:
            pass
        if use_cuda:
            sess = getattr(self.voice, "session", None)
            provs = sess.get_providers() if sess else []
            if provs and "CUDAExecutionProvider" not in provs:
                print("[tts] WARNING: CUDA not active, running on CPU "
                      "(run ./setup.sh --gpu)", file=sys.stderr)
            # warm-up: first CUDA run initialises kernels (~1 s), do it now, not on sentence 1
            for _ in self.chunks("Hallo."):
                pass

    def chunks(self, sentence: str):
        """Yield raw int16 PCM bytes for one sentence."""
        kw = {"syn_config": self.syn_config} if self.syn_config else {}
        for chunk in self.voice.synthesize(sentence, **kw):
            yield chunk.audio_int16_bytes

    def speak_stream(self, sentences, player: Player):
        """sentences: any iterable (can be a live generator, e.g. stdin)."""
        q: queue.Queue = queue.Queue(maxsize=8)

        def producer():
            try:
                for s in sentences:
                    for pcm in self.chunks(s):
                        q.put(pcm)
                    q.put(b"\x00\x00" * int(self.rate * 0.12))  # short pause
            finally:
                q.put(None)

        threading.Thread(target=producer, daemon=True).start()
        while (pcm := q.get()) is not None:
            player.write(pcm)

    def speak(self, text: str):
        player = Player(self.rate)
        try:
            self.speak_stream(split_sentences(text), player)
        finally:
            player.close()

    def to_wav(self, text: str, path: str):
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.rate)
            for s in split_sentences(text):
                for pcm in self.chunks(s):
                    w.writeframes(pcm)


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def stream_sentences(lines):
    """Turn a live stream of lines into complete sentences.
    A sentence broken over several lines is joined; an unfinished sentence is
    held back until it ends, a blank line appears, or the input ends."""
    buf = ""
    for line in lines:
        if not line.strip():                 # blank line = paragraph end -> flush
            if buf.strip():
                yield from split_sentences(buf)
            buf = ""
            continue
        buf += " " + line.strip()
        parts = split_sentences(buf)
        if not parts:
            continue
        if re.search(r"[.!?…][\"'»«”“)]*$", buf.strip()):
            yield from parts                 # everything is complete
            buf = ""
        else:
            yield from parts[:-1]            # keep the unfinished tail
            buf = parts[-1]
    if buf.strip():
        yield from split_sentences(buf)


def stdin_sentences():
    """Speak stdin as it arrives, sentence by sentence."""
    yield from stream_sentences(sys.stdin)


def serve(tts: GermanTTS, port: int):
    """POST text -> spoken in order. Requests are queued, never overlap."""
    jobs: queue.Queue = queue.Queue()

    def worker():
        player = Player(tts.rate)
        def gen():
            while True:
                yield from split_sentences(jobs.get())
        tts.speak_stream(gen(), player)

    threading.Thread(target=worker, daemon=True).start()

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            jobs.put(self.rfile.read(n).decode("utf-8", "replace"))
            self.send_response(202)
            self.end_headers()
            self.wfile.write(b"queued\n")

        def log_message(self, *a):
            pass

    print(f"[tts] listening on http://127.0.0.1:{port}  (curl -d 'Hallo' ...)")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(description="Real-time German TTS (Piper, offline)")
    ap.add_argument("text", nargs="*", help="text to speak (else stdin)")
    ap.add_argument("-v", "--voice", default=DEFAULT_VOICE, help="voice name or .onnx path")
    ap.add_argument("-f", "--file", help="read text from file")
    ap.add_argument("-o", "--out", help="write WAV file instead of playing")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 normal, 1.3 faster")
    ap.add_argument("--serve", type=int, metavar="PORT", help="run local HTTP server")
    ap.add_argument("--gpu", type=int, metavar="N", nargs="?", const=0,
                    help="use NVIDIA GPU N (index as shown by nvidia-smi, default 0)")
    ap.add_argument("--list-voices", action="store_true")
    a = ap.parse_args()

    if a.list_voices:
        for p in sorted(VOICE_DIR.glob("*.onnx")):
            print(p.stem)
        return

    if a.gpu is not None:
        # make the index match nvidia-smi, and expose only that card to CUDA
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    tts = GermanTTS(a.voice, a.speed, use_cuda=a.gpu is not None)

    if a.serve:
        return serve(tts, a.serve)
    if a.interactive:
        print("Text eingeben, Enter = sprechen, Strg+D = Ende")
        player = Player(tts.rate)
        try:
            def lines():
                while True:
                    try:
                        yield from split_sentences(input("> "))
                    except EOFError:
                        return
            tts.speak_stream(lines(), player)
        finally:
            player.close()
        return

    if a.file:
        text = Path(a.file).read_text(encoding="utf-8")
    elif a.text:
        text = " ".join(a.text)
    else:
        text = None

    if a.out:
        tts.to_wav(text if text is not None else sys.stdin.read(), a.out)
        print(f"[tts] wrote {a.out}")
    elif text is not None:
        tts.speak(text)
    else:  # live stdin streaming
        player = Player(tts.rate)
        try:
            tts.speak_stream(stdin_sentences(), player)
        finally:
            player.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
