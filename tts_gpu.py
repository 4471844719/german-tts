#!/usr/bin/env python3
"""
Real-time German TTS on NVIDIA GPUs with large neural models (local, offline).

Backends (each lives in its own venv, see setup_gpu.sh):
  fastqwen    Qwen3-TTS via faster-qwen3-tts (CUDA graphs, streaming) - fastest, recommended
  qwen        Qwen3-TTS 1.7B / 0.6B   Apache-2.0   preset voices, voice cloning, voice design
  chatterbox  Chatterbox Multilingual MIT          voice cloning, emotion intensity
  xtts        Coqui XTTS-v2           CPML (non-commercial!)  true chunk streaming, cloning

Pipeline: text -> sentences -> GPU worker(s) -> ordered playback.
With --gpu 0,1 sentences alternate between both cards and are played in order,
so long texts are synthesized ~2x faster than they are spoken.

Usage (via the ./say wrapper, which picks the right venv):
  ./say "Guten Abend, das ist ein Test."
  ./say -b qwen --speaker Serena --instruct "fröhlich und etwas schneller" "Text"
  ./say -b qwen --clone meine_stimme.wav --ref-text "Was in der Aufnahme gesagt wird."
  ./say -b qwen --design "tiefe, ruhige Männerstimme, Nachrichtensprecher" "Text"
  ./say -b chatterbox --clone stimme.wav --exaggeration 0.7 "Text"
  ./say -b xtts --clone stimme.wav "Text"
  ./say --gpu 0,1 -f buch.txt            # both GPUs
  ./say -i                               # interactive
  ./say --serve 5006                     # HTTP: curl -d "Hallo" localhost:5006
  ./say "Text" -o out.wav
"""
import argparse
import http.server
import re
import shutil
import subprocess
import os
import sys
import threading
import queue
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent

PAUSE_S = 0.12

# ==========================================================================
# Shared helpers (same as in tts.py, so this file runs on its own)
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



def serve(tts, port: int):
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




def first_chunk(sentences, max_len=70):
    """Split the very first sentence at a comma so audio starts sooner."""
    sentences = list(sentences)
    if sentences and len(sentences[0]) > max_len:
        s = sentences[0]
        cut = s.find(",", 20)
        if 0 < cut < len(s) - 15:
            sentences[0:1] = [s[:cut + 1], s[cut + 1:].strip()]
    return sentences


def to_pcm16(audio) -> bytes:
    import numpy as np
    if hasattr(audio, "detach"):  # torch tensor
        audio = audio.detach().float().cpu().numpy()
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    return (np.clip(a, -1.0, 1.0) * 32767).astype("<i2").tobytes()


# ==========================================================================
# Backends: each has .sr and .stream(text) -> yields float32 audio arrays
# ==========================================================================
class QwenBackend:
    def __init__(self, device, a):
        import torch
        from qwen_tts import Qwen3TTSModel

        if a.clone:
            self.mode, repo = "clone", f"Qwen/Qwen3-TTS-12Hz-{a.size}-Base"
        elif a.design:
            self.mode, repo = "design", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        else:
            self.mode, repo = "custom", f"Qwen/Qwen3-TTS-12Hz-{a.size}-CustomVoice"
        attn = "sdpa"
        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except ImportError:
            pass
        self.model = Qwen3TTSModel.from_pretrained(
            a.model or repo, device_map=device, dtype=torch.bfloat16,
            attn_implementation=attn)
        self.a = a
        self.clone_prompt = None
        if self.mode == "clone":
            if not a.ref_text:
                sys.exit("[tts] Qwen cloning needs --ref-text (the words spoken in the clip)")
            if hasattr(self.model, "create_voice_clone_prompt"):  # compute once, reuse
                self.clone_prompt = self.model.create_voice_clone_prompt(
                    ref_audio=a.clone, ref_text=a.ref_text)
        self.sr = None

    def stream(self, text):
        a = self.a
        if self.mode == "custom":
            kw = dict(speaker=a.speaker or "Ryan")
            if a.instruct:
                kw["instruct"] = a.instruct
            wavs, sr = self.model.generate_custom_voice(text=text, language="German", **kw)
        elif self.mode == "design":
            wavs, sr = self.model.generate_voice_design(
                text=text, language="German", instruct=a.design)
        else:
            kw = ({"voice_clone_prompt": self.clone_prompt} if self.clone_prompt
                  else {"ref_audio": a.clone, "ref_text": a.ref_text})
            wavs, sr = self.model.generate_voice_clone(text=text, language="German", **kw)
        self.sr = sr
        yield wavs[0]


class ChatterboxBackend:
    def __init__(self, device, a):
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device(device))
        self.sr = self.model.sr
        self.a = a
        if a.clone:  # compute speaker conditioning once
            self.model.prepare_conditionals(a.clone, exaggeration=a.exaggeration)

    def stream(self, text):
        yield self.model.generate(
            text, language_id="de", exaggeration=self.a.exaggeration,
            cfg_weight=self.a.cfg_weight)


class XttsBackend:
    """XTTS-v2 streams audio chunks while it generates (~0.2 s to first audio)."""

    def __init__(self, device, a):
        from TTS.api import TTS

        api = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.model = api.synthesizer.tts_model
        self.sr = self.model.config.audio.output_sample_rate  # 24000
        if a.clone:
            self.latent, self.emb = self.model.get_conditioning_latents(audio_path=[a.clone])
        else:
            spk = self.model.speaker_manager.speakers
            name = a.speaker or "Ana Florence"
            if name not in spk:
                sys.exit(f"[tts] unknown XTTS speaker {name}. Available: {', '.join(spk)}")
            self.latent, self.emb = spk[name]["gpt_cond_latent"], spk[name]["speaker_embedding"]
        self.speed = a.speed

    def stream(self, text):
        import torch
        with torch.inference_mode():
            for chunk in self.model.inference_stream(
                    text, "de", self.latent, self.emb,
                    stream_chunk_size=20, speed=self.speed, enable_text_splitting=False):
                yield chunk


class FastQwenBackend:
    """Qwen3-TTS via faster-qwen3-tts (CUDA graphs): ~5x real time on a 4090,
    true streaming (~0.2 s to first audio). Same models/voices as 'qwen'."""

    def __init__(self, device, a):
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        self.device, self.a = device, a
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
        if a.clone:
            self.mode, repo = "clone", f"Qwen/Qwen3-TTS-12Hz-{a.size}-Base"
        elif a.design:
            self.mode, repo = "design", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        else:
            self.mode, repo = "custom", f"Qwen/Qwen3-TTS-12Hz-{a.size}-CustomVoice"
        try:
            self.model = FasterQwen3TTS.from_pretrained(a.model or repo, device=device)
        except TypeError:  # older versions without a device argument
            self.model = FasterQwen3TTS.from_pretrained(a.model or repo)
        if hasattr(self.model, "warmup"):
            self.model.warmup(prefill_len=100)   # captures the CUDA graphs now
        self.kw = {}
        if self.mode == "clone":
            if a.ref_text:
                self.kw = dict(ref_audio=a.clone, ref_text=a.ref_text)
                try:  # encode the reference voice once instead of per sentence
                    self.kw = dict(voice_clone_prompt=self.model.model.create_voice_clone_prompt(
                        ref_audio=a.clone, ref_text=a.ref_text))
                except Exception:
                    pass
            else:
                print("[tts] no --ref-text: using speaker-embedding-only cloning "
                      "(accent transfer is weaker; add the transcript for best results)",
                      file=sys.stderr)
                self.kw = dict(ref_audio=a.clone, ref_text="", xvec_only=True)
        self.sr = None

    def thread_init(self):  # CUDA current device is per thread
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.set_device(self.device)

    def stream(self, text):
        a, m = self.a, self.model
        if self.mode == "clone":
            fn, kw = "generate_voice_clone", dict(self.kw)
        elif self.mode == "design":
            fn, kw = "generate_voice_design", dict(instruct=a.design)
        else:
            fn, kw = "generate_custom_voice", dict(speaker=a.speaker or "Ryan")
            if a.instruct:
                kw["instruct"] = a.instruct
        streaming = getattr(m, fn + "_streaming", None)
        if streaming:
            for chunk, sr, _timing in streaming(text=text, language="German",
                                                chunk_size=a.chunk_size, **kw):
                self.sr = sr
                yield chunk
        else:
            wavs, sr = getattr(m, fn)(text=text, language="German", **kw)
            self.sr = sr
            yield wavs[0]


BACKENDS = {"fastqwen": FastQwenBackend, "qwen": QwenBackend,
            "chatterbox": ChatterboxBackend, "xtts": XttsBackend}


# ==========================================================================
# Multi-GPU ordered streaming pipeline
# ==========================================================================
class Engine:
    def __init__(self, a):
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # match nvidia-smi
        gpus = [g.strip() for g in a.gpu.split(",") if g.strip()]
        devices = ["cpu"] if gpus == ["cpu"] else [f"cuda:{g}" for g in gpus]
        cls = BACKENDS[a.backend]
        self.workers = []
        for d in devices:
            print(f"[tts] loading {a.backend} on {d} ...", file=sys.stderr)
            self.workers.append(cls(d, a))
        # warm-up: compiles kernels / allocates memory, and tells us the sample rate
        for w in self.workers:
            for _ in w.stream("Hallo."):
                pass
        self.sr = self.workers[0].sr
        self.rate = self.sr  # name used by tts.serve()
        self.lookahead = max(2, 2 * len(self.workers))
        print(f"[tts] ready ({len(self.workers)} GPU worker(s), {self.sr} Hz)", file=sys.stderr)

    def _run_jobs(self, sentences, sink):
        """Each sentence = one job with its own chunk queue; worker i takes job i % n.
        Playback drains job queues strictly in order, so streaming and parallelism
        both work and the order of sentences is never mixed up."""
        n = len(self.workers)
        inboxes = [queue.Queue() for _ in range(n)]

        def worker(idx):
            w = self.workers[idx]
            if hasattr(w, "thread_init"):
                w.thread_init()
            while (job := inboxes[idx].get()) is not None:
                text, out = job
                try:
                    for audio in w.stream(text):
                        out.put(to_pcm16(audio))
                except Exception as e:
                    print(f"[tts] error on '{text[:40]}…': {e}", file=sys.stderr)
                out.put(None)

        for i in range(n):
            threading.Thread(target=worker, args=(i,), daemon=True).start()

        # dispatcher runs in its own thread, so live input (stdin, -i, server)
        # never blocks playback; bounded queue = how far the GPUs may work ahead
        order: queue.Queue = queue.Queue(maxsize=self.lookahead)

        def dispatch():
            try:
                for i, s in enumerate(sentences):
                    out = queue.Queue()
                    order.put(out)          # blocks when too far ahead
                    inboxes[i % n].put((s, out))
            finally:
                order.put(None)
                for q in inboxes:
                    q.put(None)

        threading.Thread(target=dispatch, daemon=True).start()
        silence = b"\x00\x00" * int(self.sr * PAUSE_S)
        while (out := order.get()) is not None:
            while (pcm := out.get()) is not None:
                sink(pcm)
            sink(silence)

    def speak_stream(self, sentences, player):
        self._run_jobs(sentences, player.write)

    def speak(self, text):
        player = Player(self.sr)
        try:
            self.speak_stream(first_chunk(split_sentences(text)), player)
        finally:
            player.close()

    def to_wav(self, text, path):
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            self._run_jobs(split_sentences(text), w.writeframes)


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Real-time German TTS on GPU")
    ap.add_argument("text", nargs="*")
    ap.add_argument("-b", "--backend", choices=BACKENDS, default="fastqwen")
    ap.add_argument("--gpu", default="0", help="GPU index(es) as in nvidia-smi: 0 | 1 | 0,1 | cpu")
    ap.add_argument("-f", "--file")
    ap.add_argument("-o", "--out")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--serve", type=int, metavar="PORT")
    ap.add_argument("--download", action="store_true", help="download/load model, then exit")
    # voice options
    ap.add_argument("--speaker", help="qwen: Ryan, Serena, Vivian, Aiden, Eric, Dylan, Uncle_Fu, "
                                      "Ono_Anna, Sohee | xtts: built-in speaker name")
    ap.add_argument("--clone", metavar="WAV", help="reference voice clip (5-15 s, clean)")
    ap.add_argument("--ref-text", help="transcript of the clip (default: read from clip.txt next to clip.wav)")
    ap.add_argument("--design", help="qwen: describe the voice in words")
    ap.add_argument("--instruct", help="qwen: style, e.g. 'ruhig und langsam'")
    ap.add_argument("--chunk-size", type=int, default=8,
                    help="fastqwen: steps per audio chunk (8 = 0.67 s; 4 = faster start)")
    ap.add_argument("--size", default="1.7B", choices=["1.7B", "0.6B"], help="qwen model size")
    ap.add_argument("--model", help="override HF repo id / local path")
    ap.add_argument("--exaggeration", type=float, default=0.5, help="chatterbox emotion 0.25-2")
    ap.add_argument("--cfg-weight", type=float, default=0.5, help="chatterbox pacing (lower=slower)")
    ap.add_argument("--speed", type=float, default=1.0, help="xtts speed")
    a = ap.parse_args()

    if a.clone and not a.ref_text:  # transcript next to the clip: stimme.wav -> stimme.txt
        t = Path(a.clone).with_suffix(".txt")
        if t.is_file():
            a.ref_text = t.read_text(encoding="utf-8").strip()
    for p in (a.file, a.clone):  # fail fast, before the slow model load
        if p and not Path(p).is_file():
            sys.exit(f"[tts] file not found: {p}")
    eng = Engine(a)
    if a.download:
        return

    if a.serve:
        return serve(eng, a.serve)

    if a.interactive:
        print("Text eingeben, Enter = sprechen, Strg+D = Ende")
        player = Player(eng.sr)

        def lines():
            while True:
                try:
                    yield from split_sentences(input("> "))
                except EOFError:
                    return
        try:
            eng.speak_stream(lines(), player)
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
        eng.to_wav(text if text is not None else sys.stdin.read(), a.out)
        print(f"[tts] wrote {a.out}", file=sys.stderr)
    elif text is not None:
        eng.speak(text)
    else:
        player = Player(eng.sr)
        try:
            eng.speak_stream(stream_sentences(sys.stdin), player)
        finally:
            player.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
