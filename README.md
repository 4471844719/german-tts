# Real-time German TTS (Ubuntu, local, offline)

Two engines in one folder:

| | Engine | Hardware | Quality | Start command |
|---|---|---|---|---|
| **Large GPU models** (recommended) | Qwen3-TTS via faster-qwen3-tts, optionally Chatterbox / XTTS-v2 | NVIDIA GPU | very natural, voice cloning | `./say` |
| **Small CPU model** | Piper + Thorsten voices | any CPU | good, robotic at times | `./tts.py` |

Internet is only needed for the install and the first model download. After that, everything runs locally.

## Files

| File | Purpose |
|---|---|
| `say` | wrapper for the GPU version: picks the right venv and starts `tts_gpu.py` |
| `tts_gpu.py` | GPU engine (runs standalone, doesn't need `tts.py`) |
| `setup_gpu.sh` | installs a GPU backend into its own venv `.venv-<backend>` |
| `record_voice.sh` | records a German reference voice for cloning |
| `serve` / `tts_server.py` | REST server on port 8887, model stays loaded, MP3 as base64 (needs `tts_gpu.py`) |
| `tts_client.py` | small test client for the server |
| `get_stock_voice.py` | reference voice without a mic: Thorsten (CC0) or a voice designed by Qwen |
| `examples/synchrotron.txt` | benchmark text |
| `voices/` | your reference voices (`.wav` + `.txt`), not tracked in git |
| `tts.py` | Piper CPU engine |
| `setup.sh` | installs Piper into `.venv` |

All files belong in **one folder**. If the browser renames a download to `tts_gpu (1).py` or similar, rename it back.
Downloaded files lose their executable flag, so run this once:
```bash
chmod +x *.sh *.py say
```

---

## Getting it from git
```bash
git clone <REPO-URL> german-tts
cd german-tts
./setup_gpu.sh fastqwen
python3 get_stock_voice.py          # or ./record_voice.sh
./say --clone voices/thorsten.wav -f examples/synchrotron.txt
```
The repo contains only code and docs. Python environments (`.venv-*`), models and voice recordings are created locally and stay out of git (see `.gitignore`).

## Quick start (GPU)

Requirements: Ubuntu with an NVIDIA driver (`nvidia-smi` must list your cards). The CUDA toolkit is **not** needed.

```bash
./setup_gpu.sh fastqwen        # NO sudo: the script asks for your password itself, only for apt
./record_voice.sh              # record your German reference voice (~10 s)
./say --clone voices/meine_stimme.wav "Willkommen in der faszinierenden Welt der Synchrotrons!"
```

Plan for about 10 GB of disk space per backend (PyTorch + model weights).

## Usage (`./say`)
```bash
./say "Text"                              # speak
./say -f text.txt                         # speak a file
./say -i                                  # interactive: type, press Enter, it speaks
some_program | ./say                      # live from another program (sentences may span lines)
./say -f text.txt -o out.wav              # save a WAV instead of playing
./say --serve 5006                        # HTTP server: curl -d "Hallo" http://127.0.0.1:5006
```
Long texts are fine: they're split into sentences (max. 240 characters per chunk) and spoken one after another.
German abbreviations such as *z. B.*, *Dr.* and *3. Mai* don't cause false splits.

### Backends (`-b`)
| Backend | Model | License | Notes |
|---|---|---|---|
| `fastqwen` (default once installed) | Qwen3-TTS 1.7B via faster-qwen3-tts | Apache-2.0 | CUDA graphs + streaming: about 5× faster than real time on a 4090, ~0.2 s until speech starts |
| `qwen` | Qwen3-TTS, official package | Apache-2.0 | same model, but ~0.5–0.8× real time and no streaming. Only as a fallback |
| `chatterbox` | Chatterbox Multilingual | MIT | good cloning, `--exaggeration` for emotion |
| `xtts` | Coqui XTTS-v2 | CPML, **non-commercial** | streaming, cloning |

Install with `./setup_gpu.sh <backend>` (or `all`). Each backend gets its own venv, because their library versions conflict.

### Voices
**For accent-free German, use voice cloning.** The Qwen preset voices (Ryan, Serena, Vivian, …) are native English or Chinese speakers and carry their accent into German.

```bash
./say --clone voices/meine_stimme.wav "…"                 # cloned voice (recommended)
./say --clone ich.wav --ref-text "exact transcript" "…"    # any clip, transcript given directly
./say --speaker Serena --instruct "ruhig und langsam" "…"  # preset voice + style
./say --design "tiefe, warme Männerstimme, Mitte 50" "…"   # voice from a description
./say -b chatterbox --clone ich.wav --exaggeration 0.8 "…"
```
Only clone voices you have permission to use.

### Recording a reference voice (`record_voice.sh`)
```bash
./record_voice.sh               # -> voices/meine_stimme.wav + voices/meine_stimme.txt
./record_voice.sh sprecherin    # -> voices/sprecherin.wav + .txt
```
1. Press Enter, read the text shown, wait about a second, press Enter again.
2. The recording plays back. The last word ("Tulpen") must be complete. If it isn't, press `n` and record again.
3. The transcript is saved as `.txt` next to the `.wav`, and `./say --clone` reads it automatically.

**Important:** audio and transcript must match **exactly**. If the recording is cut off or you said something different, the cloned voice adds stray sounds (for example an "n" after every sentence). In that case, record again or edit the `.txt`.
Tips: a quiet room, the mic at a steady distance, a natural pace, 8–15 s long. sox (installed by `setup_gpu.sh`) trims silence and normalizes the volume.

### No microphone? Stock voices (`get_stock_voice.py`)
```bash
python3 get_stock_voice.py                       # Thorsten (male, studio quality) -> voices/thorsten.wav + .txt
./say --clone voices/thorsten.wav -f synchrotron.txt
```
**Thorsten-Voice** (Thorsten Müller): native German speaker, recorded on a Rode Podcaster, about 40 hours, **CC0** (free for research and commercial use).
The script downloads only 2–4 consecutive clips (not the 8 GB dataset), joins them into a reference of about 11 s, and saves the exact transcript.
A `voices/<name>.source.txt` file records the source and clip IDs, for citation (DOI 10.57967/hf/3290).

| Command | Result |
|---|---|
| `--seed 7` | different clips (if you'd like a different intonation) |
| `--subset emotional --style <style>` | emotional Thorsten (`--list-styles` shows the available styles) |
| `--subset hessisch` | Thorsten speaking Hessian dialect |
| `--design "Deutsche Muttersprachlerin, Mitte 30, warme, klare Stimme, akzentfreies Hochdeutsch" --name sprecherin` | invents a voice (e.g. female) with Qwen VoiceDesign and saves it as a fixed reference |

`--design` downloads the 1.7B VoiceDesign model on first use. Whether a designed voice sounds accent-free depends on the description, so try it and listen.

### Using both GPUs
```bash
nvidia-smi -L                    # card numbers
./say --gpu 0 "…"                # one card (default 0)
./say --gpu 0,1 -f buch.txt      # both cards
```
With `0,1`, a copy of the model runs on each card. Sentences alternate between the cards and always play in order.
That roughly doubles throughput for long texts. A single short sentence won't start any faster.
`TTS_GPU=1` and `TTS_BACKEND=…` set the defaults (e.g. `export TTS_GPU=0` in `~/.bashrc`).

### Fine-tuning (fastqwen)
| Option | Effect |
|---|---|
| `--chunk-size 4` | speech starts sooner (default 8 ≈ 0.67 s of audio per chunk) |
| `--size 0.6B` | smaller model, faster, slightly lower quality |
| `--instruct "…"` | speaking style for preset voices |

### Benchmark
Playback always takes as long as the speech itself, so measure with `-o`:
```bash
time ./say --clone voices/meine_stimme.wav -f examples/synchrotron.txt -o test.wav
```
If `real` is shorter than the length of the audio, the setup is fast enough for real time.
Reference (RTX 4090, ~75 s of speech): `qwen` took 2:24 min, so it's too slow, and `fastqwen` should take about 15–25 s.

---

## REST server (`./serve` / `tts_server.py`)
The model loads **once** and stays on the GPU. Each request returns an MP3 as base64 in JSON.
`tts_server.py` uses the model code from `tts_gpu.py` (same folder, unchanged).

```bash
./serve --clone voices/meine_stimme.wav                       # port 8887, all network interfaces
./serve --clone voices/meine_stimme.wav --gpu 0,1 --api-key geheim
```
Same voice and model options as `./say` (`-b`, `--gpu`, `--clone`, `--speaker`, `--design`, `--size`, …), plus:
`--port 8887`, `--host 0.0.0.0`, `--api-key KEY`, `--bitrate 128k`.

### API
**`POST http://<ip>:8887/speech/synthesizeRT`**
```json
{"text": "Willkommen in der faszinierenden Welt der Synchrotrons!"}
```
Optional: `"format": "mp3" | "wav"`, `"bitrate": "192k"`. A body sent as `text/plain` works too.
If the server runs with `--api-key`, clients send the header `X-API-Key: <key>`.

Response `200`:
```json
{"audio": "SUQzBAAAAAAA…", "format": "mp3", "sample_rate": 24000,
 "duration_s": 3.12, "synthesis_s": 0.71, "sentences": 1}
```
Errors: `400` (no text / bad JSON), `401` (API key), `404` (wrong path), `413` (over 20,000 characters), `500` (model error). The body is always `{"error": "..."}`.

**`GET http://<ip>:8887/health`** returns status, backend, voice, number of requests, and `busy`.

### Clients
```bash
# curl + jq
curl -s -X POST http://192.168.1.50:8887/speech/synthesizeRT \
     -H "Content-Type: application/json" -d '{"text":"Hallo Welt!"}' \
  | jq -r .audio | base64 -d > hallo.mp3

# included Python client (stdlib only, runs on any machine)
python3 tts_client.py --host 192.168.1.50 "Hallo Welt!" --play
python3 tts_client.py --host 192.168.1.50 -f synchrotron.txt -o synchro.mp3
```
JavaScript in the browser (CORS is enabled):
```js
const r = await fetch("http://192.168.1.50:8887/speech/synthesizeRT", {
  method: "POST", headers: {"Content-Type": "application/json"},
  body: JSON.stringify({text: "Hallo Welt!"})});
const {audio} = await r.json();
new Audio("data:audio/mpeg;base64," + audio).play();
```

### Notes
- **Latency:** the reply arrives once the **whole** text is synthesized, because an MP3 file can only be sent complete. For a quick start, send short texts (one or two sentences per request).
- **Concurrency:** requests run one after another on the GPU (CUDA graphs aren't thread-safe). They wait in turn and don't fail. With `--gpu 0,1`, each request's sentences are split across both cards.
- **Network:** find your IP with `hostname -I`, and open the port in the firewall if needed: `sudo ufw allow 8887/tcp`.
- **Security:** there's no encryption and no login without `--api-key`. Only run it on a trusted network, or limit it to the local machine with `--host 127.0.0.1`.
- **Autostart:** the systemd service below, with `ExecStart=/PATH/TO/serve --clone /PATH/TO/voices/meine_stimme.wav`.

---

## CPU version (Piper)
```bash
./setup.sh
source .venv/bin/activate
./tts.py "Hallo! Das ist eine Sprachausgabe in Echtzeit."
./tts.py -f text.txt | -i | -o out.wav | --serve 5005
./tts.py --speed 1.25 "Schneller"
./tts.py -v de_DE-thorsten-medium "…"     # ~2× faster, for weak CPUs
./tts.py --list-voices
```
Voices: `de_DE-thorsten-high` (default), `de_DE-thorsten-medium`. More with
`python -m piper.download_voices de_DE-kerstin-low --data-dir voices`.
Optional GPU: `./setup.sh --gpu`, then `./tts.py --gpu 0 "…"`. Piper is so small that the GPU barely helps.

---

## Extras

### Autostart as a service (systemd)
`~/.config/systemd/user/german-tts.service`. Replace `/PATH/TO` with your folder, e.g. `/home/sensorium/Documents/GermanTTS2`:
```ini
[Unit]
Description=German TTS server
[Service]
ExecStart=/PATH/TO/serve --clone /PATH/TO/voices/meine_stimme.wav
Restart=on-failure
[Install]
WantedBy=default.target
```
```bash
systemctl --user enable --now german-tts
journalctl --user -u german-tts -f                       # follow the log
python3 /PATH/TO/tts_client.py "Neue Nachricht eingegangen." --play
```
Instead of the REST server, you can also autostart the playback server: `ExecStart=/PATH/TO/say --clone … --serve 5006`. It speaks directly through the machine's speakers, with `curl -d "Text" http://127.0.0.1:5006`.

### Read selected text aloud with a hotkey
```bash
sudo apt install xclip                    # Wayland: wl-clipboard, use `wl-paste -p`
xclip -o -selection primary | /PATH/TO/say
```
Bind it under Settings → Keyboard → Custom Shortcuts. It's faster if the playback server (`./say … --serve 5006`) is running, because the model is already loaded:
`xclip -o -selection primary | curl --data-binary @- http://127.0.0.1:5006`

### Make `say` available everywhere
```bash
sudo ln -s /PATH/TO/say /usr/local/bin/say
```

---

## Troubleshooting

| Problem | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'tts'` | old `tts_gpu.py`. Use the current version, which runs standalone |
| `.venv-…` belongs to `root` | `setup_gpu.sh` was run with sudo. Fix: `sudo rm -rf .venv-<backend>`, then run it again **without** sudo |
| `property 'upsample' … has no setter` or `no attribute 'standardize_rope_params'` | wrong transformers version in `.venv-fastqwen`. Fix: `.venv-fastqwen/bin/pip uninstall -y qwen-tts` then `.venv-fastqwen/bin/pip install "transformers==5.15.1" "huggingface-hub>=1.16,<2"` (`setup_gpu.sh` now does this automatically) |
| still fails, Python 3.14 | qwen-tts-hf officially supports Python 3.10–3.13. Create a venv on Python 3.12 with `uv`: `uv venv --python 3.12 .venv-fastqwen`, then `uv pip install --python .venv-fastqwen faster-qwen3-tts sounddevice soundfile numpy "transformers==5.15.1"` |
| `CUDA False` during setup | NVIDIA driver missing or too old: `sudo ubuntu-drivers install`, then reboot |
| American / English accent | preset voice in use. Use `--clone` with a German recording |
| "n" or other sounds after every sentence | reference recording cut off, or `.txt` doesn't match. Run `./record_voice.sh` again |
| `file not found` | check the path. Text files have to be created first, e.g. `nano synchrotron.txt` |
| Warnings about flash-attn, Triton, `pad_token_id`, `torch_dtype` | harmless, ignore them |
| No sound | test with `speaker-test -t wav -c 2`. The scripts use PortAudio, falling back to `pw-play`/`paplay`/`aplay` |
| Stutters during live playback | GPU too slow for that backend: use `fastqwen`, `--size 0.6B` or `--gpu 0,1` |
