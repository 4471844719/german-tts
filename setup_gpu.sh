#!/usr/bin/env bash
# GPU backends for tts_gpu.py. Each backend gets its own venv, because their
# torch/transformers pins conflict.
#   ./setup_gpu.sh fastqwen      # Qwen3-TTS, CUDA-graph accelerated + streaming (recommended)
#   ./setup_gpu.sh qwen          # Qwen3-TTS, official package (slower)
#   ./setup_gpu.sh chatterbox    # Chatterbox Multilingual
#   ./setup_gpu.sh xtts          # Coqui XTTS-v2 (non-commercial license)
#   ./setup_gpu.sh all
#   FLASH=1 ./setup_gpu.sh qwen  # also build flash-attn (faster, ~10-30 min compile)
set -euo pipefail
cd "$(dirname "$0")"

if [[ $EUID -eq 0 ]]; then
  echo "Please run WITHOUT sudo:  ./setup_gpu.sh ${1:-qwen}"
  echo "(the script asks for your password itself, only for apt)"
  exit 1
fi
for d in .venv-*; do
  if [[ -d $d && ! -w $d ]]; then
    echo "$d belongs to root (earlier sudo run). Fix with:  sudo rm -rf $PWD/$d"
    exit 1
  fi
done

command -v nvidia-smi >/dev/null || { echo "Install the NVIDIA driver first (sudo ubuntu-drivers install)"; exit 1; }
nvidia-smi -L

sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential libportaudio2 alsa-utils ffmpeg sox

install_backend() {
  local b=$1 venv=".venv-$1"
  echo; echo "================ $b ================"
  python3 -m venv "$venv"
  "$venv/bin/pip" install --upgrade pip wheel
  case $b in
    fastqwen)
      "$venv/bin/pip" install -U faster-qwen3-tts sounddevice soundfile numpy
      # faster-qwen3-tts ships qwen-tts-hf (needs transformers 5); its upstream twin
      # qwen-tts must not be in the same venv. Newer 5.x (e.g. 5.24) breaks model
      # loading ("property 'upsample' ... has no setter"), so pin the lowest supported one.
      "$venv/bin/pip" uninstall -y qwen-tts 2>/dev/null || true
      "$venv/bin/pip" install "transformers==5.15.1" "huggingface-hub>=1.16,<2" ;;
    qwen)
      "$venv/bin/pip" install -U qwen-tts sounddevice soundfile numpy
      if [[ "${FLASH:-0}" == "1" ]]; then
        "$venv/bin/pip" install ninja packaging
        MAX_JOBS=8 "$venv/bin/pip" install flash-attn --no-build-isolation
      fi ;;
    chatterbox)
      "$venv/bin/pip" install chatterbox-tts sounddevice soundfile numpy ;;
    xtts)
      echo "NOTE: XTTS-v2 weights are under the Coqui Public Model License (non-commercial)."
      "$venv/bin/pip" install coqui-tts sounddevice soundfile numpy ;;
    *) echo "unknown backend $b"; exit 1 ;;
  esac
  "$venv/bin/python" -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
  echo ">> downloading model weights (first run only)"
  "$venv/bin/python" tts_gpu.py -b "$b" --gpu 0 --download
  if [[ $b == fastqwen || $b == qwen ]]; then   # Base model for voice cloning
    "$venv/bin/python" -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"
  fi
}

target=${1:-fastqwen}
if [[ $target == all ]]; then
  for b in fastqwen qwen chatterbox xtts; do install_backend "$b"; done
else
  install_backend "$target"
fi
chmod +x say
echo; echo "Done. Record a German reference voice with ./record_voice.sh, then:"; echo "  ./say --clone voices/meine_stimme.wav \"Guten Abend! Das ist ein Test.\""
