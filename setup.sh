#!/usr/bin/env bash
# Local German real-time TTS setup for Ubuntu (22.04 / 24.04).
# Installs Piper TTS into a venv and downloads German voices.
#   ./setup.sh          CPU only
#   ./setup.sh --gpu    + NVIDIA CUDA support (needs the NVIDIA driver, no CUDA toolkit)
set -euo pipefail

cd "$(dirname "$0")"

echo ">> System packages (audio output + venv)"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libportaudio2 alsa-utils

echo ">> Python venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install piper-tts sounddevice numpy

if [[ "${1:-}" == "--gpu" ]]; then
  echo ">> NVIDIA GPU support (onnxruntime-gpu + bundled CUDA 12 / cuDNN 9)"
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first"; exit 1; }
  # only one onnxruntime package may be installed at a time
  pip uninstall -y onnxruntime onnxruntime-gpu
  pip install "onnxruntime-gpu[cuda,cudnn]"
  python - <<'EOF'
import onnxruntime as ort
ort.preload_dlls()
print("ONNX Runtime providers:", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA provider missing"
EOF
fi

echo ">> German voices"
mkdir -p voices
# high = best quality (~110 MB), medium = faster, good for weak CPUs
for v in de_DE-thorsten-high de_DE-thorsten-medium; do
  python -m piper.download_voices "$v" --data-dir voices
done

echo
echo "Done. Test it with:"
echo "  source .venv/bin/activate"
echo "  ./tts.py \"Hallo! Das ist eine Sprachausgabe in Echtzeit.\""
