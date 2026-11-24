#!/usr/bin/env bash
# Record a German reference voice for cloning.
# Creates voices/<name>.wav + voices/<name>.txt (exact transcript).
#   ./record_voice.sh              -> voices/meine_stimme.wav
#   ./record_voice.sh sprecherin   -> voices/sprecherin.wav
#
# You start AND stop the recording with Enter, so nothing gets cut off.
# Audio and transcript must match exactly, otherwise the cloned voice adds
# stray sounds (e.g. an "n" after every sentence).
set -euo pipefail
cd "$(dirname "$0")"
name=${1:-meine_stimme}
mkdir -p voices
raw="voices/$name.raw.wav"
out="voices/$name.wav"

TEXT="Guten Tag! Ich lese diesen kurzen Text ganz natürlich vor, in meinem normalen Sprechtempo. Die Sonne scheint, und im Garten blühen schon die ersten Tulpen."

while true; do
  echo
  echo "Text to read (calm, natural, like talking to someone):"
  echo
  echo "   $TEXT"
  echo
  read -r -p "Press Enter to START recording ... "
  sleep 0.3                                   # don't record the key click
  arecord -q -f S16_LE -r 24000 -c 1 -d 40 "$raw" &
  pid=$!
  echo "● RECORDING – read the text, wait one second, then press Enter to STOP"
  read -r
  kill -INT "$pid" 2>/dev/null || true        # SIGINT = arecord closes the file cleanly
  wait "$pid" 2>/dev/null || true

  if command -v sox >/dev/null; then
    # cut the Enter click at the end, trim silence gently, keep a short tail, normalise
    sox "$raw" "$out" trim 0 -0.25 \
        silence 1 0.05 0.3% reverse silence 1 0.05 0.3% reverse \
        pad 0.1 0.4 norm -1
    rm -f "$raw"
  else
    mv "$raw" "$out"
    echo "(install sox for automatic trimming: sudo apt install sox)"
  fi

  dur=$(python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(round(w.getnframes()/w.getframerate(),1))" "$out")
  echo "Length: ${dur}s – listen to the end, 'Tulpen' must be complete:"
  aplay -q "$out"
  if python3 -c "import sys;sys.exit(0 if 5<=float(sys.argv[1])<=25 else 1)" "$dur"; then :; else
    echo "⚠ ${dur}s is unusual (ideal: 8–15 s)."
  fi
  read -r -p "Keep this recording? [Y/n] " ok
  [[ ${ok:-y} =~ ^[Yy]$ ]] && break
done

echo "$TEXT" > "voices/$name.txt"
echo
echo "Saved $out + voices/$name.txt"
echo "If you said anything differently, edit voices/$name.txt to match EXACTLY."
echo "Use it:  ./say --clone $out \"Willkommen in der Welt der Synchrotrons!\""
