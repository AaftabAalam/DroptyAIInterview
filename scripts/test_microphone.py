#!/usr/bin/env python3
"""
Test that the microphone is detected and recording. Run from project root:
  python scripts/test_microphone.py

You should see "Input devices:" and a 3-second recording level. If you see
"Error opening stream" or no level, fix microphone permissions or choose
another device index (see http://localhost:8000/api/audio/devices when app runs).
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import numpy as np
import sounddevice as sd

def main():
    print("Input devices (use index if default is wrong):")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            default = " (default)" if i == sd.default.device[0] else ""
            print(f"  {i}: {dev.get('name', 'Unknown')}{default}")
    print()

    sample_rate = 16000
    duration_sec = 3
    print(f"Recording for {duration_sec} seconds at {sample_rate} Hz... Speak now.")
    try:
        rec = sd.rec(
            int(duration_sec * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16,
        )
        sd.wait()
        peak = np.abs(rec).max()
        print(f"Done. Peak level: {peak} (if 0 or very low, mic may be muted or wrong device).")
        if peak < 100:
            print("  -> Try another device index: INTERVIEW_AUDIO_INPUT_DEVICE=1 (or 2, 3...) in .env")
        return 0
    except Exception as e:
        print("Error opening stream:", e)
        print("  -> Grant microphone permission to Terminal (or your IDE) in System Settings > Privacy.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
