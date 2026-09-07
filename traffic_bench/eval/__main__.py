import os

# Before importing the CLI (and anything that may pull MetaDrive/Panda3D).
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PULSE_SERVER", "none")

from traffic_bench.eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
