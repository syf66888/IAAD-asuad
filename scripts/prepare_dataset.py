"""Prepare video frames, caption TSVs, and offline LK features for asuad."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.prepare_dataset import main

if __name__ == '__main__':
    main()
