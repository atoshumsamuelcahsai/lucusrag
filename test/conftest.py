# test/conftest.py
import sys
from pathlib import Path

# Get repo root (parent of 'lucusrag')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
