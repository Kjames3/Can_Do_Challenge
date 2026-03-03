import glob
import sys
from pathlib import Path

path_str = "project/*_best.pt"
print(f"Testing glob for: {path_str}")
matched = glob.glob(path_str)
print(f"Matched: {matched}")

if not matched:
    print("Zero matches found.")
else:
    print(f"Found {len(matched)} matches.")
