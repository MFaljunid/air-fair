"""
_versioning.py — shared helper so no experiment script ever silently
overwrites a previous result file. If out_path already exists, returns
out_path with _v2, _v3, ... appended (whichever is the next free version),
so every run's results stay on disk permanently.
"""
import os


def get_versioned_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    version = 2
    while os.path.exists(f"{base}_v{version}{ext}"):
        version += 1
    versioned = f"{base}_v{version}{ext}"
    print(f"(note: {path} already exists -- saving this run to {versioned} instead, "
          f"so nothing is overwritten)")
    return versioned