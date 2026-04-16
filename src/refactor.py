import ast
from pathlib import Path
import os

src_dir = Path("/home/carolluca/work/data-analysis/src")
atlas_path = src_dir / "accident_atlas.py"

with open(atlas_path, "r", encoding="utf-8") as f:
    source = f.read()

tree = ast.parse(source)

# I can get source code of each node but this gets complicated.
# Instead, grep and awk might be too prone to error.
