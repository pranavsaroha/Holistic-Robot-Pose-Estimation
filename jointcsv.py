#!/usr/bin/env python3
"""
Convert a Hugging-Face episode CSV (episode-7-state-observations.csv)
into HoRoPose-ready joint_states.csv.

Input columns expected:
    frame_index            int
    observation.state      JSON list of 6 floats

Output:
    joint_states.csv       Nx6 numeric CSV, no header
"""

import pandas as pd
import numpy as np
import json, pathlib, sys

# ──── EDIT THESE THREE PATHS ────────────────────────────────────────────────
HF_CSV   = "episode-7-state-observations.csv"      # input you downloaded
OUT_CSV  = "data/so100_real/joint_states.csv"      # output path
IMG_DIR  = pathlib.Path("data/so100_real/images")  # PNG frames for this ep
# ────────────────────────────────────────────────────────────────────────────

print(f"loading {HF_CSV} …")
df = pd.read_csv(HF_CSV, usecols=["state"])


# JSON list → python list → numeric array
angles = df["state"].apply(json.loads).to_list()
angles = np.array(angles, dtype=float)           # shape (N, 6)

# sanity-check PNG count ↔ angle rows
n_png = len(list(IMG_DIR.glob("*.png")))
if n_png != len(angles):
    sys.exit(f"❌  {n_png} PNGs but {len(angles)} angle rows – mismatch!")

# write plain numeric CSV
OUT_CSV = pathlib.Path(OUT_CSV)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(angles).to_csv(OUT_CSV,
                            header=False, index=False, float_format="%.6f")
print(f"✓ wrote {OUT_CSV}  rows={len(angles)}")
