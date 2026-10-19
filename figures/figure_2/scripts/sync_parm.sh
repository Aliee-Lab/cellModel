#!/bin/bash
# ============================================================================
# Sync edited PARM multi-output files into the conda env's site-packages
# before training starts. Run this at the TOP of your sbatch script, before
# any `parm train` invocation.
#
# Why this exists: PARM is installed as a REGULAR (non-editable) package in
# this conda env, so editing a working copy of PARM_train.py etc. does
# nothing until those files are copied into site-packages. This makes that
# step explicit and automatic on every job submission, instead of relying on
# remembering to do it by hand.
# ============================================================================

set -euo pipefail

# --- EDIT THESE TWO PATHS FOR YOUR CLUSTER ---------------------------------
# Directory containing YOUR edited copies of: PARM_train.py, PARM_predict.py,
# PARM_utils_load_model.py, __main__.py
EDITED_PARM_DIR="/mnt/scratche/slow/halab/Jack/parm/PARM"

# The PARM package's actual install location inside the conda env.
PARM_SITE_PKG="/Users/kendri01/miniforge3/envs/parm_env/lib/python3.12/site-packages/PARM"
# ----------------------------------------------------------------------------

FILES_TO_SYNC=(
    "PARM_train.py"
    "PARM_predict.py"
    "PARM_utils_load_model.py"
    "__main__.py"
)

echo "[sync_parm] Syncing edited PARM files into $PARM_SITE_PKG"

if [ ! -d "$PARM_SITE_PKG" ]; then
    echo "[sync_parm] ERROR: PARM_SITE_PKG does not exist: $PARM_SITE_PKG"
    echo "[sync_parm] Run: python -c \"import PARM; print(PARM.__file__)\" in this conda env to find the correct path."
    exit 1
fi

for f in "${FILES_TO_SYNC[@]}"; do
    src="$EDITED_PARM_DIR/$f"
    dst="$PARM_SITE_PKG/$f"

    if [ ! -f "$src" ]; then
        echo "[sync_parm] ERROR: expected edited file not found: $src"
        exit 1
    fi

    cp "$src" "$dst"
    echo "[sync_parm]   copied $f"
done

# Clear cached bytecode so a stale .pyc can't mask the change, particularly
# relevant on cluster filesystems (NFS/Lustre) where clock skew between nodes
# can occasionally make Python trust a stale .pyc over a newer .py.
find "$PARM_SITE_PKG" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "[sync_parm]   cleared __pycache__"

# Verify the sync actually took, before burning any GPU time on a job that
# would just fail again like before.
if ! grep -q "n_outputs = model.linear1.out_features" "$PARM_SITE_PKG/PARM_train.py"; then
    echo "[sync_parm] ERROR: verification failed -- PARM_train.py in site-packages"
    echo "[sync_parm]        still doesn't contain the multi-output fix after copying."
    echo "[sync_parm]        Check EDITED_PARM_DIR actually points at the fixed files."
    exit 1
fi

echo "[sync_parm] Verified: multi-output fix present in site-packages copy."
echo "[sync_parm] Sync complete."