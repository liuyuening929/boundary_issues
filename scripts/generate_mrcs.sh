#!/bin/bash

MOD_DIR="/mnt/efs/aimbl_2025/student_data/S-TB/Contours/processed_modfiles"
MRC_DIR="/mnt/efs/aimbl_2025/student_data/S-TB/Tomograms/tomograms"
OUT_DIR="/mnt/efs/aimbl_2025/student_data/S-TB/Contours/segmentation_mrcs"
mrc_ending="_10.00Apx"
mod_ending="_18.00Apx"

mkdir -p "$OUT_DIR"

for modfile in "$MOD_DIR"/*.mod; do
    base=$(basename "$modfile" .mod)
    # Set correct MRC directory
    mrc_base="${base/${mod_ending}/}"
    mrcfile="$MRC_DIR/${mrc_base}${mrc_ending}.mrc"
    if [[ -f "$mrcfile" ]]; then
        outmrc="$OUT_DIR/${base}.mrc"
        echo "Running imodmop for $modfile and $mrcfile -> $outmrc"
        imodmop -tu 1-100 -diam 10 -l / "$modfile" "$mrcfile" "$outmrc"
    else
        echo "No matching mrc for $modfile"
    fi
done