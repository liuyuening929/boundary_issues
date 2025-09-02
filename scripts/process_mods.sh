#!/bin/bash
# Shell script to process all .mod files in a source directory using split-imod CLI
echo "This script requires"
SRC_DIR="/mnt/efs/aimbl_2025/student_data/S-TB/Contours/modfiles"   # Source directory containing .mod files
TMP_DIR="/mnt/efs/aimbl_2025/student_data/S-TB/Contours/tmp_modtxtfiles" # Temporary directory for intermediate files
DST_DIR="/mnt/efs/aimbl_2025/student_data/S-TB/Contours/processed_modfiles" # Destination directory for processed files

mkdir -p "$DST_DIR"

for modfile in "$SRC_DIR"/*.mod; do
    filename=$(basename "$modfile")
    txtfile="$TMP_DIR/${filename%.mod}.txt"
    outpath="$DST_DIR/$filename"
    echo "Processing $modfile -> $outpath"
    model2point -inp "$modfile" -ou "$txtfile" -ob
    split-imod-txt "$txtfile" "${txtfile%.txt}_s.txt"
    # point2model -in "$txtfile" -ou "$outpath" -th 1 -m 0.10 -op
done

echo "All .mod files processed."

# Clean up temporary txt files
rm -r "$TMP_DIR"
echo "Temporary txt files removed from $TMP_DIR."
