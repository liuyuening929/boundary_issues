#!/bin/bash
# Shell script to process all .mod files in a source directory using split-imod CLI
SRC_DIR="/mnt/efs/aimbl_2025/student_data/S-TB/Contours/modfiles"   # Source directory containing .mod files
TMP_DIR="/mnt/efs/aimbl_2025/student_data/S-TB/Contours/tmp_modtxtfiles" # Temporary directory for intermediate files
DST_DIR="/mnt/efs/aimbl_2025/student_data/S-TB/Contours/processed_modfiles" # Destination directory for processed files

mkdir -p "$DST_DIR"
mkdir -p "$TMP_DIR"

for modfile in "$SRC_DIR"/*.mod; do
    filename=$(basename "$modfile")
    txtfile="$TMP_DIR/${filename%.mod}.txt"
    # Switch 18.00Apx to 10.00Apx in output filename
    model2point -inp "$modfile" -ou "$txtfile" -ob
    split-imod-txt "$txtfile" "$txtfile"
    point2model -in "$txtfile" -ou "$outpath" -th 1 -op -m 1.8 -pi 18.,18.,18.
done

cd $DST_DIR
for prefix in $(ls *.mod | sed -E 's/(_[0-9]+)?\.mod$//' | sort | uniq); do
    files=$(ls ${prefix}*.mod 2>/dev/null | sort)
    if [ $(echo "$files" | wc -l) -gt 1 ]; then
        echo "Merging files for prefix $prefix: $files"
        imodjoin $files "${prefix}.mod"
        for f in $files; do
            if [[ "$f" =~ _[0-9]+\.mod$ ]]; then
                rm "$f";
            fi
        done
    fi
done
cd - >/dev/null
echo "All .mod files processed."

# Clean up temporary txt files
rm -r "$TMP_DIR"
echo "Temporary txt files removed from $TMP_DIR."

