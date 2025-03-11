#!/bin/bash

# Source directory where original patient data is stored
SRC_DIR="/mnt/nas6/data/Target/mrct1000_nobatch"

# Destination directories for known and unknown batches
DEST_KNOWN="/mnt/nas6/data/Target/symlinked_batches_mrct_1000/known_no_issues"
DEST_UNKNOWN="/mnt/nas6/data/Target/symlinked_batches_mrct_1000/unknown"

# Create the destination directories if they don't exist
mkdir -p "$DEST_KNOWN" "$DEST_UNKNOWN"

# Symlink known patients (0001-0519)
for i in $(seq -f "%04g" 1 519); do
    patient="sub-PAT-$i"
    src_path="$SRC_DIR/$patient"

    if [[ -d "$src_path" ]]; then
        ln -s "$src_path" "$DEST_KNOWN/$patient"
        echo "Symlinked $patient -> $DEST_KNOWN"
    else
        echo "Skipping missing patient: $patient"
    fi
done

# Handle all other patients except sub-PAT-0520
for src_path in "$SRC_DIR"/sub-PAT-*; do
    patient=$(basename "$src_path")

    # Skip if already symlinked or if it's the bad patient (0520)
    if [[ "$patient" == "sub-PAT-0520" ]] || [[ -L "$DEST_KNOWN/$patient" ]]; then
        continue
    fi

    ln -s "$src_path" "$DEST_UNKNOWN/$patient"
    echo "Symlinked $patient -> $DEST_UNKNOWN"
done

echo "Symlinking complete!"
