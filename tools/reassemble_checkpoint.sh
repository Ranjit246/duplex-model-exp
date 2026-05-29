#!/usr/bin/env bash
# Reassemble large checkpoint files that were uploaded in <50GB chunks.
#
# After downloading a step_<N>/ folder from the hub, run:
#   bash reassemble_checkpoint.sh /path/to/step_<N>
#
# It walks the folder, finds any file with siblings named "<file>.part_00",
# "<file>.part_01", ..., concatenates them back into <file>, then removes the parts.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint_dir>" >&2
  exit 1
fi

ckpt_dir=$1
if [[ ! -d "$ckpt_dir" ]]; then
  echo "not a directory: $ckpt_dir" >&2
  exit 1
fi

# Find unique base names that have .part_* siblings.
mapfile -t bases < <(
  find "$ckpt_dir" -type f -name '*.part_[0-9][0-9]' \
    | sed -E 's/\.part_[0-9]+$//' \
    | sort -u
)

if [[ ${#bases[@]} -eq 0 ]]; then
  echo "no chunked files found under $ckpt_dir"
  exit 0
fi

for base in "${bases[@]}"; do
  parts=( "$base".part_* )
  # Sort numerically (split-by-suffix gives lexicographic — same since suffixes are zero-padded).
  echo "reassembling $base from ${#parts[@]} chunks"
  cat "${parts[@]}" > "$base"
  rm -f "${parts[@]}"
  echo "  -> $base ($(stat -c %s "$base") bytes)"
done

echo "done"
