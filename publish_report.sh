#!/bin/zsh
# Copy a finished report PDF into deliverables/ with the next version number.
# usage: ./publish_report.sh <run-dir> <Student Name>
#   ./publish_report.sh wwrag/runs/v1/aadya_aggarwal "Aadya Aggarwal"
cd "$(dirname "$0")"
SRC="$1/report.pdf"
NAME="${2// /_}"
[ -f "$SRC" ] || { echo "no report at $SRC"; exit 1; }
mkdir -p deliverables
n=1
while [ -e "deliverables/${NAME}_USC_v${n}.pdf" ]; do n=$((n+1)); done
OUT="deliverables/${NAME}_USC_v${n}.pdf"
cp "$SRC" "$OUT"
# keep the markdown beside it so changes between versions are diffable
[ -f "$1/report.md" ] && cp "$1/report.md" "deliverables/${NAME}_USC_v${n}.md"
echo "$OUT"
ls -la "$OUT"
