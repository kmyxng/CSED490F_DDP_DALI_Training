#!/bin/bash

########################################################
# Build the Lab 3 submission zip (spec: Submission slide).
#   zip : lab3_DDP_DALI_team{N}.zip
#   contents : handler/ , launch.sh , lab3_DDP_DALI_team{N}.pdf
#
# Usage (at the repository root):
#   bash report/make_submission.sh <team_number> [en|ko]   # default: en
########################################################

set -euo pipefail

TEAM=${1:?"usage: bash report/make_submission.sh <team_number> [en|ko]"}
LANG_SEL=${2:-en}
NAME="lab3_DDP_DALI_team${TEAM}"

case "$LANG_SEL" in
    en) SRC="lab3_report.tex" ;;
    ko) SRC="lab3_report_ko.tex" ;;
    *) echo "language must be 'en' or 'ko'"; exit 1 ;;
esac

cd "$(dirname "$0")/.."

# The team number itself lives in report/tex/preamble.tex (\team)
grep -q "\\\\newcommand{\\\\team}{${TEAM}}" report/tex/preamble.tex \
    || { echo "report/tex/preamble.tex does not say Team ${TEAM}"; exit 1; }

# Compile the report (XeLaTeX for Hangul), twice for references
mkdir -p report/build
for _ in 1 2; do
    (cd report && xelatex -interaction=nonstopmode -halt-on-error -output-directory=build -jobname="${NAME}" "$SRC" > /dev/null)
done

STAGE="report/build/${NAME}"
rm -rf "$STAGE" "report/build/${NAME}.zip"
mkdir -p "$STAGE"
rsync -a --exclude "__pycache__" handler "$STAGE/"
cp scripts/launch.sh "$STAGE/"
cp "report/build/${NAME}.pdf" "$STAGE/"

(cd "$STAGE" && zip -qr "../${NAME}.zip" handler launch.sh "${NAME}.pdf")
echo "[submission] report/build/${NAME}.zip (${LANG_SEL})"
unzip -l "report/build/${NAME}.zip"
