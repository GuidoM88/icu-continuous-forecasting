#!/usr/bin/env bash
# Downloads the PhysioNet/CinC Challenge 2012 dataset ("Predicting Mortality
# of ICU Patients"). Unlike MIMIC-III/IV, this Challenge dataset is listed as
# "Open Access" on PhysioNet -- no credentialing / data-use agreement is
# required, just run this script.
#
# Source: https://physionet.org/content/challenge-2012/1.0.0/
#
# NB: this sandbox's network egress does not include physionet.org, so this
# script needs to be run on YOUR machine, not in here. It downloads set-a
# (4000 records WITH outcome labels -- the only split used by this repo),
# plus the outcomes file.

set -euo pipefail
cd "$(dirname "$0")"

BASE_URL="https://physionet.org/files/challenge-2012/1.0.0"
mkdir -p raw

echo "Downloading set-a (training, with outcomes)..."
wget -r -N -c -np -nH --cut-dirs=3 -P raw \
    -A "*.txt" \
    "${BASE_URL}/set-a/"

echo "Downloading Outcomes-a.txt..."
wget -N -c -P raw "${BASE_URL}/Outcomes-a.txt"

echo "Done. Expect: data/raw/set-a/*.txt (4000 files) and data/raw/Outcomes-a.txt"
echo "Sanity check:"
ls raw/set-a | wc -l
wc -l raw/Outcomes-a.txt
