#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python data/download_physionet2012.py