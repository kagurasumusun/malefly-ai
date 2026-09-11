#!/bin/sh
# Fetch the public MaleCNS v1.0 flat-connectome annotations (CC-BY).
# Source: https://male-cns.janelia.org/download/  (gs://flyem-male-cns)
# The full synaptic edge list (6.8 GB) is intentionally NOT fetched here;
# aggregate statistics come from the annotation table + published numbers.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/malecns
curl -L --fail -o data/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather"
python3 tools/malecns_stats.py
