#!/usr/bin/env bash
# Downloads both datasets into ./data.
set -euo pipefail
DATA_DIR="${DATA_DIR:-$(dirname "$0")/../data}"
mkdir -p "$DATA_DIR"
# Hillstrom (MineThatData 2008)
curl -fsSL -o "$DATA_DIR/hillstrom.csv.gz" https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz
# MegaFon uplift (scikit-uplift mirror)
curl -fsSL -o "$DATA_DIR/megafon.csv.gz" https://sklift.s3.eu-west-2.amazonaws.com/megafon_dataset.csv.gz
echo "Data saved to $DATA_DIR"
