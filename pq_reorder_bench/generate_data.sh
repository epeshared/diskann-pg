#!/usr/bin/env bash
set -euo pipefail

APPS=/home/xtang/DiskANN-epeshared/build/apps
mkdir -p tmpdata

$APPS/utils/rand_data_gen --data_type float --output_file tmpdata/base_f32_10K.bin -D 768 -N 10000 --norm 1.0
$APPS/utils/rand_data_gen --data_type float --output_file tmpdata/query_f32_100.bin -D 768 -N 100 --norm 1.0