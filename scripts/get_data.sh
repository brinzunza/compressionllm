#!/usr/bin/env bash
# Download and verify the text8 corpus (100 MB of cleaned Wikipedia text) into data/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -sSL -o data/text8.zip http://mattmahoney.net/dc/text8.zip
unzip -q -o data/text8.zip -d data
rm data/text8.zip
echo "0dc3edebc970dcc96137e7deda4d9995af9d93de  data/text8" | shasum -a 1 -c -
