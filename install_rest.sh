#!/usr/bin/env bash

set -o xtrace
set -euo pipefail

shopt -s extglob

PKGS="$(poetry env list --full-path | cut -d' ' -f1)"
if [ -z "$PKGS" ]; then
    SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
else
    SITE="$(echo $PKGS/lib/python*/site-packages)"
fi
echo "$(pwd)/submodules/" > "$SITE/lighttrack.pth"
echo "$(pwd)/submodules/lighttrack" > "$SITE/lighttrackinner.pth"
echo "$(pwd)/submodules/lighttrack/graph" > "$SITE/lighttrackgraph.pth"
echo "$(pwd)/submodules/lighttrack/graph/torchlight" > "$SITE/torchlight.pth"

poetry run pip install -e $(pwd)/submodules/opencv_wrapper
poetry run pip install -e $(pwd)/submodules/ufunclab
