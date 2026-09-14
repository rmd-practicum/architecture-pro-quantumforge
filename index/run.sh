#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
"$SCRIPT_DIR/../venv/bin/python3" index.py \
    --chroma-host localhost --chroma-port 8000 \
    --docs "$SCRIPT_DIR/../knowledge_base/1_replace/out"
