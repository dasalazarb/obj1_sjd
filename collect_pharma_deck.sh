#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
    printf '%s\n' 'Usage: bash collect_pharma_deck.sh' >&2
    exit 2
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec python3 "$root/src/studies/pharma/stage05/deck_collection.py"
