#!/usr/bin/env bash
# Validate the runtime repository with owned files, including source archives.
sjd_obj1_root() {
    local root
    root="$(cd -- "$1" && pwd -P)" || return
    if [[ ! -f "$root/src/block_A/10_build_integrated_longitudinal_dataset.py" ||
          ! -f "$root/src/semantic_harmonization/run_pipeline.py" ]]; then
        printf '%s\n' "Expected the obj1_sjd repository root." >&2
        return 1
    fi
    printf '%s\n' "$root"
}
