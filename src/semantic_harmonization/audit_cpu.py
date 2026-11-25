"""Run the deterministic audit locally without publishing a candidate."""
import os
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd

from .compare_runs import review
from .inference import force_offline, routing
from .inventory import inventory
from .resources import verify, vocabularies
from .run_pipeline import build_identity, inputs_for
from .semantic_facts import parse_mapping, verify_contract
from .settings import digest, load_config, write_json


def main() -> None:
    os.umask(0o077)
    force_offline()
    try:
        cfg = load_config()
        hashes = inputs_for(cfg)
        resources = verify(cfg)
        identity = build_identity(cfg, resources, hashes)
        paths = cfg["paths"]
        context, events = [pd.read_parquet(paths[key]) if key in hashes else None for key in ("context", "source_events")]
        pairs, _, variables = inventory(pd.read_parquet(paths["integrated"]), context, events, hashes, cfg)
        vocab = vocabularies(resources)
        facts, segments, routes = [], [], {}
        for row in pairs:
            semantic, parts, parsed = parse_mapping(row, cfg, vocab)
            facts.extend(parsed)
            segments.extend(parts)
            routes[row["mapping_id"]] = routing(row, semantic, parts, parsed)
        verify_contract(pairs, segments, facts)
        if inputs_for(cfg) != hashes:
            raise ValueError("Source inputs changed during CPU audit")
        root = Path(paths["obj1_root"]) / "work" / "semantic_harmonization" / "audit_v2"
        root.mkdir(parents=True, exist_ok=True)
        run_id = digest(identity)
        out = Path(mkdtemp(prefix=f"cpu-{run_id}-", dir=root))
        summary = review(out, cfg, hashes, pairs, segments, facts, facts, routes, {}, variables, identity["baseline"])
        write_json(out / "validation_summary.json", dict(status="deterministic CPU audit", identity=identity, v2=summary))
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Semantic CPU audit stopped ({type(error).__name__}).") from None
    print(f"Deterministic CPU audit complete. Run identity {run_id}. Report {out.name}")


if __name__ == "__main__":
    main()
