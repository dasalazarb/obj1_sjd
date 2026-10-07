from dataclasses import dataclass, field
import hashlib
import json
import re
from shared.loaders.registry import MissingOutput, SchemaError
from shared.loaders.json_loader import read_json_bytes


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(path):
    try:
        stat = path.stat()
        if path.suffix == '.csv':
            from shared.loaders.csv_loader import check_header
            try:
                check_header(path)
            except SchemaError as exc:
                return ('rejected', str(exc))
        content_hash = digest(path.read_bytes()) if path.suffix in {".csv", ".json"} else None
        return (stat.st_size, stat.st_mtime_ns, content_hash)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, UnicodeError) as exc:
        return ('unreadable', str(exc))


@dataclass
class Run:
    key: str
    registry: object
    fingerprints: dict
    tables: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    run_id: str = ""
    manifest: dict = field(default_factory=dict)

    def data(self, key):
        if key in self.errors:
            raise MissingOutput(self.errors[key])
        if key not in self.tables:
            raise MissingOutput("Not available for this run: " + self.registry.get(key).path)
        return self.tables[key]

    def figure_bytes(self, key):
        path = self.registry.path(key)
        if fingerprint(path) != self.fingerprints.get(key):
            raise MissingOutput("Figure changed during this session — Refresh")
        if not path.is_file():
            raise MissingOutput("Figure not found for this run")
        data = path.read_bytes()
        if fingerprint(path) != self.fingerprints.get(key):
            raise MissingOutput("Figure changed while reading — Refresh")
        return data


def current_key(registry):
    stamps = {key: fingerprint(registry.path(key)) for key in registry.entries}
    catalog=getattr(registry,'catalog',[])
    figures=getattr(registry,'figures',[])
    key = digest(json.dumps({'files':stamps,'catalog':catalog,'figures':figures}, sort_keys=True).encode())[:12]
    return key, stamps


def capture_run(registry):
    from shared.loaders.csv_loader import read_csv_bytes
    key, stamps = current_key(registry)
    run = Run(key, registry, stamps)
    for rid, entry in registry.entries.items():
        if entry.spec["kind"] == "figure":
            continue
        try:
            if entry.spec['kind']=='csv':
                from shared.loaders.csv_loader import check_header
                check_header(registry.path(rid))
            data = registry.path(rid).read_bytes()
            run.tables[rid] = read_csv_bytes(data, entry) if entry.spec["kind"] == "csv" else read_json_bytes(data, entry)
        except (OSError, ValueError) as exc:
            run.errors[rid] = str(exc)
    if current_key(registry)[0] != key:
        raise MissingOutput("Outputs changed while capturing this run — Refresh")
    run.provenance = run.tables.get("prov11", {})
    if stamps.get('run_manifest') is not None and 'run_manifest' in run.errors:
        raise SchemaError('Registered run manifest cannot be read: ' + run.errors['run_manifest'])
    manifest = run.tables.get("run_manifest", {})
    if manifest:
        if manifest.get("status") != "complete":
            raise SchemaError("The registered run manifest is not complete")
        if not isinstance(manifest.get('run_id'), str) or not manifest['run_id']:
            raise SchemaError('Run manifest requires a nonempty run_id')
        assets = manifest.get('assets')
        if not isinstance(assets, list):
            raise SchemaError('Run manifest assets must be a list')
        run.run_id = manifest["run_id"]
        run.manifest = manifest
        covered = set()
        for asset in assets:
            if not isinstance(asset, dict) or any(not isinstance(asset.get(field), str) for field in ['id', 'path', 'sha256']):
                raise SchemaError('Run manifest asset requires string id, path and sha256')
            rid = asset["id"]
            if rid in covered or rid == 'run_manifest':
                raise SchemaError('Duplicate or self-referencing run manifest asset')
            covered.add(rid)
            if not re.fullmatch(r'[0-9a-f]{64}', asset['sha256']):
                raise SchemaError('Run manifest asset checksum must be SHA-256')
            entry = registry.get(rid)
            if asset["path"] != entry.path:
                raise SchemaError("Run manifest path differs from registry")
            if rid in run.errors:
                raise SchemaError('Run manifest asset is unavailable: ' + rid)
            if digest(registry.path(rid).read_bytes()) != asset["sha256"]:
                raise SchemaError("Run manifest asset checksum mismatch")
        # A complete manifest must cover every present evidence asset, avoiding mixed runs.
        present = {rid for rid, stamp in stamps.items() if stamp is not None and rid != "run_manifest"}
        if not present.issubset(covered):
            raise SchemaError("Complete run manifest does not cover all present registered evidence")
    if current_key(registry)[0] != key:
        raise MissingOutput('Outputs changed while validating this run — Refresh')
    return run
