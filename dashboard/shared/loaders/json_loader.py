import json
from shared.loaders.registry import SchemaError


def read_json_bytes(data, entry):
    obj = json.loads(data)
    if not isinstance(obj, dict):
        raise SchemaError("JSON output must be an object")
    missing = set(entry.spec.get("fields", [])) - set(obj)
    if missing:
        raise SchemaError("Missing JSON fields: " + ", ".join(sorted(missing)))
    return obj
