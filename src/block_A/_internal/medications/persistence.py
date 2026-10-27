"""Private generations and locked publication; consumers fail closed after interruption."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager
from uuid import uuid4


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def manifest_path(output):
    return Path(output).with_suffix('.manifest.json')


@contextmanager
def publication_lock(output, exclusive=False):
    path=Path(output).with_suffix('.lock')
    with path.open('a+b' if exclusive else 'rb') as stream:
        if exclusive: os.chmod(path,0o600)
        fcntl.flock(stream,fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def private_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    os.chmod(path,0o600)


def publish(features, manifest, output, tables, qc, audit_root):
    from .contracts import validate
    import pandas as pd
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    restricted=Path(audit_root)/'restricted'
    restricted.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(restricted,0o700)
    generations=restricted/'generations'; generations.mkdir(exist_ok=True,mode=0o700); os.chmod(generations,0o700)
    with publication_lock(output,exclusive=True), tempfile.TemporaryDirectory(dir=output.parent,prefix='.08-stage-') as directory:
        os.chmod(directory,0o700); stage=Path(directory)
        table=stage/'features.parquet'; features.to_parquet(table,index=False); os.chmod(table,0o600)
        validate(pd.read_parquet(table),features)
        generation=uuid4().hex
        details=generations/generation;details.mkdir(mode=0o700)
        # Preserve the previous public dataset and manifest in this restricted
        # generation before replacing them (including v1.0 -> v1.1 upgrades).
        for original, archive_name in (
            (output, "08_previous_episode_features.parquet"),
            (manifest_path(output), "08_previous_episode_features.manifest.json"),
        ):
            if original.is_file():
                archive = details / archive_name
                archive.write_bytes(original.read_bytes())
                os.chmod(archive, 0o600)
        # Every run retains its own long products. Never overwrite an older review trail.
        for name,frame in tables.items():
            path=details/name;frame.to_parquet(path,index=False);os.chmod(path,0o600)
        private_json(details/'08_medication_reconciliation_qc.json',qc)
        manifest=dict(manifest,sha256=sha256(table),columns=list(features),
            dtypes={c:str(features[c].dtype) for c in features},restricted_generation=generation,
            restricted_outputs_sha256={p.name:sha256(p) for p in details.iterdir()})
        metadata=stage/'manifest.json';private_json(metadata,manifest)
        qcpath=stage/'qc.json';private_json(qcpath,qc)
        old_table=output.read_bytes() if output.exists() else None
        old_meta=manifest_path(output).read_bytes() if manifest_path(output).exists() else None
        current=restricted/'current';old_current=os.readlink(current) if current.is_symlink() else None
        temporary_link=restricted/('.current-'+generation)
        temporary_link.symlink_to(Path('generations')/generation,target_is_directory=True)
        try:
            os.replace(table,output)
            os.replace(metadata,manifest_path(output))
            os.replace(temporary_link,current)
        except BaseException:
            for path,content in ((output,old_table),(manifest_path(output),old_meta)):
                if content is None:path.unlink(missing_ok=True)
                else:path.write_bytes(content);os.chmod(path,0o600)
            if old_current is not None and (not current.is_symlink() or os.readlink(current)!=old_current):
                temporary_link.unlink(missing_ok=True);temporary_link.symlink_to(old_current,target_is_directory=True);os.replace(temporary_link,current)
            raise
        finally:
            temporary_link.unlink(missing_ok=True)
        for path in details.iterdir():
            alias=restricted/path.name
            if not alias.exists():alias.symlink_to(Path('current')/path.name)
        os.replace(qcpath,Path(audit_root)/'08_medication_publication_qc.json')
    return manifest
