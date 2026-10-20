"""Build a source-only handoff; never include local data, weights or Git history."""

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    ".gitignore",
    ".gitattributes",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "constraints-v5-server.txt",
    "pytest.ini",
    "train.py",
    "generate_scaffold.py",
    "validate_scaffolds.py",
    "benchmark_scaffolds.py",
    "data/.gitkeep",
)
SOURCE_FOLDERS = {
    "rna_scaffold": {".py"},
    "scripts": {".py", ".sh"},
    "tests": {".py"},
    "configs": {".yaml"},
}


def build(output: Path) -> dict:
    paths = [ROOT / name for name in ROOT_FILES]
    for folder, suffixes in SOURCE_FOLDERS.items():
        paths.extend(
            path
            for path in (ROOT / folder).rglob("*")
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts
        )
    payloads = {}
    for path in sorted(set(paths)):
        path.resolve(strict=True).relative_to(ROOT.resolve())
        if path.is_symlink():
            raise ValueError(f"Refuse symlink: {path}")
        name = path.relative_to(ROOT).as_posix()
        payloads[name] = path.read_bytes().replace(b"\r\n", b"\n")
    manifest = {
        "kind": "source_only_gitlab_handoff",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_release": "V5",
        "includes_weights": False,
        "includes_training_data": False,
        "benchmark_evidence": "Historical summaries in README.md; not a new benchmark run.",
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
    }
    payloads["SOURCE_MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checksum = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or checksum.exists():
        raise FileExistsError("Refusing to overwrite an existing source package or checksum")
    with output.open("xb") as raw, zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            entry = zipfile.ZipInfo(name)
            entry.create_system = 3
            entry.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(payloads):
            raise ValueError("Archive verification failed")
        for name, data in payloads.items():
            if archive.read(name) != data:
                raise ValueError(f"Archive content mismatch: {name}")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    with checksum.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}  {output.name}\n")
    print(
        json.dumps(
            {"archive": str(output), "files": len(payloads), "bytes": output.stat().st_size, "sha256": digest}, indent=2
        )
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "exports" / (
        "RNA-V5-IGEM-GITLAB-SOURCE-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + ".zip"
    )
    build(output)


if __name__ == "__main__":
    main()
