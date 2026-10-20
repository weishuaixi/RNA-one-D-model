"""Explicit allowlist release builder, with verified offline model/data inputs."""

import hashlib
import io
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "data/processed/rna_linker_v3_training.csv": "5220058e6074c8fed218f968aa8d1c5d69dbdb72e856fb3b76f444eb33249de1",
    "data/processed/rna_linker_v3_clusters.tsv": "50eab29ce5494698f2e7db0ab6ab0fa8e54bbb9db0128e51b25bcc83e07a1a4a",
    ".cache/torch/hub/checkpoints/RNA-FM_pretrained.pth": "5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    for name, expected in REQUIRED.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Input hash mismatch: " + name)
    names = [
        "train.py",
        "generate_scaffold.py",
        "validate_scaffolds.py",
        "benchmark_scaffolds.py",
        "requirements.txt",
        "pyproject.toml",
        "pytest.ini",
        "README.md",
        "constraints-v5-server.txt",
    ]
    scripts = [
        "setup_v5_retrain.sh",
        "start_v5_retrain.sh",
        "export_v5_best.py",
        "train_linker_v5_offline.sh",
        "train_linker_offline.sh",
        "benchmark_linker_v5_fast_offline.sh",
        "check_linker_v5_validation.py",
        "validate_linker_v5_expanded.sh",
        "diagnose_linker_v5_conditioning.py",
        "diagnose_linker_v5_conditioning.sh",
    ]
    names += ["scripts/" + name for name in scripts]
    for folder, pattern in [("rna_scaffold", "*.py"), ("configs", "*.yaml")]:
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob(pattern)]
    names += ["tests/" + name for name in ["test_decoder_bias.py", "test_conditioning_diagnostic.py"]]
    names += list(REQUIRED)
    names = sorted(set(names))
    output = ROOT / "project_artifacts/RNA-V5-RETRAIN-FULL-20260910.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    digests = []
    with output.open("xb") as raw, tarfile.open(fileobj=raw, mode="w:gz", compresslevel=1) as archive:
        for name in names:
            path = ROOT / name
            if path.is_symlink():
                raise ValueError("Refuse symlink: " + name)
            if path.suffix == ".pth":
                digest = sha(path)
                archive.add(path, arcname="RNA-V5-RETRAIN/" + name, recursive=False)
            else:
                data = path.read_bytes()
                if path.suffix == ".sh":
                    data = data.replace(b"\r\n", b"\n")
                digest = hashlib.sha256(data).hexdigest()
                info = tarfile.TarInfo("RNA-V5-RETRAIN/" + name)
                info.size = len(data)
                info.mode = 0o755 if path.suffix == ".sh" else 0o644
                archive.addfile(info, io.BytesIO(data))
            digests.append(digest + "  " + name)
        data = ("\n".join(digests) + "\n").encode()
        info = tarfile.TarInfo("RNA-V5-RETRAIN/PACKAGE_SHA256SUMS")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    # Verify archived bytes, not just source files.
    expected = dict(line.split("  ", 1)[::-1] for line in digests)
    with tarfile.open(output) as archive:
        for member in archive:
            name = member.name.removeprefix("RNA-V5-RETRAIN/")
            if name not in expected:
                continue
            digest = hashlib.sha256()
            with archive.extractfile(member) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected[name]:
                raise ValueError("Archive mismatch: " + name)
    digest = sha(output)
    output.with_suffix(".gz.sha256").write_text(digest + "  " + output.name + "\n", encoding="utf-8")
    print("Verified files:", len(names), "bytes:", output.stat().st_size, "SHA256:", digest, flush=True)
    print(output)


if __name__ == "__main__":
    main()
