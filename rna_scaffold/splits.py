from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

from rna_scaffold.records import RnaSequenceRecord

PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitManifest:
    partitions: dict[str, tuple[str, ...]]
    sequence_hashes: dict[str, str]
    seed: int

    def partition_for(self, target_id: str) -> str:
        matches = [name for name, members in self.partitions.items() if target_id in members]
        if len(matches) != 1:
            raise KeyError(f"target_id must occur in exactly one partition: {target_id}")
        return matches[0]


def _group_key(record: RnaSequenceRecord) -> str:
    return f"family:{record.family}" if record.family else f"sequence:{record.sequence_sha256}"


def _partition_for_group(group_key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def build_family_disjoint_manifest(
    records: Sequence[RnaSequenceRecord], seed: int = 42
) -> SplitManifest:
    grouped: dict[str, list[RnaSequenceRecord]] = {}
    for record in records:
        grouped.setdefault(_group_key(record), []).append(record)
    partitions: dict[str, list[str]] = {name: [] for name in PARTITIONS}
    for group_key in sorted(grouped):
        partition = _partition_for_group(group_key, seed)
        partitions[partition].extend(record.target_id for record in grouped[group_key])
    return validate_manifest(records, partitions, seed=seed)


def validate_manifest(
    records: Sequence[RnaSequenceRecord],
    partitions: Mapping[str, Sequence[str]],
    seed: int = 42,
) -> SplitManifest:
    if set(partitions) != set(PARTITIONS):
        raise ValueError(f"manifest partitions must be {PARTITIONS}")
    by_id = {record.target_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("record target IDs must be unique")
    owner: dict[str, str] = {}
    for partition, members in partitions.items():
        for target_id in members:
            if target_id not in by_id:
                raise ValueError(f"unknown target ID in manifest: {target_id}")
            if target_id in owner:
                raise ValueError(f"target ID occurs in multiple partitions: {target_id}")
            owner[target_id] = partition
    if set(owner) != set(by_id):
        missing = sorted(set(by_id) - set(owner))
        raise ValueError(f"manifest omits target IDs: {missing}")

    family_owner: dict[str, str] = {}
    sequence_owner: dict[str, str] = {}
    for target_id, partition in owner.items():
        record = by_id[target_id]
        if record.family:
            previous = family_owner.setdefault(record.family, partition)
            if previous != partition:
                raise ValueError(f"family crosses partitions: {record.family}")
        previous = sequence_owner.setdefault(record.sequence_sha256, partition)
        if previous != partition:
            raise ValueError(f"exact sequence overlap across partitions: {target_id}")

    normalized = {
        name: tuple(sorted(partitions[name]))
        for name in PARTITIONS
    }
    hashes = {record.target_id: record.sequence_sha256 for record in records}
    return SplitManifest(normalized, hashes, seed)
