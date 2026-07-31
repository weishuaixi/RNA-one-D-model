import os
from pathlib import Path
from unittest.mock import patch

import torch

from rna_scaffold_3d.data import (
    _align_parsed_structure,
    _assemble_cif_rna_atoms,
    StanfordRnaAllAtomDataset,
    StanfordRna3DDataset,
    _load_sequences,
    collate_3d_batch,
    load_stanford_rna_3d_records,
    parse_cif_rna_chain,
)
from rna_scaffold_3d.rna_atoms import RNA_ATOM_NAMES, atom_names_for_base


def test_load_stanford_rna_3d_records_joins_sequences_and_coordinates(tmp_path: Path):
    sequences = tmp_path / "train_sequences.csv"
    labels = tmp_path / "train_labels.csv"
    sequences.write_text(
        "target_id,sequence,temporal_cutoff,description,all_sequences\n"
        "1ABC_A,AUG,2020-01-01,example,\n",
        encoding="utf-8",
    )
    labels.write_text(
        "ID,resname,resid,x_1,y_1,z_1\n"
        "1ABC_A_1,A,1,1.0,2.0,3.0\n"
        "1ABC_A_2,U,2,4.0,5.0,6.0\n"
        "1ABC_A_3,G,3,7.0,8.0,9.0\n",
        encoding="utf-8",
    )

    records = load_stanford_rna_3d_records(sequences, labels)

    assert len(records) == 1
    assert records[0].target_id == "1ABC_A"
    assert records[0].sequence == "AUG"
    assert records[0].coords.shape == (3, 3)
    assert records[0].coord_mask.tolist() == [True, True, True]


def test_load_sequences_reads_only_target_id_and_sequence_columns(tmp_path: Path):
    sequences = tmp_path / "train_sequences.csv"
    sequences.write_text(
        "target_id,sequence,temporal_cutoff,description,all_sequences\n"
        "1ABC_A,AUG,2020-01-01,example,\"bad quote that should be ignored\n"
        "2ABC_A,GCA,2020-01-01,example,\"another bad quote\n",
        encoding="utf-8",
    )

    loaded = _load_sequences(sequences, max_records=None, max_sequence_length=None)

    assert loaded == {"1ABC_A": "AUG", "2ABC_A": "GCA"}


def test_load_stanford_rna_3d_records_masks_missing_coordinates(tmp_path: Path):
    sequences = tmp_path / "train_sequences.csv"
    labels = tmp_path / "train_labels.csv"
    sequences.write_text(
        "target_id,sequence,temporal_cutoff,description,all_sequences\n"
        "1ABC_A,AU,2020-01-01,example,\n",
        encoding="utf-8",
    )
    labels.write_text(
        "ID,resname,resid,x_1,y_1,z_1\n"
        "1ABC_A_1,A,1,1.0,2.0,3.0\n"
        "1ABC_A_2,U,2,-1e18,-1e18,-1e18\n",
        encoding="utf-8",
    )

    records = load_stanford_rna_3d_records(sequences, labels)

    assert records[0].coord_mask.tolist() == [True, False]


def test_stanford_rna_3d_dataset_and_collate_pad_variable_lengths(tmp_path: Path):
    sequences = tmp_path / "train_sequences.csv"
    labels = tmp_path / "train_labels.csv"
    sequences.write_text(
        "target_id,sequence,temporal_cutoff,description,all_sequences\n"
        "1ABC_A,AU,2020-01-01,example,\n"
        "2ABC_A,GCA,2020-01-01,example,\n",
        encoding="utf-8",
    )
    labels.write_text(
        "ID,resname,resid,x_1,y_1,z_1\n"
        "1ABC_A_1,A,1,1,2,3\n"
        "1ABC_A_2,U,2,4,5,6\n"
        "2ABC_A_1,G,1,7,8,9\n"
        "2ABC_A_2,C,2,10,11,12\n"
        "2ABC_A_3,A,3,13,14,15\n",
        encoding="utf-8",
    )
    dataset = StanfordRna3DDataset.from_csv(sequences, labels)

    batch = collate_3d_batch([dataset[0], dataset[1]])

    assert batch["input_ids"].shape == (2, 3)
    assert batch["coords"].shape == (2, 3, 3)
    assert batch["padding_mask"].tolist() == [[False, False, True], [False, False, False]]
    assert torch.equal(batch["coord_mask"], torch.tensor([[True, True, False], [True, True, True]]))


def test_load_stanford_rna_3d_records_filters_by_length_and_coordinate_coverage(tmp_path: Path):
    sequences = tmp_path / "train_sequences.csv"
    labels = tmp_path / "train_labels.csv"
    sequences.write_text(
        "target_id,sequence,temporal_cutoff,description,all_sequences\n"
        "GOOD_A,AUGC,2020-01-01,example,\n"
        "LONG_A,AUGCA,2020-01-01,example,\n"
        "SPARSE_A,AUGC,2020-01-01,example,\n",
        encoding="utf-8",
    )
    labels.write_text(
        "ID,resname,resid,x_1,y_1,z_1\n"
        "GOOD_A_1,A,1,1,2,3\n"
        "GOOD_A_2,U,2,4,5,6\n"
        "GOOD_A_3,G,3,7,8,9\n"
        "GOOD_A_4,C,4,10,11,12\n"
        "LONG_A_1,A,1,1,2,3\n"
        "LONG_A_2,U,2,4,5,6\n"
        "LONG_A_3,G,3,7,8,9\n"
        "LONG_A_4,C,4,10,11,12\n"
        "LONG_A_5,A,5,13,14,15\n"
        "SPARSE_A_1,A,1,1,2,3\n"
        "SPARSE_A_2,U,2,-1e18,-1e18,-1e18\n"
        "SPARSE_A_3,G,3,-1e18,-1e18,-1e18\n"
        "SPARSE_A_4,C,4,-1e18,-1e18,-1e18\n",
        encoding="utf-8",
    )

    records = load_stanford_rna_3d_records(
        sequences,
        labels,
        max_sequence_length=4,
        min_coord_coverage=0.8,
    )

    assert [record.target_id for record in records] == ["GOOD_A"]


def test_load_stanford_rna_3d_records_can_center_valid_coordinates(tmp_path: Path):
    sequences = tmp_path / "train_sequences.csv"
    labels = tmp_path / "train_labels.csv"
    sequences.write_text(
        "target_id,sequence,temporal_cutoff,description,all_sequences\n"
        "1ABC_A,AU,2020-01-01,example,\n",
        encoding="utf-8",
    )
    labels.write_text(
        "ID,resname,resid,x_1,y_1,z_1\n"
        "1ABC_A_1,A,1,10.0,0.0,0.0\n"
        "1ABC_A_2,U,2,14.0,0.0,0.0\n",
        encoding="utf-8",
    )

    records = load_stanford_rna_3d_records(
        sequences,
        labels,
        center_coordinates=True,
    )

    assert torch.allclose(records[0].coords, torch.tensor([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    assert torch.allclose(records[0].coords[records[0].coord_mask].mean(dim=0), torch.zeros(3))


def test_all_atom_dataset_reads_matching_cif_chain(tmp_path: Path):
    sequences = tmp_path / "train_sequences.csv"
    cif_dir = tmp_path / "PDB_RNA"
    cif_dir.mkdir()
    sequences.write_text(
        "target_id,sequence,temporal_cutoff,description,all_sequences\n"
        "1ABC_A,A,2020-01-01,example,\n",
        encoding="utf-8",
    )
    (cif_dir / "1abc.cif").write_text(
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "ATOM 1 P P A A 1 1.0 2.0 3.0\n"
        "ATOM 2 C \"C4'\" A A 1 4.0 5.0 6.0\n"
        "#\n",
        encoding="utf-8",
    )

    dataset = StanfordRnaAllAtomDataset.from_csv_and_cif(
        sequences_csv=sequences,
        cif_dir=cif_dir,
        min_atom_coverage=0.01,
    )
    item = dataset[0]

    assert item["coords"].shape == (1, len(RNA_ATOM_NAMES), 3)
    assert item["coord_mask"][0, RNA_ATOM_NAMES.index("P")]
    assert item["coord_mask"][0, RNA_ATOM_NAMES.index("C4'")]


def test_all_atom_dataset_target_allowlist_filters_before_cif_parsing(
    tmp_path: Path,
):
    sequences = tmp_path / "train_sequences.csv"
    cif_dir = tmp_path / "PDB_RNA"
    cif_dir.mkdir()
    sequences.write_text(
        "target_id,sequence\n1ABC_A,A\n2ABC_A,A\n",
        encoding="utf-8",
    )
    cif_text = (
        "loop_\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "P A A 1 1.0 2.0 3.0\n"
        "#\n"
    )
    (cif_dir / "1abc.cif").write_text(cif_text, encoding="utf-8")
    (cif_dir / "2abc.cif").write_text(cif_text, encoding="utf-8")

    dataset = StanfordRnaAllAtomDataset.from_csv_and_cif(
        sequences_csv=sequences,
        cif_dir=cif_dir,
        min_atom_coverage=0.01,
        target_ids={"2ABC_A"},
    )

    assert [record.target_id for record in dataset.records] == ["2ABC_A"]
    assert dataset.stats["candidate_sequences"] == 1


def test_cif_atom_selection_uses_one_model_coherent_altloc_and_ccd_parent():
    def row(
        component,
        atom,
        residue,
        x,
        *,
        model=1,
        alternate=".",
        occupancy=1.0,
    ):
        return {
            "label_comp_id": component,
            "label_atom_id": atom,
            "label_asym_id": "ENTITY_CHAIN",
            "auth_asym_id": "A",
            "label_seq_id": str(residue),
            "Cartn_x": str(x),
            "Cartn_y": "0",
            "Cartn_z": "0",
            "label_alt_id": alternate,
            "occupancy": str(occupancy),
            "pdbx_PDB_model_num": str(model),
        }

    rows = [
        row("A", "P", 1, 1.0),
        row("A", "P", 1, 99.0, model=2),
        row("A", "C1'", 1, 4.0, alternate="A", occupancy=0.4),
        row("A", "C2'", 1, 4.1, alternate="A", occupancy=0.4),
        row("A", "C3'", 1, 4.2, alternate="A", occupancy=0.4),
        row("A", "C1'", 1, 5.0, alternate="B", occupancy=0.5),
        row("A", "C2'", 1, 5.1, alternate="B", occupancy=0.5),
        row("PSU", "P", 2, 2.0),
        row("PSU", "O5'", 2, 3.0),
        row("PSU", "OP1", 2, 20.0, occupancy=0.0),
        row("5MC", "N1", 3, 6.0),
        row("5MC", "N4", 3, 7.0),
        row("5MC", "C2", 3, 8.0),
    ]

    parsed = _assemble_cif_rna_atoms(
        rows, "A", component_parents={"PSU": "U"}
    )

    assert parsed is not None
    sequence, coords, mask = parsed
    assert sequence == "AUC"
    assert torch.isclose(coords[0, RNA_ATOM_NAMES.index("P"), 0], torch.tensor(1.0))
    assert torch.isclose(coords[0, RNA_ATOM_NAMES.index("C1'"), 0], torch.tensor(4.0))
    assert torch.isclose(coords[0, RNA_ATOM_NAMES.index("C2'"), 0], torch.tensor(4.1))
    assert torch.isclose(coords[1, RNA_ATOM_NAMES.index("P"), 0], torch.tensor(2.0))
    assert not mask[1, RNA_ATOM_NAMES.index("OP1")]
    assert torch.isclose(
        coords[2, RNA_ATOM_NAMES.index("N4"), 0], torch.tensor(7.0)
    )


def test_cif_parser_reads_ccd_parent_altloc_occupancy_and_first_model(
    tmp_path: Path,
):
    cif = tmp_path / "modified.cif"
    cif.write_text(
        "data_modified\n"
        "loop_\n"
        "_chem_comp.id\n"
        "_chem_comp.mon_nstd_parent_comp_id\n"
        "PSU ?\n"
        "#\n"
        "loop_\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.occupancy\n"
        "_atom_site.pdbx_PDB_model_num\n"
        "PSU P ENTITY A 1 1 0 0 . 1.0 1\n"
        "PSU \"C1'\" ENTITY A 1 4 0 0 A 0.4 1\n"
        "PSU \"C1'\" ENTITY A 1 5 0 0 B 0.6 1\n"
        "PSU P ENTITY A 1 99 0 0 . 1.0 2\n"
        "ALA N ENTITY A 2 10 0 0 . 1.0 1\n"
        "ALA C ENTITY A 2 11 0 0 . 1.0 1\n"
        "#\n",
        encoding="utf-8",
    )

    parsed = parse_cif_rna_chain(cif, "A")

    assert parsed is not None
    sequence, coords, _ = parsed
    assert sequence == "U"
    assert coords.shape[0] == 1
    assert torch.isclose(
        coords[0, RNA_ATOM_NAMES.index("P"), 0], torch.tensor(1.0)
    )
    assert torch.isclose(
        coords[0, RNA_ATOM_NAMES.index("C1'"), 0], torch.tensor(5.0)
    )


def test_cif_parser_uses_expected_sequence_to_resolve_label_auth_collision(
    tmp_path: Path,
):
    cif = tmp_path / "ambiguous_chain.cif"
    cif.write_text(
        "data_ambiguous\n"
        "loop_\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "C P A X 1 10 0 0\n"
        "C \"C1'\" A X 1 11 0 0\n"
        "U P B A 1 20 0 0\n"
        "U \"C1'\" B A 1 21 0 0\n"
        "#\n",
        encoding="utf-8",
    )

    legacy_precedence = parse_cif_rna_chain(cif, "A")
    resolved = parse_cif_rna_chain(
        cif, "A", expected_sequence="U"
    )

    assert legacy_precedence is not None
    assert legacy_precedence[0] == "C"
    assert resolved is not None
    sequence, coords, _ = resolved
    assert sequence == "U"
    assert torch.isclose(
        coords[0, RNA_ATOM_NAMES.index("P"), 0], torch.tensor(20.0)
    )


def test_all_atom_dataset_resolves_label_auth_chain_collision(
    tmp_path: Path,
):
    sequences = tmp_path / "train_sequences.csv"
    cif_dir = tmp_path / "PDB_RNA"
    cif_dir.mkdir()
    sequences.write_text(
        "target_id,sequence\n1ABC_A,U\n",
        encoding="utf-8",
    )
    (cif_dir / "1abc.cif").write_text(
        "loop_\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "C P A X 1 10 0 0\n"
        "U P B A 1 20 0 0\n"
        "#\n",
        encoding="utf-8",
    )

    dataset = StanfordRnaAllAtomDataset.from_csv_and_cif(
        sequences_csv=sequences,
        cif_dir=cif_dir,
        min_atom_coverage=0.01,
    )

    assert len(dataset) == 1
    assert dataset.records[0].sequence == "U"
    assert dataset.stats["exact_match"] == 1


def test_atom_coverage_denominator_contains_only_atoms_present_in_the_base(
    tmp_path: Path,
):
    sequences = tmp_path / "train_sequences.csv"
    cif_dir = tmp_path / "PDB_RNA"
    cif_dir.mkdir()
    sequences.write_text(
        "target_id,sequence\n1ABC_A,U\n2ABC_A,U\n",
        encoding="utf-8",
    )

    def cif_text(atom_names):
        rows = [
            f"{atom_name!r} U A 1 {index + 1}.0 0.0 0.0"
            for index, atom_name in enumerate(atom_names)
        ]
        return (
            "loop_\n"
            "_atom_site.label_atom_id\n"
            "_atom_site.label_comp_id\n"
            "_atom_site.label_asym_id\n"
            "_atom_site.label_seq_id\n"
            "_atom_site.Cartn_x\n"
            "_atom_site.Cartn_y\n"
            "_atom_site.Cartn_z\n"
            + "\n".join(rows)
            + "\n#\n"
        )

    expected_atoms = atom_names_for_base("U")
    (cif_dir / "1abc.cif").write_text(
        cif_text(expected_atoms),
        encoding="utf-8",
    )
    (cif_dir / "2abc.cif").write_text(
        cif_text(expected_atoms[:-1]),
        encoding="utf-8",
    )

    dataset = StanfordRnaAllAtomDataset.from_csv_and_cif(
        sequences,
        cif_dir,
        min_atom_coverage=0.99,
        center_coordinates=False,
    )

    assert [record.target_id for record in dataset.records] == ["1ABC_A"]
    assert dataset.stats["low_atom_coverage"] == 1


def test_all_atom_dataset_cache_round_trip_skips_cif_reparse(tmp_path: Path):
    sequences = tmp_path / "train_sequences.csv"
    cif_dir = tmp_path / "PDB_RNA"
    cache = tmp_path / "cache" / "records.pt"
    cif_dir.mkdir()
    sequences.write_text("target_id,sequence\n1ABC_A,A\n", encoding="utf-8")
    (cif_dir / "1abc.cif").write_text(
        "loop_\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "P A A 1 1.0 2.0 3.0\n"
        "\"C4'\" A A 1 4.0 5.0 6.0\n"
        "#\n",
        encoding="utf-8",
    )
    first = StanfordRnaAllAtomDataset.from_csv_and_cif(
        sequences, cif_dir, min_atom_coverage=0.01, cache_path=cache
    )

    with patch("rna_scaffold_3d.data.parse_cif_rna_chain") as parser:
        second = StanfordRnaAllAtomDataset.from_csv_and_cif(
            sequences, cif_dir, min_atom_coverage=0.01, cache_path=cache
        )

    assert cache.exists()
    assert len(first) == len(second) == 1
    assert torch.equal(first.records[0].coords, second.records[0].coords)
    parser.assert_not_called()


def test_all_atom_cache_invalidates_when_existing_cif_changes(
    tmp_path: Path,
):
    sequences = tmp_path / "train_sequences.csv"
    cif_dir = tmp_path / "PDB_RNA"
    cache = tmp_path / "records.pt"
    cif_dir.mkdir()
    sequences.write_text(
        "target_id,sequence\n1ABC_A,A\n", encoding="utf-8"
    )
    cif = cif_dir / "1abc.cif"

    def write_cif(x: str) -> None:
        cif.write_text(
            "loop_\n"
            "_atom_site.label_atom_id\n"
            "_atom_site.label_comp_id\n"
            "_atom_site.label_asym_id\n"
            "_atom_site.label_seq_id\n"
            "_atom_site.Cartn_x\n"
            "_atom_site.Cartn_y\n"
            "_atom_site.Cartn_z\n"
            f"P A A 1 {x} 2.0 3.0\n"
            "#\n",
            encoding="utf-8",
        )

    write_cif("1.0")
    first = StanfordRnaAllAtomDataset.from_csv_and_cif(
        sequences,
        cif_dir,
        min_atom_coverage=0.01,
        center_coordinates=False,
        cache_path=cache,
    )
    directory_stat = cif_dir.stat()
    old_file_mtime = cif.stat().st_mtime_ns
    write_cif("9.0")
    os.utime(
        cif,
        ns=(cif.stat().st_atime_ns, old_file_mtime + 1_000_000_000),
    )
    os.utime(
        cif_dir,
        ns=(directory_stat.st_atime_ns, directory_stat.st_mtime_ns),
    )

    second = StanfordRnaAllAtomDataset.from_csv_and_cif(
        sequences,
        cif_dir,
        min_atom_coverage=0.01,
        center_coordinates=False,
        cache_path=cache,
    )

    p_index = RNA_ATOM_NAMES.index("P")
    assert first.records[0].coords[0, p_index, 0].item() == 1.0
    assert second.records[0].coords[0, p_index, 0].item() == 9.0
    assert second.stats["loaded_from_cache"] is False


def test_all_atom_cache_metadata_treats_invalid_cif_name_as_missing(
    tmp_path: Path,
):
    sequences = tmp_path / "train_sequences.csv"
    cif_dir = tmp_path / "PDB_RNA"
    cif_dir.mkdir()
    sequences.write_text(
        "target_id,sequence\n>1ABC_A,A\n", encoding="utf-8"
    )

    dataset = StanfordRnaAllAtomDataset.from_csv_and_cif(
        sequences,
        cif_dir,
        min_atom_coverage=0.01,
        cache_path=tmp_path / "records.pt",
    )

    assert len(dataset) == 0
    assert dataset.stats["missing_cif"] == 1


def test_align_parsed_structure_retains_coordinates_across_missing_residue():
    coords = torch.zeros(2, len(RNA_ATOM_NAMES), 3)
    coords[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    coords[1, 0] = torch.tensor([3.0, 0.0, 0.0])
    mask = torch.zeros(2, len(RNA_ATOM_NAMES), dtype=torch.bool)
    mask[:, 0] = True

    sequence, aligned_coords, aligned_mask = _align_parsed_structure(
        "ACG", "AG", coords, mask, min_identity=0.6, min_coverage=0.6
    )

    assert sequence == "ACG"
    assert aligned_mask[:, 0].tolist() == [True, False, True]
    assert aligned_coords[2, 0, 0].item() == 3.0


def test_align_parsed_structure_rejects_weak_accidental_subsequence_match():
    coords = torch.zeros(2, len(RNA_ATOM_NAMES), 3)
    mask = torch.ones(2, len(RNA_ATOM_NAMES), dtype=torch.bool)

    aligned = _align_parsed_structure("AAAAACCCCC", "AG", coords, mask)

    assert aligned is None


def test_global_alignment_recovers_repetitive_sequence_matches():
    coords = torch.zeros(3, len(RNA_ATOM_NAMES), 3)
    coords[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.zeros(3, len(RNA_ATOM_NAMES), dtype=torch.bool)
    mask[:, 0] = True

    aligned = _align_parsed_structure(
        "ACAA",
        "AAA",
        coords,
        mask,
        min_identity=0.7,
        min_coverage=0.7,
    )

    assert aligned is not None
    _, aligned_coords, aligned_mask = aligned
    assert aligned_mask[:, 0].tolist() == [True, False, True, True]
    assert aligned_coords[:, 0, 0].tolist() == [1.0, 0.0, 2.0, 3.0]


def test_global_alignment_uses_phosphodiester_break_to_place_gap():
    coords = torch.zeros(3, len(RNA_ATOM_NAMES), 3)
    mask = torch.zeros(3, len(RNA_ATOM_NAMES), dtype=torch.bool)
    marker = RNA_ATOM_NAMES.index("OP1")
    o3 = RNA_ATOM_NAMES.index("O3'")
    p = RNA_ATOM_NAMES.index("P")
    coords[:, marker, 0] = torch.tensor([1.0, 2.0, 3.0])
    mask[:, marker] = True
    coords[0, o3, 0] = 0.0
    coords[1, p, 0] = 20.0
    coords[1, o3, 0] = 21.0
    coords[2, p, 0] = 22.6
    mask[0, o3] = True
    mask[1, p] = True
    mask[1, o3] = True
    mask[2, p] = True

    aligned = _align_parsed_structure(
        "AAAA",
        "AAA",
        coords,
        mask,
        min_identity=0.7,
        min_coverage=0.7,
    )

    assert aligned is not None
    _, aligned_coords, aligned_mask = aligned
    assert aligned_mask[:, marker].tolist() == [True, False, True, True]
    assert aligned_coords[:, marker, 0].tolist() == [1.0, 0.0, 2.0, 3.0]


def test_alignment_mismatch_retains_only_common_backbone_atoms():
    coords = torch.randn(3, len(RNA_ATOM_NAMES), 3)
    mask = torch.ones(3, len(RNA_ATOM_NAMES), dtype=torch.bool)

    aligned = _align_parsed_structure(
        "ACG",
        "AUG",
        coords,
        mask,
        min_identity=0.6,
        min_coverage=0.6,
    )

    assert aligned is not None
    _, aligned_coords, aligned_mask = aligned
    assert aligned_mask[1, RNA_ATOM_NAMES.index("P")]
    assert not aligned_mask[1, RNA_ATOM_NAMES.index("N1")]
    assert torch.equal(
        aligned_coords[1, RNA_ATOM_NAMES.index("P")],
        coords[1, RNA_ATOM_NAMES.index("P")],
    )
