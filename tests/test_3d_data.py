from pathlib import Path

import torch

from rna_scaffold_3d.data import (
    StanfordRnaAllAtomDataset,
    StanfordRna3DDataset,
    collate_3d_batch,
    load_stanford_rna_3d_records,
)
from rna_scaffold_3d.rna_atoms import RNA_ATOM_NAMES


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
