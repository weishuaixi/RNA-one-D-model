from pathlib import Path

import torch

from rna_scaffold_3d.data import (
    StanfordRna3DDataset,
    collate_3d_batch,
    load_stanford_rna_3d_records,
)


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
