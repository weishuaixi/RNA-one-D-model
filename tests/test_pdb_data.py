from pathlib import Path

from rna_scaffold.data import load_pdb_rna_sequences, load_sequences


PDB_WITH_SEQRES = """\
HEADER    RNA TEST
SEQRES   1 A   10    G   C   A   U   G   C   G   U   A   C
SEQRES   1 B    4    DA  DT  DG  DC
END
"""


PDB_WITH_ATOMS_ONLY = """\
ATOM      1  P     G A   1      11.104  13.207   7.178  1.00 20.00           P
ATOM      2  P     C A   2      12.104  14.207   8.178  1.00 20.00           P
ATOM      3  P     A A   3      13.104  15.207   9.178  1.00 20.00           P
ATOM      4  P     U A   4      14.104  16.207  10.178  1.00 20.00           P
ATOM      5  P     G A   4A     15.104  17.207  11.178  1.00 20.00           P
END
"""


def test_load_pdb_rna_sequences_prefers_seqres_and_ignores_dna_chain(tmp_path: Path):
    pdb_path = tmp_path / "rna.pdb"
    pdb_path.write_text(PDB_WITH_SEQRES, encoding="utf-8")

    assert load_pdb_rna_sequences(pdb_path) == ["GCAUGCGUAC"]


def test_load_pdb_rna_sequences_falls_back_to_atom_records(tmp_path: Path):
    pdb_path = tmp_path / "atoms_only.pdb"
    pdb_path.write_text(PDB_WITH_ATOMS_ONLY, encoding="utf-8")

    assert load_pdb_rna_sequences(pdb_path) == ["GCAUG"]


def test_load_sequences_accepts_pdb_directory(tmp_path: Path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    (pdb_dir / "one.pdb").write_text(PDB_WITH_SEQRES, encoding="utf-8")
    (pdb_dir / "two.pdb").write_text(PDB_WITH_ATOMS_ONLY, encoding="utf-8")

    assert load_sequences(pdb_dir) == ["GCAUGCGUAC", "GCAUG"]
