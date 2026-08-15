import csv, json, time, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fold_3d import _fold_with_model, _load_model
from rna_scaffold_3d.pdb_writer import write_pdb

CSV = ROOT / '.codex_tmp/baseline/monomer_test.csv'
OUT = ROOT / '.codex_tmp/baseline/ours'
CKPT = ROOT / 'checkpoints_3d/checkpoints_3d_a800_full/rna_3d_best.pt'
TARGETS = ['8i43_A','8apo_ED','8utg_A','7sxp_A','9aus_C','8q4o_A','7ps8_A','8tns_A']

rows = {}
with CSV.open(encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        key = f"{row['PDB ID'].lower()}_{row['Asym. Chain ID']}"
        rows[key] = row['Sequence (unmod.)'].strip().upper()

OUT.mkdir(parents=True, exist_ok=True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
t0 = time.perf_counter()
model, _ = _load_model(CKPT, device)
load_seconds = time.perf_counter() - t0
records=[]
for target in TARGETS:
    sequence = rows[target]
    start=time.perf_counter()
    result=_fold_with_model(sequence, model, device)
    seconds=time.perf_counter()-start
    write_pdb(sequence, result.coords, OUT / f'{target}.pdb')
    torch.save({'sequence':sequence,'coords':result.coords,'plddt':result.plddt},OUT/f'{target}.pt')
    records.append({'target_id':target,'sequence':sequence,'length':len(sequence),'seconds':seconds,'mean_plddt':float(result.plddt.mean()),'min_plddt':float(result.plddt.min())})
    print(json.dumps(records[-1]),flush=True)
(OUT/'run_summary.json').write_text(json.dumps({'device':device,'load_seconds':load_seconds,'records':records},indent=2),encoding='utf-8')
print(json.dumps({'load_seconds':load_seconds,'count':len(records)}),flush=True)
