import argparse, csv, gc, json, sys, time, traceback
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fold_3d import _fold_with_model,_load_model
from rna_scaffold_3d.pdb_writer import write_pdb

ap=argparse.ArgumentParser()
ap.add_argument('--max-length',type=int,default=384)
args=ap.parse_args()
base=ROOT/'.codex_tmp/baseline'
out=base/'ours_all';out.mkdir(parents=True,exist_ok=True)
checkpoint=ROOT/'checkpoints_3d/checkpoints_3d_a800_full/rna_3d_best.pt'

targets=[]
with (base/'monomer_test.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        seq=r['Sequence (unmod.)'].strip().upper()
        target=f"{r['PDB ID'].lower()}_{r['Asym. Chain ID']}"
        if set(seq)<=set('AUGC') and len(seq)<=args.max_length:
            targets.append((len(seq),target,seq))
targets.sort()

device='cuda:0' if torch.cuda.is_available() else 'cpu'
t0=time.perf_counter();model,_=_load_model(checkpoint,device);load_seconds=time.perf_counter()-t0
status_path=out/f'status_le_{args.max_length}.jsonl'
done=set()
if status_path.exists():
    for line in status_path.read_text(encoding='utf-8').splitlines():
        try:
            row=json.loads(line)
            if row.get('status')=='ok':done.add(row['target_id'])
        except Exception:pass

with status_path.open('a',encoding='utf-8') as log:
    for length,target,sequence in targets:
        if target in done or (out/f'{target}.pdb').exists():continue
        start=time.perf_counter()
        try:
            result=_fold_with_model(sequence,model,device)
            write_pdb(sequence,result.coords,out/f'{target}.pdb')
            torch.save({'sequence':sequence,'coords':result.coords,'plddt':result.plddt},out/f'{target}.pt')
            row={'target_id':target,'length':length,'status':'ok','seconds':time.perf_counter()-start,'mean_plddt':float(result.plddt.mean()),'min_plddt':float(result.plddt.min())}
        except RuntimeError as e:
            row={'target_id':target,'length':length,'status':'oom' if 'out of memory' in str(e).lower() else 'runtime_error','seconds':time.perf_counter()-start,'error':str(e)[:500]}
        except Exception as e:
            row={'target_id':target,'length':length,'status':'error','seconds':time.perf_counter()-start,'error':repr(e)}
        log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush();print(json.dumps(row,ensure_ascii=False),flush=True)
        gc.collect()
        if torch.cuda.is_available():
            try:torch.cuda.empty_cache()
            except RuntimeError:pass
print(json.dumps({'finished':True,'eligible':len(targets),'load_seconds':load_seconds,'max_length':args.max_length}),flush=True)
