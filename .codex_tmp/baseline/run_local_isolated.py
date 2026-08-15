import csv,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];base=ROOT/'.codex_tmp/baseline';out=base/'ours_all';out.mkdir(exist_ok=True)
ckpt=ROOT/'checkpoints_3d/checkpoints_3d_a800_full/rna_3d_best.pt'
targets=[]
with (base/'monomer_test.csv').open(encoding='utf-8-sig',newline='') as f:
 for r in csv.DictReader(f):
  seq=r['Sequence (unmod.)'].strip().upper();target=f"{r['PDB ID'].lower()}_{r['Asym. Chain ID']}"
  if set(seq)<=set('AUGC') and len(seq)<=1536 and not (out/f'{target}.pdb').exists():targets.append((len(seq),target,seq))
targets.sort();log=out/'status_isolated.jsonl'
with log.open('a',encoding='utf-8') as f:
 for length,target,seq in targets:
  cmd=[sys.executable,str(ROOT/'fold_3d.py'),'--sequence',seq,'--checkpoint',str(ckpt),'--device','cuda:0','--output',str(out/f'{target}.pt'),'--output-pdb',str(out/f'{target}.pdb')]
  t=time.perf_counter()
  try:
   x=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=600)
   status='ok' if x.returncode==0 else ('oom' if 'out of memory' in (x.stderr+x.stdout).lower() else 'error')
   row={'target_id':target,'length':length,'status':status,'seconds':time.perf_counter()-t,'returncode':x.returncode,'error':(x.stderr+x.stdout)[-800:] if x.returncode else ''}
  except subprocess.TimeoutExpired:
   row={'target_id':target,'length':length,'status':'timeout','seconds':time.perf_counter()-t}
  f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();print(json.dumps(row,ensure_ascii=False),flush=True)
