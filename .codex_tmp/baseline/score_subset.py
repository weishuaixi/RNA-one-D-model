import csv, json, math, sys
from pathlib import Path
from collections import OrderedDict
import torch
from Bio.PDB import PDBParser, MMCIFParser
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from evaluate_3d import prediction_physical_metrics
from rna_scaffold_3d.rna_atoms import RNA_ATOM_NAMES, RNA_ATOM_TO_INDEX
from rna_scaffold_3d.sequence import encode_rna_sequence

BASE=ROOT/'.codex_tmp/baseline'
TARGETS=['8i43_A','8apo_ED','8utg_A','7sxp_A','9aus_C','8q4o_A','7ps8_A','8tns_A']
METHODS={'AF3':'af3.pdb','NuFold':'nu.pdb','RoseTTAFoldNA':'rf2na.pdb','RhoFold+':'rho.pdb','trRosettaRNA':'trRNA.pdb','Ours':None}

seqs={}
with (BASE/'monomer_test.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f): seqs[f"{r['PDB ID'].lower()}_{r['Asym. Chain ID']}"]=r['Sequence (unmod.)'].strip().upper()

def parse_c1(path):
    structure=load_structure(path); points=[]
    for residue in structure.get_residues():
        for name in ("C1'",'C1*'):
            if name in residue:
                points.append(torch.tensor(residue[name].coord.copy()));break
    return torch.stack(points)

def parse_all(path,L):
    items=[]
    for residue in load_structure(path).get_residues():
        atoms={a.name.replace('*',"'"):a.coord.copy() for a in residue if a.name.replace('*',"'") in RNA_ATOM_TO_INDEX}
        if atoms:items.append(atoms)
    items=items[:L]
    if len(items)!=L:return None
    coords=torch.zeros(L,len(RNA_ATOM_NAMES),3)
    for i,atoms in enumerate(items):
        for atom,xyz in atoms.items():coords[i,RNA_ATOM_TO_INDEX[atom]]=torch.tensor(xyz)
    return coords

def load_structure(path):
    with open(path,encoding='utf-8',errors='ignore') as f:first=f.readline()
    parser=MMCIFParser(QUIET=True) if first.startswith(('#','data_')) else PDBParser(QUIET=True)
    return parser.get_structure(path.stem,str(path))

def kabsch(pred,target):
    p=pred.float()-pred.float().mean(0);t=target.float()-target.float().mean(0)
    u,s,vh=torch.linalg.svd(p.T@t)
    d=torch.det(vh.T@u.T)
    corr=torch.eye(3);corr[-1,-1]=d
    r=vh.T@corr@u.T
    aligned=p@r.T
    return aligned,t

def metrics(pred,target):
    n=min(len(pred),len(target));pred=pred[:n];target=target[:n]
    pa,ta=kabsch(pred,target)
    rmsd=torch.sqrt(((pa-ta)**2).sum(-1).mean()).item()
    dp=torch.cdist(pred.float(),pred.float());dt=torch.cdist(target.float(),target.float())
    mask=(dt<15)&(~torch.eye(n,dtype=torch.bool))
    err=(dp-dt).abs();score=sum(((err<x)&mask).sum().item() for x in (.5,1,2,4))/(4*mask.sum().item())*100 if mask.sum() else float('nan')
    drms=torch.sqrt(((dp-dt)[mask]**2).mean()).item() if mask.sum() else float('nan')
    return {'c1_kabsch_rmsd':rmsd,'c1_lddt':score,'c1_distance_rmsd':drms,'aligned_residues':n}

rows=[]
for target in TARGETS:
    ref_path=BASE/'monomers'/target/'rcsb.pdb';ref=parse_c1(ref_path);seq=seqs[target]
    for method,file in METHODS.items():
        path=(BASE/'ours'/f'{target}.pdb') if method=='Ours' else (BASE/'monomers'/target/file)
        if not path.exists():continue
        pred=parse_c1(path);m=metrics(pred,ref)
        all_coords=parse_all(path,len(seq))
        if all_coords is not None:
            try:m.update(prediction_physical_metrics(all_coords,torch.tensor(encode_rna_sequence(seq))))
            except Exception:pass
        rows.append({'target_id':target,'method':method,'length':len(seq),'pred_c1_count':len(pred),'ref_c1_count':len(ref),**m})

methods={}
for method in METHODS:
    subset=[r for r in rows if r['method']==method]
    if not subset:continue
    numeric=['c1_lddt','c1_kabsch_rmsd','c1_distance_rmsd','covalent_bond_rmse','backbone_angle_rmse_deg','clash_penetration_rms','base_planarity_rms','sugar_closure_rmse','o3_p_bond_rmse']
    methods[method]={'n':len(subset)}
    for k in numeric:
        vals=[r[k] for r in subset if k in r and math.isfinite(r[k])]
        if vals:methods[method][k+'_mean']=sum(vals)/len(vals);methods[method][k+'_median']=sorted(vals)[len(vals)//2]

result={'targets':TARGETS,'per_target':rows,'summary':methods}
(BASE/'baseline_subset_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
with (BASE/'baseline_subset_per_target.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
print(json.dumps(methods,indent=2))
