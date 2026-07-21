# RNA Motif Scaffold + Local RhoFold Training

核心流水线：

```text
fixed motif
  -> jointly trained sequence embedding / scaffold head
  -> complete RNA sequence
  -> the same local RhoFold-style model
  -> 3D coordinates / PDB
```

不接外部 RNAfold、RhoFold 或 RhoFold+ 命令。三维折叠只使用本项目训练出来的 checkpoint。

## 主要文件

```text
rna_scaffold/                      # motif scaffold 输入与生成接口

rna_scaffold_3d/rhofold.py         # 内置 RhoFold-style 3D 模型
rna_scaffold_3d/data.py            # Stanford RNA 3D / CIF 全原子数据读取
rna_scaffold_3d/losses.py          # 3D 训练损失
rna_scaffold_3d/pdb_writer.py      # 本地坐标写 PDB
train_3d.py                        # 3D RhoFold-style 训练入口
fold_3d.py                         # motif/sequence -> 本地 checkpoint 折叠 -> PDB
configs/train_3d_a800_card1.yaml   # 3D 训练配置
configs/train_3d_a800_full.yaml    # 服务器全数据高通量配置，长度上限 2048
configs/train_3d_local_windows.yaml # Windows 本地 smoke 配置
```

## 一维 Motif Scaffold

给定 motif，生成完整 RNA 序列：

```python
from rna_scaffold.generate import generate_rna_sequence

sequence = generate_rna_sequence(
    motif="GCGG",
    num_candidates=128,
    rng_seed=42,
)
print(sequence)
```

它会自动采样总长度和 motif 位置，内部构造 masked scaffold prompt，然后返回：

```text
left_sequence + motif + right_sequence
```

一维 scaffold 不再使用 GC、互补率或复杂度等手工惩罚进行训练。`train_3d.py`
会内部遮盖序列，仅保留部分 motif，将序列重建项与全部三维结构项合并为一个联合目标；
MASK 位置的 A/U/C/G 概率会转换成软碱基 embedding 后送入三维模块，因此三维损失可以
直接更新 scaffold head。完整输入序列使用真实碱基 embedding，不会被模型改写。
序列和结构两个任务使用可学习的不确定性权重；motif 推理默认进行 6 步迭代去噪，
逐步固定高置信度碱基，并将低置信度位置留到后续步骤重新预测。

## 训练 3D RhoFold-style 模型

服务器训练集默认路径：

```text
/home/weisx/workdir/igem one-model/stanford-rna-3d-folding-data
```

服务器直接运行：

```bash
python train_3d.py --config configs/train_3d_a800_card1.yaml
```

服务器全数据高通量训练：

```bash
python train_3d.py --config configs/train_3d_a800_full.yaml
```

`train_sequences.v2.csv` 最长序列超过 4000 nt。当前模型含 residue pair 表征，显存近似按序列长度平方增长，所以全通量配置默认使用全部记录但过滤到 `max_sequence_length: 2048`，覆盖大部分训练样本并保持 A800 上更稳。更长序列建议后续使用 crop/chunk 训练策略。

默认输出：

```text
checkpoints_3d_a800_card1/rna_3d_best.pt
```

Windows 本地只做 smoke test 时运行：

```bash
python train_3d.py --config configs/train_3d_local_windows.yaml
```

3D 模型包含：

- sequence embedding
- optional MSA embedding fallback
- pair representation
- E2Eformer-style sequence/pair blocks
- recycling
- structure head
- distogram head
- jointly trained sequence reconstruction head
- pLDDT-like confidence head
- 27 canonical RNA heavy atoms per residue

## Motif 到 PDB

用本地训练 checkpoint 从 motif 直接生成一维序列并折叠成 PDB：

```bash
python fold_3d.py \
  --motif GCGG \
  --checkpoint checkpoints_3d_a800_card1/rna_3d_best.pt \
  --output outputs/fold_3d.pt \
  --output-pdb outputs/fold_3d.pdb \
  --seed 42
```

如果已经有完整 RNA 序列，也可以直接折叠：

```bash
python fold_3d.py \
  --sequence AUGCGGCUA \
  --checkpoint checkpoints_3d_a800_card1/rna_3d_best.pt \
  --output outputs/fold_3d.pt \
  --output-pdb outputs/fold_3d.pdb
```

## 测试

```bash
python -m pytest -q
```
