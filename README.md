# RNA-one-D-model
一维RNAmodel

## 目标

输入固定 RNA 功能 motif，训练一个 motif-protected scaffold 模型，让系统学习在 motif 周围重新生成 RNA 支架，最终返回一条完整 RNA：

```text
S* = scaffold_left + fixed_motif + scaffold_right
```

当前默认训练任务参考 EvoFlow-RNA 论文里的 aptamer scaffold 设计思路：

```text
XXXGCGGXXXXXXX  ->  GUCGCGG...GCUAC
```

也就是固定功能识别 motif，把其余 scaffold 位置替换成 `<MASK>`，训练模型根据固定 motif 的上下文补全完整 RNA。论文中作者对 HIV-1 Rev peptide aptamer（PDB: 1ULL）和 AMP aptamer（PDB: 1RAW）固定关键识别碱基、重建周围序列，再用三维结构预测评估 motif 局部 RMSD。本项目先实现对应的一维 mask-inpainting 训练形式。

使用时不需要手写左右 `<MASK>`，也不需要指定左右长度。你只给 motif，程序会在内部自动采样多个总长度和 motif 位置，构造多条 masked scaffold prompt，让模型分别补全后再筛选最佳序列。

仍然保留早期的左右 flank 生成任务：可以在配置中把 `data.task` 改回 `flank_scaffold`。

## 目录

```text
rna_scaffold/
  tokenizer.py          # RNA + MASK/控制符 tokenizer
  utils.py              # RNA 校验、反向互补、互补率
  data.py               # FASTA/CSV/PDB 读取与 scaffold/masked-inpainting 样本构造
  datamodule.py         # LightningDataModule
  lightning_module.py   # Encoder-Decoder Transformer LightningModule
  generate.py           # 单条最佳结果 JSON 构造/贪心解码工具
train.py                # Lightning + W&B 训练入口
generate_3d.py          # motif -> RNA 序列 -> FASTA/可选 3D 预测入口
train_3d.py             # Stanford RNA 3D 数据训练坐标预测模型
predict_3d.py           # 用本地 3D checkpoint 输出伪 C4' PDB
configs/train_a800.yaml # 2 卡 A800 推荐配置
configs/train_3d_a800_card1.yaml # 1 张 A800 且使用 1 号卡的 3D 训练配置
rna_scaffold_3d/        # RNA sequence -> 3D coordinate baseline
tests/                  # pytest 测试
```

## 安装

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
wandb login
```

如果你的环境已经装好了 PyTorch，可以跳过第一行。

## 数据格式：PDB

你的训练数据是 PDB 文件时，把 `.pdb` 文件放到：

```text
data/pdbs/
```

训练配置里默认已经设置：

```yaml
data:
  train_data: data/pdbs
```

PDB 解析逻辑：

1. 优先读取 `SEQRES` 记录里的 RNA 链；
2. 如果没有 `SEQRES`，再从 `ATOM/HETATM` 的残基顺序中提取 RNA；
3. 默认只保留 RNA 碱基 `A/U/C/G`，忽略 DNA 链和未知修饰残基；
4. 提取到 RNA 序列后，默认从序列中间截取固定 motif，将其余位置替换成 `<MASK>`，完整原序列作为训练目标。

仍然保留 FASTA 兼容：如果 `train_data` 指向 `.fa/.fasta/.fna` 文件，也可以读取。

## 训练

```bash
python train.py --config configs/train_a800.yaml
```

如果使用本机的 Stanford RNA 3D Folding 数据集做一维 scaffold 预训练：

```bash
python train.py --config configs/train_stanford_1d.yaml
```

这个配置读取：

```text
F:/github_item/stanford-rna-3d-folding-data/train_sequences.csv
```

默认 `masked_scaffold` 任务会从完整 RNA 序列中自动构造：

```text
input:  <MASK><MASK>...fixed_motif...<MASK><MASK>
target: original_full_sequence
```

其中 `fixed_motif` 默认来自序列中部，代表需要保留的功能核心。这个设置最接近“固定 aptamer 结合位点、重新设计周围 RNA scaffold”的任务。

推理时的外部接口只需要输入 motif，直接返回一条完整 RNA 序列：

```python
from rna_scaffold.generate import generate_rna_sequence

sequence = generate_rna_sequence(
    motif="GCGG",
    num_candidates=128,
    rng_seed=42,
)
print(sequence)
```

`<MASK>`、候选总长度、motif 在序列中的位置都是内部细节。当前函数先提供可运行的一维 baseline；有训练好的 masked scaffold checkpoint 后，可以沿用同一个接口，把内部候选交给模型补全，再用结构/能量评分筛选。

## 生成 RNA 与 3D 准备

如果只想从 motif 得到 RNA 序列和后续 3D 预测输入：

```bash
python generate_3d.py --motif GCGG --output-dir outputs/rna_3d --name candidate_1 --seed 42
```

它会打印完整 RNA 序列，并写出：

```text
outputs/rna_3d/candidate_1.fa
```

如果本机已经安装并配置了 RhoFold/RhoFold+，可以把命令模板传进去。模板里可用：

```text
{fasta}       生成的 FASTA
{output_pdb}  期望输出的 PDB
{output_dir}  输出目录
```

示例：

```bash
python generate_3d.py --motif GCGG --name candidate_1 --predictor-command python path/to/rhofold_infer.py --input {fasta} --output {output_pdb}
```

不同 RhoFold/RhoFold+ 安装方式的参数名可能不一样，按你本机实际命令替换 `--predictor-command` 后面的模板即可。

## 训练 RNA 3D 模型

Stanford RNA 3D Folding 数据集路径已经接入服务器配置：

```text
/home/weisx/workdir/igem one-model/stanford-rna-3d-folding-data
```

服务器上一张 A800、使用物理 1 号卡训练：

```bash
bash scripts/setup_server_env.sh
bash scripts/train_3d_a800_card1.sh
```

如果想同时在屏幕看进度并保存日志：

```bash
mkdir -p logs
bash scripts/train_3d_a800_card1.sh 2>&1 | tee logs/train_3d_a800_card1.log
```

训练时会显示每个 epoch 的 `train`/`val` batch 进度条、当前 batch loss 和平均 loss。另开一个终端可以实时看 1 号卡：

```bash
watch -n 1 nvidia-smi -i 1
```

这个第一版是轻量 RhoFold-like baseline：

```text
RNA sequence -> 每个 residue 一个伪 C4' 三维坐标 -> PDB
```

训练完成后会保存：

```text
checkpoints_3d/rna_3d_best.pt
```

然后可以预测一条 RNA 的伪 C4' PDB：

```bash
python predict_3d.py --checkpoint checkpoints_3d/rna_3d_best.pt --sequence GCGG --output-pdb outputs/rna_3d/gcgg.pdb
```

对应配置是：

```text
configs/train_3d_a800_card1.yaml
```

里面已经设置：

```yaml
trainer:
  accelerator: gpu
  cuda_device: 1
```

如果服务器路径和本地不同，只需要改这个配置里的：

```yaml
data:
  sequences_csv: ...
  labels_csv: ...
```

当前服务器配置默认使用 v2 训练集，并做了两层过滤，适合第一阶段稳定训练：

```yaml
data:
  sequences_csv: "/home/weisx/workdir/igem one-model/stanford-rna-3d-folding-data/train_sequences.v2.csv"
  labels_csv: "/home/weisx/workdir/igem one-model/stanford-rna-3d-folding-data/train_labels.v2.csv"
  max_sequence_length: 1024
  min_coord_coverage: 0.8
```

在你当前这份 Stanford 数据上，过滤后约有 `3182` 条 RNA 可用于训练，最长 `1024 nt`，最低坐标覆盖率 `80%`。

如果要使用旧的左右 flank 预测任务：

```yaml
data:
  task: flank_scaffold
```

此时训练样本会是：

```text
input:  motif
target: <LEFT>left_sequence<END_LEFT><RIGHT>right_sequence<END_RIGHT>
```

默认配置：

- Lightning Trainer
- W&B Logger
- 2 GPU DDP
- bf16 mixed precision
- Transformer encoder-decoder

## 最终输出格式

```json
{
  "left_sequence": "...",
  "motif": "AUGCGUACGA",
  "right_sequence": "...",
  "left_length": 37,
  "right_length": 37,
  "full_sequence": "...AUGCGUACGA...",
  "quality_score": 0.91,
  "motif_preserved": true,
  "left_right_complementarity": 0.87
}
```

## 一维随机 scaffold baseline

如果还没有训练好的 checkpoint，可以先用规则采样生成一条 motif-protected scaffold：

```python
from rna_scaffold.generate import build_random_natural_scaffold_result, result_to_json

result = build_random_natural_scaffold_result(
    motif="AUGCGUACGA",
    min_left_length=30,
    max_left_length=120,
    num_candidates=128,
    rng_seed=42,
)
print(result_to_json(result))
```

这个 baseline 会随机采样多条左侧序列，按“多数互补、GC 合理、避免长同聚物”的规则打分，并返回当前分数最高的一条。

## 测试

```bash
python -m pytest -q
```
