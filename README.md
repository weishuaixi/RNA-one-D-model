# RNA Motif-Conditioned Scaffold Generator

本项目的唯一核心模型是一维 RNA 支架序列生成器：输入一段必须保留的功能
motif，自动生成其左右两侧支架，输出完整 RNA 序列。

```text
输入 motif（最短 4 nt，上限由当前训练窗口动态决定）
  │
  ├─ RNA-FM（外部、冻结）：只提供 640 维预训练特征
  │
  └─ OUR MODEL / 本项目核心模型（可训练）
       ├─ 640→768 特征投影
       ├─ motif 条件双向 Transformer 支架生成器
       ├─ 左右支架联合碱基预测
       ├─ 完整长度预测
       └─ motif 位置预测
              │
              └─ motif 原样锁定的完整 RNA（最长 512 nt）
```

项目不再训练或发布自研 RNA 三维结构模型。ViennaRNA RNAfold 和官方
RhoFold+ 只作为可选的外部验证工具，分别评价二级结构和三维结构；它们不属于
本项目的训练模型。

## 当前代码

```text
rna_scaffold/data.py              # RNA 序列读取与训练样本
rna_scaffold/datamodule.py        # Lightning 数据模块
rna_scaffold/lightning_module.py  # 一维 Transformer 训练模块
rna_scaffold/model.py             # 联合去噪生成器及长度/位置预测头
rna_scaffold/pretrained.py        # 可选 RNA-FM 特征编码器
rna_scaffold/generate.py          # checkpoint推理、迭代去噪和显式Markov baseline
rna_scaffold/validators/rnafold.py # 可选RNAfold外部验证适配器
rna_scaffold/evaluation.py        # benchmark指标与bootstrap统计
rna_scaffold/tokenizer.py         # RNA 与控制 token
train.py                          # 一维模型训练入口
configs/train_stanford_1d.yaml    # 一维训练配置
configs/train_scaffold_a800.yaml  # A800 准确性优先配置（启用 RNA-FM）
```

## 训练现有一维模型

```bash
python train.py --config configs/train_stanford_1d.yaml
```

服务器启用 RNA-FM：

```bash
python -m pip install -e '.[rnafm]'
python train.py --config configs/train_scaffold_a800.yaml
```

A800 配置冻结官方 RNA-FM，将其 12 层编码器产生的 640 维逐碱基表征投影到
生成器的 768 维隐空间。RNA-FM 仅提供预训练先验；长度、motif 位置和左右支架
仍由本项目的生成模型学习。若服务器已手动下载权重，可在配置中的
`model.pretrained.checkpoint` 填入权重路径，以便记录权重 SHA-256。

正式生成必须提供训练后的checkpoint；程序不会在checkpoint缺失时退回随机或
Markov结果：

```bash
python generate_scaffold.py \
  --motif GCGG \
  --checkpoint checkpoints_scaffold_a800_mmseqs80/best.ckpt \
  --num-candidates 256 \
  --max-length 512 \
  --seed 42 \
  --output outputs/GCGG_candidates.jsonl
```

可选RNAfold验证与多指标重排：

```bash
python validate_scaffolds.py \
  --input outputs/GCGG_candidates.jsonl \
  --output outputs/GCGG_validated.jsonl
```

统一预算的baseline与消融：

```bash
python benchmark_scaffolds.py \
  --config configs/benchmark_scaffolds.yaml
```

`generate_markov_baseline()` 仅用于明确标注的对照实验，不属于正式模型推理。

完整重构目标、数据防泄漏方案、生成架构、外部验证边界和 Benchmark 定义见：

- `docs/superpowers/specs/2026-08-15-scaffold-generator-redesign.md`
- `docs/superpowers/plans/2026-08-15-scaffold-generator-redesign.md`
- `docs/superpowers/specs/2026-08-19-scaffold-generation-completion-design.md`
- `docs/superpowers/plans/2026-08-19-scaffold-generation-completion.md`

## 设计原则

- 原始训练 RNA 不设长度上限；超过 512 nt 时，每个 epoch 动态裁剪一个 512 nt 窗口；
- motif 最短 4 nt，95% 的采样权重位于 4–127 nt，上限由当前窗口动态决定；
- 总支架硬下限为 8 nt、左右各至少 2 nt；序列足够长时优先保留 24 nt 总支架；
- 单次模型输入和输出的硬上限仍为 512 nt；
- 左右支架必须联合建模，不能以随机或 Markov 结果冒充模型输出；
- 数据按 RNA 家族拆分，并审计精确重复与近重复泄漏；
- RNA-FM 保持可选，并通过独立测试集消融报告其实际增益；
- RNAfold 与 RhoFold+ 缺失时明确报告 unavailable，不伪造结构分数；
- Benchmark 使用相同 motif、候选数、随机种子与结构计算预算。
