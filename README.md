# RNA Motif-Conditioned Scaffold Generator

本项目的唯一核心模型是一维 RNA 支架序列生成器：输入一段必须保留的功能
motif，自动生成其左右两侧支架，输出完整 RNA 序列。

```text
输入 motif（训练覆盖 8–256 nt，主体分布 8–127 nt）
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
rna_scaffold/generate.py          # 支架生成接口（正在升级为 checkpoint 推理）
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

完整重构目标、数据防泄漏方案、生成架构、外部验证边界和 Benchmark 定义见：

- `docs/superpowers/specs/2026-08-15-scaffold-generator-redesign.md`
- `docs/superpowers/plans/2026-08-15-scaffold-generator-redesign.md`

## 设计原则

- motif 训练长度覆盖 8–256 nt，其中 95% 样本位于 8–127 nt；
- 完整序列必须留下至少 24 nt 总支架，且 motif 左右各至少保留 4 nt；
- 总长度由模型自动决定，硬上限为 512 nt；
- 左右支架必须联合建模，不能以随机或 Markov 结果冒充模型输出；
- 数据按 RNA 家族拆分，并审计精确重复与近重复泄漏；
- RNA-FM 保持可选，并通过独立测试集消融报告其实际增益；
- RNAfold 与 RhoFold+ 缺失时明确报告 unavailable，不伪造结构分数；
- Benchmark 使用相同 motif、候选数、随机种子与结构计算预算。
