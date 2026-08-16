# RNA Motif-Conditioned Scaffold Generator

本项目的唯一核心模型是一维 RNA 支架序列生成器：输入一段必须保留的功能
motif，自动生成其左右两侧支架，输出完整 RNA 序列。

```text
motif (4–64 nt)
  -> 自动预测完整长度和 motif 位置
  -> 联合生成左、右支架
  -> motif 原样锁定
  -> 完整 RNA（最长 512 nt）
```

项目不再训练或发布自研 RNA 三维结构模型。ViennaRNA RNAfold 和官方
RhoFold+ 只作为可选的外部验证工具，分别评价二级结构和三维结构；它们不属于
本项目的训练模型。

## 当前代码

```text
rna_scaffold/data.py              # RNA 序列读取与训练样本
rna_scaffold/datamodule.py        # Lightning 数据模块
rna_scaffold/lightning_module.py  # 一维 Transformer 训练模块
rna_scaffold/generate.py          # 支架生成接口（正在升级为 checkpoint 推理）
rna_scaffold/tokenizer.py         # RNA 与控制 token
train.py                          # 一维模型训练入口
configs/train_stanford_1d.yaml    # 一维训练配置
```

## 训练现有一维模型

```bash
python train.py --config configs/train_stanford_1d.yaml
```

完整重构目标、数据防泄漏方案、生成架构、外部验证边界和 Benchmark 定义见：

- `docs/superpowers/specs/2026-08-15-scaffold-generator-redesign.md`
- `docs/superpowers/plans/2026-08-15-scaffold-generator-redesign.md`

## 设计原则

- motif 长度支持 4–64 nt，并在所有生成步骤中保持不变；
- 总长度由模型自动决定，硬上限为 512 nt；
- 左右支架必须联合建模，不能以随机或 Markov 结果冒充模型输出；
- 数据按 RNA 家族拆分，并审计精确重复与近重复泄漏；
- RNA-FM 是否采用由独立测试集消融结果决定；
- RNAfold 与 RhoFold+ 缺失时明确报告 unavailable，不伪造结构分数；
- Benchmark 使用相同 motif、候选数、随机种子与结构计算预算。
