# RNA-one-D-model
一维RNAmodel

## 目标

输入固定 RNA motif，训练一个 motif-conditioned scaffold 模型，让系统自动决定左右长度和序列，最终只返回一条最佳完整 RNA：

```text
S* = L* + M + R*
```

当前一维生成骨架强化一个约束：

```text
left_sequence 与 right_sequence 大部分反向互补，但允许少量自然缺陷
```

生产推理时推荐“模型或采样器生成一侧 stem，程序从反向互补模板构造另一侧，并加入少量 mismatch / wobble-like 扰动”，这样 motif 永远不被改写，同时左右两侧不会机械地 100% 完全互补。

## 目录

```text
rna_scaffold/
  tokenizer.py          # RNA + 控制符 tokenizer
  utils.py              # RNA 校验、反向互补、互补率
  data.py               # FASTA 读取与训练样本构造
  datamodule.py         # LightningDataModule
  lightning_module.py   # Encoder-Decoder Transformer LightningModule
  generate.py           # 单条最佳结果 JSON 构造/贪心解码工具
train.py                # Lightning + W&B 训练入口
configs/train_a800.yaml # 2 卡 A800 推荐配置
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
4. 提取到 RNA 序列后，从序列中间截取 motif，并保留 motif 左右两侧的真实序列片段作为训练目标。

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

训练样本会从完整 RNA 序列中自动切成：

```text
left_sequence + motif + right_sequence
```

其中 `motif` 来自序列中部，左右两侧保留训练数据里的真实片段；`stem_length` 表示最多截取多长 flank，`min_flank_length` 控制太短的左右片段是否丢弃。

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
