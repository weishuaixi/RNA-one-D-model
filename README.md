# RNA Scaffold Generator V2

本仓库只发布一种项目自有模型：给定必须原样保留的 RNA motif，生成其左右支架的
一维序列模型。V2 使用左右 flank 长度头、motif 锁定的置信度迭代重掩码，以及可选的
RNA-FM frozen/LoRA 特征。RNAfold 只是可选的二级结构合理性代理，不进入训练图，也
不能证明生物学功能。

```text
motif-only placement input
        │
        ├── left/right flank-length logits ── legal joint pair selection
        │
        └── RNA-FM none/frozen/LoRA + project Transformer
                                      │
                         confidence-based iterative remasking
                                      │
                         left flank + immutable motif + right flank
```

V2 故意不兼容旧 checkpoint。加载器只接受整数
`architecture_version: 2`；缺失、布尔值、浮点值、字符串或其他版本都会要求重新训练，
不会回退到启发式、Markov 或旧模型。

## 唯一权威服务器流程

以下流程均从仓库根目录执行，唯一正式训练配置是
`configs/train_scaffold_a800.yaml`。先把两个规范输入放在配置指定位置：

- `data/processed/rna_merged_mmseqs80.csv`
- `outputs/mmseqs80_search/rna80_connected_components.tsv`

配置要求 1,815 条规范且唯一的 RNA、1,185 个 MMseqs2 连通分量，以及
1,513/133/169 的 train/validation/test 记录数。安装脚本会在下载模型前验证字符、ID、
规范化序列、cluster 完整性和跨 split 泄漏，并把两个源文件的 SHA-256、split manifest
与计数写入 `outputs/server_setup/canonical_data_audit.json`。

```bash
bash scripts/setup_server_env.sh
source .venv/bin/activate
```

脚本把官方 `RNA-FM_pretrained.pth` 缓存到
`.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth`，记录其 SHA-256 到
`outputs/server_setup/rna_fm_sha256.txt`，并运行完整 release audit。正式配置显式引用
该文件，因此 checkpoint 元数据能够记录 RNA-FM 权重 hash。默认 LoRA 边界为最后 4 层
attention 的 q/k/v/out projection，rank 8、alpha 16、input dropout 0.05；RNA-FM 基座冻结，
adapter、项目 Transformer、投影和输出头可训练。

### V2 CPU smoke

这个 smoke 执行一次真实优化，保存并重新加载 tiny V2 checkpoint，生成至少两个唯一、
motif-preserved 候选，再在不调用 RNAfold 的情况下汇总序列指标：

```bash
python -m pytest \
  tests/test_release_audit.py::test_cpu_tiny_v2_workflow_trains_loads_generates_and_evaluates \
  -q
```

### 完整 A800 训练与恢复

```bash
python train.py --config configs/train_scaffold_a800.yaml
```

中断后只从 V2 checkpoint 恢复；版本不符会在加载时失败：

```bash
python train.py \
  --config configs/train_scaffold_a800.yaml \
  --resume checkpoints_scaffold_a800/last.ckpt
```

完整运行应保留 `last.ckpt` 和按 validation loss 选择的 checkpoint。用于正式 benchmark
的 no-RNA-FM、frozen RNA-FM、LoRA iterative 与 single-pass 路径必须分别指向真实 V2
checkpoint；缺失或推理失败会被明确标为 `unavailable`/`failed`，绝不替换成 baseline。

### 生成与 RNAfold 验证

将 `<BEST_V2_CHECKPOINT>` 替换为训练产生并按 validation loss 选出的文件：

```bash
python generate_scaffold.py \
  --motif GCGG \
  --checkpoint <BEST_V2_CHECKPOINT> \
  --num-candidates 256 \
  --max-length 512 \
  --denoise-steps 12 \
  --top-p 0.95 \
  --seed 42 \
  --output outputs/GCGG_v2_candidates.jsonl \
  --fasta-output outputs/GCGG_v2_candidates.fasta
```

```bash
python validate_scaffolds.py \
  --input outputs/GCGG_v2_candidates.jsonl \
  --output outputs/GCGG_v2_rnafold.jsonl \
  --rnafold-executable RNAfold \
  --timeout-seconds 30
```

RNAfold 状态为 `ok`、`unavailable`、`timeout`、`failed` 或 `parse_error`；未安排结构预算
时为 `not_run`。MFE 只能作为 plausibility proxy，不能单独用于功能结论。

### Benchmark smoke 与正式 benchmark

每次都使用全新的输出目录。smoke 不需要 server checkpoint，并且必须产生恰好六个 V2
artifact：

```bash
SMOKE_DIR="outputs/v2_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "${SMOKE_DIR}"
python benchmark_scaffolds.py \
  --config configs/benchmark_scaffolds.yaml \
  --output-dir "${SMOKE_DIR}" \
  --smoke-test
```

正式运行前，先把 `configs/benchmark_scaffolds.yaml` 中声明的所有 V2 checkpoint 放到位：

```bash
BENCHMARK_DIR="outputs/v2_formal_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "${BENCHMARK_DIR}"
python benchmark_scaffolds.py \
  --config configs/benchmark_scaffolds.yaml \
  --output-dir "${BENCHMARK_DIR}"
```

六个 artifact 为 `candidate_results.csv`、`motif_summary.csv`、`model_summary.csv`、
`paired_bootstrap.json`、`run_manifest.json` 和 `benchmark_report.md`。manifest 记录命令、
seed、配置/数据/checkpoint hash、软件版本、方法终态与 artifact hash。

## 发布门

```bash
python -m pytest -q
python scripts/verify_scaffold_release.py
git diff --check
python -m ruff check --ignore EXE002 \
  rna_scaffold scripts tests train.py generate_scaffold.py \
  validate_scaffolds.py benchmark_scaffolds.py
```

`verify_scaffold_release.py` 会审计 runtime Python/YAML 的 V2 边界、运行 CPU tiny
端到端工作流、检查 CLI，并如实记录 RNAfold 是否可用。完整 A800 训练和正式 benchmark
仍是服务器实验，不会被本地 smoke 结果冒充。

完整架构与验收契约见
`docs/superpowers/specs/2026-08-20-scaffold-generator-v2-design.md`。
