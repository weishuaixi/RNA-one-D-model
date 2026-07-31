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
全原子构建使用 A/U/C/G 四套独立重原子模板：已知碱基严格选择对应模板，MASK
位置按预测概率进行可微模板混合；PDB 输出只写出该碱基真实存在的原子。该变更对应
`rna_ipa_internal_coords_v10` 检查点；旧模板、错误糖苷键方向/χ 旋转轴、缺少第三边 triangle bias、
或只在第一个 trunk block 后计算序列 logits
或使用逐通道乘法伪 OPM 的检查点会被明确拒绝。
mmCIF 标签固定选择最小 PDB model number，并在每个残基内选择单一的最高总占有率
altloc；零占有率原子被排除。若目标 ID 同时命中不同的 label/auth chain，则依据
CSV 期望序列、序列覆盖率和有效原子覆盖率选择候选链。修饰核苷依据
CCD `mon_nstd_parent_comp_id`、Gemmi residue table 和判别性碱基原子模式依次映射到
A/U/C/G。结构序列到 CSV 序列使用 Gemmi/ksw2 全局 affine-gap alignment，
并利用 O3′–P 或 C1′ 空间断点消除重复序列中的 gap 歧义；碱基 mismatch 仅保留
共同 backbone 坐标，不把错误碱基原子当监督。该标签语义使用 cache version 8，
旧的混合 model/altloc 缓存不会被复用。
每次 recycle 都重新组合初始 seq/pair 与上一轮的 sequence、完整 pair tensor 和
C1′ 距离特征；上一轮特征经过可学习的 0.01 LayerScale 后注入，避免未训练模型在
recycle 1→2 时发生过大的结构跳变。非最终 recycle 默认整体 stop-gradient。
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

两份 A800 配置默认使用第 2 张卡 `cuda:1`。单卡服务器应将 `cuda_device` 改为
`0`。当 CUDA 不可用或设备编号超出实际卡数时，训练会在读取 mmCIF/构建缓存前
立即失败，不会静默退回 CPU。

`train_sequences.v2.csv` 最长序列超过 4000 nt。全通量配置读取到 1536 nt，
再按 `crop_length: 384` 采样训练片段，以控制 triangle 模块的立方计算成本。

默认输出：

```text
checkpoints_3d_a800_card1/rna_3d_best.pt
```

Windows 本地只做 smoke test 时运行：

```bash
python train_3d.py --config configs/train_3d_local_windows.yaml
```

全原子 mmCIF 会在首次加载后写入配置中的 `data.cache_path`；缓存记录源 CSV
内容哈希、实际候选 CIF 的逐文件大小/纳秒时间戳摘要和全部过滤参数，数据或配置
变化时会自动重建，且不会为了命中缓存而重复读取数十 GB CIF 内容。
`min_atom_coverage` 的分母是序列中化学上真实存在的重原子，而不是固定 27-slot
并集，因此不会因 U/C 天生缺少 purine 原子而系统性降低覆盖率。中断训练可以恢复：

```bash
python train_3d.py \
  --config configs/train_3d_a800_full.yaml \
  --resume checkpoints_3d_a800_full/rna_3d_last.pt
```

当前 checkpoint format v5 同时保存模型、优化器、scheduler、Python/NumPy、
CPU/CUDA RNG 以及 train/validation DataLoader generator 状态。正式配置默认
`persistent_workers: false`，使随机 shuffle、crop、sequence mask、SE(3) 增强、
dropout 和随机 recycle 在同一软件/硬件环境中可精确续训。跨 PyTorch/CUDA 版本或
不同硬件仍不承诺逐位一致；这是 PyTorch 本身的复现边界。checkpoint 还保存模型、
数据预处理、loss、batch/scheduler、实际 train/validation 成员及坐标标签的训练语义
SHA-256；resume 时若这些内容变化会明确拒绝。数据和输出路径、CUDA 卡编号可以迁移。
v5 还嵌入内部 split manifest 与 external-holdout manifest 的 SHA-256；发布流水线会
同时核对 best/last checkpoint，防止旧 checkpoint 与新切分清单被错误拼接。

训练/验证切分先合并同一 PDB 的链，再对所有在 k-mer 总数上可能达到阈值的
序列对计算精确 weighted Jaccard。长度上界剪枝不会漏掉近重复对，也不依赖
MinHash/LSH 的概率召回；整个连通簇只会进入一侧。
`trainer.sequence_split.manifest_path` 保存固定切分、数据指纹和泄漏审计；数据、
随机种子或聚类参数变化时旧 manifest 自动失效。该步骤用于阻止近重复泄漏，
不等同于基于序列比对或结构域注释的严格同源家族划分。
复用一个元数据匹配的 manifest 前，训练器还会重新验证 indices 完整且唯一、
indices 与 target ID 一致、PDB/精确序列不跨分区，并穷举检查全部 train×validation
序列对的精确 weighted Jaccard；内部被截断或手工改坏的文件会 fail-closed。

此外，正式 3D 配置会在内部切分之前读取
`data.external_holdout.sequences_csv`，从训练数据中剔除与官方独立验证集完全相同或
weighted k-mer Jaccard 不低于 0.8 的序列，并写出 `holdout_exclusions.json`。
该外部边界会穷举全部 train×holdout 序列对，不使用允许假阴性的 LSH 候选近似；
manifest 会记录比较对数和 `cross_pair_audit_exhaustive: true`。
当前 Stanford v2 CSV 审计发现 13 条训练记录需要排除（12 条完全相同、1 条近重复）；
未执行这一步时，`validation_sequences.csv` 不能视为独立评估集。
正式 `scripts/train_and_evaluate_3d.py` 会 fail-closed：训练配置必须将同一份
`validation_sequences.csv` 声明为 external holdout，内部切分与外部排除必须使用
相同的 k-mer 大小和 Jaccard 阈值，而且两份 manifest 必须分开保存。

3D 模型包含：

- sequence embedding
- optional MSA embedding fallback
- directed pair representation with explicit padding masks
- pair-biased sequence attention and chunked true cross-channel outer-product
  sequence-to-pair updates
- incoming/outgoing triangle multiplicative updates
- starting/ending-node triangle attention and pair transition
- explicit third-edge pair bias in starting/ending-node triangle attention
- stop-gradient recycling with random recycle counts during training
- differentiable NeRF α–ζ backbone construction with explicit χ torsion
- residue rigid frames, C3′-endo-initialized sugar pucker and base SO(3) orientation
- wwPDB CCD-derived A/U/C/G ideal geometry and χ rotation about the true C1′–N1/N9 axis
- frame-aware invariant point attention (IPA) with full pair bias/value access
- optional non-reentrant trunk activation checkpointing for cubic triangle blocks
- rigid-frame coordinate construction and SE(3)-invariant torsion refinement
  followed by exact internal-coordinate reconstruction
- distogram head
- directed orientation and contact heads
- jointly trained sequence reconstruction head
- pLDDT-like confidence head
- 27-slot canonical RNA atom tensor；写 PDB 时只输出各碱基真实存在的重原子

三维训练以 residue-frame FAPE 为主损失，并使用 Kabsch 对齐坐标损失、C1' 距离图、
与验证指标同阈值（0.5/1/2/4 Å）的可微 soft-lDDT 辅助损失、
真实目标周期 torsion/pucker 损失、糖环到碱基的相对 SO(3) 朝向损失、完整核苷酸
共价键图、跨残基磷酸二酯键、碱基平面性、
糖环闭合，以及排除 1–2/1–3 共价邻居的同残基/跨残基全原子碰撞约束。
Frame、orientation 和 χ 标签由真实序列选择：A/G 只使用 N9 与 purine χ，
U/C 只使用 N1 与 pyrimidine χ；正确糖苷原子未解析时跳过该局部标签，不用错误原子回退。
FAPE 与 Kabsch 先在每条 RNA 内按有效原子归一化，再对 batch 求均值，避免长序列以
原子数或 frame×point 数量压过短序列。目标监督只使用实验观测 mask；共价、糖环、
平面性和 clash 等物理正则使用真实序列派生的化学原子 mask，因此未解析但化学上存在
的原子仍受约束，而 A/U/C/G 中不存在的原子不会被误罚。
Clash 能量先排除完整的同残基/跨残基 1–2、1–3 共价邻居，再按每条 RNA 的有效原子数
归一化并对 RNA 求均值，避免局部严重碰撞在长链中被所有可能原子对的二次分母稀释。
训练数据默认执行随机 SE(3) 旋转/平移增强。旧版 `rna_3d_best.pt` 的参数结构与新版
模型不完全兼容；升级后应重新训练，不能把旧 checkpoint 的验证损失与新版直接比较。

Triangle 模块的计算随 crop 长度近似立方增长。配置中的 `crop_length` 用于保留长 RNA
记录的同时控制单步成本；训练 crop 随机采样。验证至少使用
`trainer.validation_crops: 3` 个固定窗口，并按 RNA 长度自动增加到
`ceil(length/crop_length)`，保证每个残基都被覆盖。每个窗口权重为该 RNA 窗口数的
倒数，短 RNA 权重为 `1`，因此每条 RNA 在 checkpoint 指标中仍然等权，且每个 epoch
的验证集合完全一致。

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

## 独立结构评估

不要使用联合 `val_loss` 代替结构指标。新版 checkpoint 可以在 Stanford 官方独立标签上
报告 Kabsch RMSD、distance RMSD、C1'-lDDT、相邻 C1' 距离和校准置信度：

```bash
python evaluate_3d.py \
  --checkpoint checkpoints_3d_a800_full/rna_3d_best.pt \
  --sequences-csv /path/to/validation_sequences.csv \
  --labels-csv /path/to/validation_labels.csv \
  --device cuda:0 \
  --recycle-counts all \
  --require-min-records 10 \
  --require-min-metric-coverage 0.9 \
  --require-min-lddt 50 \
  --require-max-kabsch-rmsd 15 \
  --require-adjacent-c1-min 4.5 \
  --require-adjacent-c1-max 7.0 \
  --require-max-plddt-mae 15 \
  --require-min-plddt-correlation 0.3 \
  --require-max-covalent-bond-rmse 0.1 \
  --require-max-backbone-angle-rmse-deg 5 \
  --require-max-clash-penetration-rms 0.05 \
  --require-max-base-planarity-rms 0.05 \
  --require-max-sugar-closure-rmse 0.1 \
  --require-max-o3-p-bond-rmse 0.1 \
  --require-max-recycle-c1-rmsd 5 \
  --output outputs/validation_metrics.json
```

也可以将 `--labels-csv` 换成 `--cif-dir /path/to/PDB_RNA` 评估未参与训练的
全原子数据。全原子参考存在时，报告还会增加 α–ζ/χ 的周期误差和圆周 MAE，
以及 sugar-pucker phase MAE、糖环局部 frame 到碱基局部 frame 的 SO(3) 朝向
MAE；可分别用 `--require-max-torsion-mae-deg`、
`--require-max-sugar-pucker-mae-deg` 和
`--require-max-base-orientation-mae-deg` 设为发布门槛。单 C1′ 标签无法定义
这些角度，因此不会伪造对应结果或覆盖率。
对训练数据内部划出的全原子验证集，必须同时传入训练产生的
`--split-manifest /path/to/split_manifest.json`；evaluator 会只选择
`val_target_ids`、拒绝 train/val ID 重叠或缺失目标，并按 manifest 的原始 indices
重建当前 target+sequence 指纹；target ID 不变但序列内容变化的 stale manifest 也会
被拒绝。训练器与 evaluator 共用同一个 partition validator，都会重新检查 PDB 分组、
精确序列和全部跨分区近重复边界。通过验证的 manifest SHA-256 会写入报告。
任一显式质量门槛不满足时命令会以非零状态退出，适合服务器训练后的自动验收。
`--require-min-records` 防止验证集意外缩小，`--require-min-metric-coverage` 要求每个启用的
质量指标都在足够比例的 target 上可计算，避免 NaN 被汇总阶段静默忽略。
`--require-min-target-pass-fraction` 进一步要求每个启用阈值在指定比例的单条 RNA
上分别通过；正式流水线设为 0.9，因此少数灾难性折叠不能被良好的平均值掩盖，
缺失或非有限的单 target 指标也计为未通过。
全原子自洽指标不依赖 validation 的全原子真值：键长、clash 穿透、碱基平面、
糖环闭合和 O3′–P 使用 Å，主链键角使用度。因此即使官方标签只提供 C1′，
退化或断裂的全原子结构也不能仅凭 C1′ 指标通过验收。
评估 JSON 使用严格标准格式（缺失指标写为 `null`，不写裸 `NaN`），并记录 evaluator
格式/架构版本、checkpoint、sequence CSV、label CSV 的 SHA-256 和全部评估参数。
全原子评估会另外记录实际进入报告的唯一 CIF 文件清单、逐文件 SHA-256 和确定性
聚合哈希；未评估的目录文件不会污染指纹，参考 CIF 内容变化则一定改变报告 provenance。
`--labels-csv` 默认自动读取标签中的全部参考构象（例如 `x_1..x_40`），丢弃覆盖率
低于 `--min-reference-coverage` 的参考，并为每条RNA报告最佳匹配实验构象。此外会报告
pLDDT 对真实逐残基 C1'-lDDT 的 MAE 与 Pearson 相关性，避免只看平均置信度。
`--recycle-counts all` 会分别运行 checkpoint 支持的每个 recycle 次数，报告各次数到
最终次数的 C1′ Kabsch/distance RMSD，并以最大漂移执行
`--require-max-recycle-c1-rmsd` 稳定性门槛；超出 checkpoint 配置的次数会明确报错，
不会被模型静默截断。

在 A800 服务器上可以用一个命令完成正式训练和带非零失败状态的独立验收：

```bash
EVAL_DEVICE=cuda:1 bash scripts/train_and_evaluate_3d.sh \
  "/home/weisx/workdir/igem one-model/stanford-rna-3d-folding-data"
```

脚本第二至第五个可选参数依次为训练配置、checkpoint 路径、外部 C1′ 评估 JSON
和内部 held-out 全原子评估 JSON。未指定第五个参数时，后者自动命名为
`<外部报告名>_all_atom.json`。流程会依次执行训练、官方多参考 C1′ 验收和
split-manifest 限定的全原子验收；后者额外要求 torsion MAE ≤ 30°、pucker phase
MAE ≤ 25°、base orientation MAE ≤ 25°。只有训练正常结束且两套独立质量门
全部通过时，脚本才返回成功。

## 测试

```bash
python -m pytest -q
```
