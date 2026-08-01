# RNA-one-D-model 技术汇报 PPT 设计规格

## 目标

制作一份可直接用于项目汇报和技术答辩的中文 PowerPoint，准确说明当前
`rna_ipa_internal_coords_v10` 实现的工程架构、算法模块、训练方法、损失函数、
推理输出、评价体系、技术边界与局限性。表达专业但不夸大：将系统描述为
“motif 条件序列补全与全原子三维结构预测的联合推理框架”，明确离散序列解码和
候选排序不是完全可微的纯端到端路径。

## 受众与表达原则

- 面向具有基础深度学习知识、可能不了解 RNA 结构预测细节的技术评审或教师。
- 正文使用专业中文；模块名、张量名和公认术语保留英文。
- 每页只回答一个核心问题，先给结论，再解释机制。
- 技术陈述以当前仓库代码、训练配置和测试为依据。
- 不把模型表述为 AlphaFold、RhoFold 的完整复现，也不宣称未经正式评估的数据表现。
- 不把 SE(3)-invariant torsion refinement 夸大为整个网络严格 SE(3)-equivariant。

## 叙事结构

建议 22 页，形成“问题—总体方案—核心表示—结构生成—训练目标—工程实现—评价与答辩”的完整链条：

1. 标题页：RNA motif 条件序列补全与全原子三维结构预测
2. 项目解决什么问题：输入、约束、输出和使用场景
3. 系统是否端到端：用户接口端到端与训练梯度边界
4. 项目工程架构：入口脚本、模型、数据、几何、损失、评估模块
5. 总体数据流：motif prompt → sequence decoding → folding → ranking
6. 数据表示：sequence tensor、pair tensor、mask、all-atom coordinates
7. Sequence/Pair 初始化：embedding、relative position、方向/距离/接触通道
8. 二维信息注入序列：pair-biased attention 与 attention pooling
9. Sequence-to-Pair 更新：Outer Product Mean
10. Triangle multiplicative update：incoming/outgoing
11. Triangle attention：starting/ending node、third-edge bias、pair transition
12. Recycling：initial + recycled feature、stop-gradient、随机次数
13. 结构模块总览：完整 pair tensor → IPA → rigid frame/torsion heads
14. 全原子坐标生成：frame、α–ζ/χ、sugar pucker、base orientation、NeRF/template
15. 几何性质：局部 frame、SE(3) 不变信息与模型边界
16. 多任务输出：sequence、coords、pLDDT、distogram、orientation/contact
17. 总损失函数：结构主损失与辅助损失的加权组合
18. 核心结构损失：FAPE、Kabsch aligned coordinate、distance、soft-lDDT
19. RNA 几何损失：torsion、bond/angle、跨残基连接、平面性、糖环、clash
20. 数据与训练：CIF/CCD、修饰残基、切分防泄漏、随机旋转、A800 配置
21. 推理、输出与评价：`.pt`、`.pdb`、C1′-lDDT、RMSD、几何误差、recycle stability
22. 技术边界与答辩高频问题：优势、限制、标准回答

## 核心图形

仅使用三张高价值图，避免把整份汇报做成密集流程图：

1. 总体联合推理架构图：清楚区分补全阶段、结构预测阶段和候选排序阶段。
2. Trunk 与结构模块图：展示 sequence/pair 双轨表示、E2Eformer block、recycling、IPA 和全原子构建。
3. 损失函数分组图：结构对齐、局部 frame、pair geometry、RNA chemistry、confidence/sequence 五组监督。

其余页面使用公式、关键张量形状、简洁流程和少量强调文字，不使用仪表盘式卡片堆砌。

## 视觉规范

- 16:9 宽屏；深蓝底或深蓝标题区，青绿色用于 sequence/pair 数据流强调，橙色用于结构/几何强调。
- 标题页简洁；主标题至少 50 pt，页标题至少 35 pt，正文至少 16 pt。
- 公式使用独立区域，变量在邻近位置解释。
- 相邻页面更换构图轮廓，避免连续使用相同的左右分栏。
- 架构图的箭头语义统一：实线表示前向数据流，虚线表示 recycling，灰色边界表示不可微离散步骤。
- 页脚包含项目版本 `rna_ipa_internal_coords_v10` 与页码。

## 技术准确性要求

- Pair tensor 的形状表述为 `[B, L, L, C_pair]`，所有更新显式应用 pair mask。
- Distance logits 仅在 distogram 输出前对称化；方向和取向信息保留非对称性。
- FAPE 是 residue-frame 中的全原子局部坐标误差，为当前配置主损失，权重为 1.0。
- Kabsch 只用于坐标损失前消除全局刚体变换，不替代局部 frame 监督。
- RNA torsion 使用归一化 `(sin θ, cos θ)` 周期表示，覆盖 α、β、γ、δ、ε、ζ、χ。
- 全原子输出采用每个残基 27 个规范重原子槽位和 atom mask。
- 推理输出同时包括补全后的 A/U/C/G 序列与对应三维坐标；“折叠后的序列”不是一种文件格式。
- 当前 sequence head 的离散解码与候选排序截断梯度，因此严格意义上不是全路径可微生成模型。
- 当前结构模块包含 IPA 和 SE(3)-invariant torsion refinement，但不能宣称整个 trunk 严格 E(3)/SE(3) 等变。

## 损失函数呈现

总损失以代码中的动态结构/序列任务权重为外层，并在结构损失中展示当前 A800 配置权重：

- FAPE `1.0`
- Kabsch aligned coordinate `0.2`
- raw coordinate `0.0`
- pairwise distance `0.5`
- local distance `0.5`
- differentiable soft-lDDT `0.5`
- distogram CE `0.1`
- orientation CE `0.1`
- clash `0.02`
- covalent bond `0.05`
- angle `0.2`
- torsion `0.02`
- confidence `0.01`
- inter-residue connectivity `0.2`
- base planarity `0.05`
- sugar closure `0.1`
- pucker `0.05`
- base orientation `0.1`

PPT 需要解释每一类损失解决的失败模式，而不是只罗列名称。

## 答辩准备

最后一页和演讲者备注覆盖以下问题及短回答：

- 这是端到端模型吗？
- Pair representation 为什么必要？
- 为什么需要 triangle update/attention？复杂度是多少？
- FAPE 与 Kabsch RMSD 有何区别？
- 模型是否严格满足 SE(3) 等变？
- 为什么使用 27 个原子槽位？
- 离散序列采样能否反向传播？
- pLDDT 是否等同于真实准确率？
- 数据切分如何避免同源泄漏？
- 与 AlphaFold/RhoFold 相比有哪些差距？

## 验收标准

- 输出一个可编辑 `.pptx`，所有页面可正常渲染。
- 无文字越界、意外重叠、断裂箭头、两行标题或不可读公式。
- 每项架构和损失陈述均可追溯到仓库实现或配置。
- 最终逐页渲染检查，并通过自动 overflow 检查。
- 演讲者备注包含必要的解释与答辩补充，不在观众可见页面暴露制作说明。
