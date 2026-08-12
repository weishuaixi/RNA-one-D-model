$ErrorActionPreference = 'Stop'
$out = 'F:\github_item\RNA-one-D-model\RNA_model_iGEM_ACEM.pptx'
$preview = 'F:\github_item\RNA-one-D-model\.codex_tmp\ppt_build\preview'
New-Item -ItemType Directory -Force -Path $preview | Out-Null

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$pres = $ppt.Presentations.Add()
$pres.PageSetup.SlideWidth = 13.333 * 72
$pres.PageSetup.SlideHeight = 7.5 * 72

$W = $pres.PageSetup.SlideWidth; $H = $pres.PageSetup.SlideHeight
$bg = 0xF7F4EE; $ink = 0x28231F; $muted = 0x716A63; $orange = 0x2567C9; $pale = 0xE8F0FA; $green = 0x5B8C66; $red = 0x4856B9
$font = 'Microsoft YaHei'

function Add-Text($s,$text,$x,$y,$w,$h,$size=20,$bold=$false,$color=$ink,$align=1) {
  $sh=$s.Shapes.AddTextbox(1,$x,$y,$w,$h); $sh.TextFrame.TextRange.Text=$text
  $sh.TextFrame.TextRange.Font.Name=$font; $sh.TextFrame.TextRange.Font.Size=$size
  $sh.TextFrame.TextRange.Font.Bold=if($bold){-1}else{0}; $sh.TextFrame.TextRange.Font.Color.RGB=$color
  $sh.TextFrame.TextRange.ParagraphFormat.Alignment=$align; $sh.TextFrame.WordWrap=-1
  $sh.TextFrame.MarginLeft=4; $sh.TextFrame.MarginRight=4; $sh.TextFrame.MarginTop=2; $sh.TextFrame.MarginBottom=2
  return $sh
}
function Add-Title($s,$title,$kicker='RNA MOTIF COMPLETION & 3D FOLDING') {
  Add-Text $s $kicker 54 24 850 24 11 $true $orange | Out-Null
  Add-Text $s $title 54 53 850 55 27 $true $ink | Out-Null
  $line=$s.Shapes.AddShape(1,54,112,72,4); $line.Fill.ForeColor.RGB=$orange; $line.Line.Visible=0
}
function Add-Footer($s,$n) { Add-Text $s ("RNA-one-D-model  •  iGEM Engineering  •  {0:00}" -f $n) 54 516 850 16 9 $false $muted | Out-Null }
function Add-Box($s,$x,$y,$w,$h,$fill=$pale,$line=$orange) {
  $b=$s.Shapes.AddShape(5,$x,$y,$w,$h); $b.Fill.ForeColor.RGB=$fill; $b.Line.ForeColor.RGB=$line; $b.Line.Weight=1
  return $b
}
function Add-Bullets($s,$items,$x,$y,$w,$h,$size=17,$color=$ink) {
  $t=($items | ForEach-Object { "• $_" }) -join "`r"
  $sh=Add-Text $s $t $x $y $w $h $size $false $color
  $sh.TextFrame.TextRange.ParagraphFormat.SpaceAfter=9
  return $sh
}
function New-Slide() { $s=$pres.Slides.Add($pres.Slides.Count+1,12); $s.FollowMasterBackground=0; $s.Background.Fill.ForeColor.RGB=$bg; return $s }

# 1
$s=New-Slide
Add-Text $s 'RNA motif 条件补全与全原子 3D 折叠' 58 105 840 85 34 $true $ink | Out-Null
Add-Text $s 'Attribution · Contribution · Engineering · Measurement' 60 204 820 35 21 $true $orange | Out-Null
Add-Text $s '从短序列输入，到可测量、可复现、可进入湿实验的候选设计系统' 60 265 760 56 18 $false $muted | Out-Null
$b=Add-Box $s 60 365 840 82 0xFFFFFF $orange
Add-Text $s '核心主张' 82 382 110 22 13 $true $orange | Out-Null
Add-Text $s '贡献不是“又一个折叠模型”，而是一条 motif → 完整 RNA → 3D → 候选排序 → 实验验证的工程闭环。' 190 376 680 46 18 $true $ink | Out-Null
Add-Footer $s 1

# 2
$s=New-Slide; Add-Title $s '我们解决的是一个完整设计问题';
$labels=@('固定 motif','补全序列','全原子折叠','质量测量','实验候选')
$xs=@(55,230,405,580,755)
for($i=0;$i -lt 5;$i++){ $b=Add-Box $s $xs[$i] 210 140 82 0xFFFFFF $orange; Add-Text $s $labels[$i] ($xs[$i]+10) 232 120 28 17 $true $ink 2|Out-Null; if($i -lt 4){Add-Text $s '→' ($xs[$i]+143) 230 30 30 22 $true $orange 2|Out-Null}}
Add-Text $s '完整序列直接折叠仍然保留，因为它是流水线的可复用子模块；真正的项目入口是短 motif。' 93 354 775 52 20 $true $ink 2 | Out-Null
Add-Footer $s 2

# 3
$s=New-Slide; Add-Title $s 'Attribution：把来源、用途和改动写清楚'
Add-Text $s '我们自主完成' 58 153 260 30 20 $true $orange | Out-Null
Add-Bullets $s @('motif 固定与条件补全任务','序列—结构联合候选排序','RNA 原子槽位与几何约束组合','数据泄漏审计、训练和推理流程') 58 190 380 240 16 | Out-Null
Add-Text $s '借鉴并适配' 510 153 260 30 20 $true $orange | Out-Null
Add-Bullets $s @('Transformer / BERT：序列表示与 Mask','AlphaFold2 / RhoFold+：Pair、IPA、FAPE','NeRF：内坐标到笛卡尔坐标','RNA 构象、Kabsch、lDDT、MolProbity') 510 190 390 240 16 | Out-Null
Add-Text $s '表述原则：inspired by / adapted from / implemented based on；不把已有方法写成自主发明。' 58 451 840 40 15 $true $red 2 | Out-Null
Add-Footer $s 3

# 4
$s=New-Slide; Add-Title $s 'Attribution 还包括工具、数据和人员'
Add-Bullets $s @('数据：wwPDB、PDBx/mmCIF、Chemical Component Dictionary','软件：PyTorch、CUDA、NumPy、Biopython、W&B、Git/GitHub','可视化与外部服务：PyMOL、ChimeraX、AlphaFold Server（如实际使用）','算力与人员：GPU 服务器提供者、指导老师、湿实验协作者','AI 辅助：记录用于代码检查、调试和文档整理的范围；科学判断由团队复核') 86 158 790 270 18 | Out-Null
$b=Add-Box $s 86 438 790 48 0xFFFFFF $orange
Add-Text $s '建议交付：Attribution 表 = 来源｜版本/链接｜用途｜我们的修改｜负责人' 102 451 760 25 16 $true $ink 2 | Out-Null
Add-Footer $s 4

# 5
$s=New-Slide; Add-Title $s 'Contribution 1：motif 条件生成—折叠一体化流水线'
Add-Bullets $s @('输入不必是完整序列：短 motif 作为不可修改条件','批量生成多个候选，并保留 motif 位置与约束','对每个候选执行全原子三维预测','联合序列概率、结构置信度和化学合理性排序','输出 FASTA、JSON、PT、PDB 与可审计排名') 58 160 510 280 18 | Out-Null
$b=Add-Box $s 625 175 260 205 0xFFFFFF $orange
Add-Text $s '可复用接口' 650 198 210 30 18 $true $orange 2|Out-Null
Add-Text $s 'predict.py`r--motif AUGGAC`r--num-candidates 32`r--output results/' 650 250 210 100 17 $true $ink 1|Out-Null
Add-Text $s '未来队伍可以直接替换 motif、checkpoint 或排序规则。' 625 405 260 60 15 $false $muted 2|Out-Null
Add-Footer $s 5

# 6
$s=New-Slide; Add-Title $s 'Contribution 2：可审计的 RNA 3D 训练框架'
Add-Text $s '数据层' 62 156 210 28 19 $true $orange | Out-Null
Add-Bullets $s @('mmCIF/CCD 解析','修饰残基标准化','holdout 与相似性审计') 62 194 240 150 16|Out-Null
Add-Text $s '模型层' 355 156 210 28 19 $true $orange | Out-Null
Add-Bullets $s @('Sequence/Pair/IPA','RNA 扭转角与全原子重建','多任务几何损失') 355 194 240 150 16|Out-Null
Add-Text $s '复现层' 650 156 210 28 19 $true $orange | Out-Null
Add-Bullets $s @('固定配置、seed、commit','PID、日志、checkpoint','模型卡与推理示例') 650 194 240 150 16|Out-Null
Add-Text $s '当前数据审计：1,807 条可用 RNA；train 1,717，validation 90；完全相同序列跨集合重叠为 0。' 72 399 820 60 18 $true $ink 2|Out-Null
Add-Footer $s 6

# 7
$s=New-Slide; Add-Title $s 'Engineering：用 Design–Build–Test–Learn 组织证据'
$names=@('DESIGN','BUILD','TEST','LEARN'); $des=@('定义问题、假设、限制与验收标准','实现数据、网络、损失与输出','用独立数据和故障实验验证','根据证据修改配置与方法')
for($i=0;$i -lt 4;$i++){ $x=55+$i*225; $b=Add-Box $s $x 175 190 190 0xFFFFFF $orange; Add-Text $s $names[$i] ($x+15) 198 160 26 18 $true $orange 2|Out-Null; Add-Text $s $des[$i] ($x+19) 250 152 78 16 $false $ink 2|Out-Null; if($i -lt 3){Add-Text $s '→' ($x+192) 250 32 30 22 $true $orange 2|Out-Null}}
Add-Text $s '失败数据同样是工程证据：必须记录现象、根因、修改和复验结果。' 70 420 820 45 19 $true $red 2|Out-Null
Add-Footer $s 7

# 8
$s=New-Slide; Add-Title $s '失败不是附录：它说明系统如何变得可靠'
$rows=@(
@('BF16 SVD 失败','Kabsch 内部转 FP32','loss 有限，gradient finite'),
@('checkpointing 梯度异常','暂时关闭 checkpointing','主训练链路可运行'),
@('crop 384 显存不足','降至 256；优化 allocator','80 GB A800 稳定训练'),
@('日志被移动后显示 deleted','唯一 run 目录与 PID/日志规范','进程与产物可追踪')
)
Add-Text $s '现象' 56 150 250 30 17 $true $orange|Out-Null; Add-Text $s '修改' 340 150 270 30 17 $true $orange|Out-Null; Add-Text $s '复验' 655 150 260 30 17 $true $orange|Out-Null
$y=190
foreach($r in $rows){$b=Add-Box $s 50 $y 870 58 0xFFFFFF 0xD8D0C8; Add-Text $s $r[0] 66 ($y+13) 245 34 15 $true $ink|Out-Null; Add-Text $s $r[1] 340 ($y+13) 280 34 15 $false $ink|Out-Null; Add-Text $s $r[2] 655 ($y+13) 245 34 15 $false $ink|Out-Null; $y+=70}
Add-Footer $s 8

# 9
$s=New-Slide; Add-Title $s '验收标准必须在看结果之前定义'
$criteria=@('数据：无完全相同序列跨 split；holdout 不进入训练','数值：loss/gradient 无 NaN 或 Inf；训练可恢复','补全：motif 100% 保留；只生成合法碱基','输出：PDB 可读取；有效残基原子完整','几何：碰撞、键长、键角、链连续性通过阈值','结构：独立集报告 RMSD、lDDT 与 motif RMSD','复现：冻结数据版本、commit、配置、seed、checkpoint','决策：模型排名能够改变下一批实验候选')
Add-Bullets $s $criteria 68 145 830 320 16 | Out-Null
Add-Text $s '阈值从 validation 基线中确定后冻结，不能在看到 holdout 结果后调整。' 75 465 820 34 15 $true $red 2|Out-Null
Add-Footer $s 9

# 10
$s=New-Slide; Add-Title $s 'Measurement：先证明生成与结构本身可靠'
Add-Text $s '序列生成' 62 150 260 30 20 $true $orange|Out-Null
Add-Bullets $s @('motif retention rate','合法率与生成成功率','uniqueness / diversity','训练集最近邻相似度','遮盖恢复准确率') 62 190 320 225 16|Out-Null
Add-Text $s '三维与化学质量' 505 150 320 30 20 $true $orange|Out-Null
Add-Bullets $s @('Kabsch RMSD、lDDT、motif RMSD','扭转角周期误差','clash score','键长/键角异常率','链断裂、核糖闭合与碱基平面性') 505 190 390 225 16|Out-Null
Add-Text $s '所有指标报告均值、分布和失败样本；不能只展示最好的一条结构。' 92 445 780 42 17 $true $ink 2|Out-Null
Add-Footer $s 10

# 11
$s=New-Slide; Add-Title $s '最有价值的 Measurement 是干湿实验闭环'
$labels=@('生成候选','按模型评分分层','湿实验测量','检验关联','更新设计规则')
$subs=@('固定 motif','高/中/低分','表达/结合/功能','Spearman、AUC、Top-k','下一轮候选')
for($i=0;$i -lt 5;$i++){ $x=40+$i*184; $b=Add-Box $s $x 205 150 115 0xFFFFFF $orange; Add-Text $s $labels[$i] ($x+8) 222 134 26 16 $true $ink 2|Out-Null; Add-Text $s $subs[$i] ($x+8) 270 134 30 13 $false $muted 2|Out-Null; if($i -lt 4){Add-Text $s '→' ($x+151) 245 32 30 21 $true $orange 2|Out-Null}}
Add-Text $s '实验设计控制长度、GC 含量、motif 和实验条件；同时设置随机选择或简单规则作为基线。' 70 385 820 55 17 $true $ink 2|Out-Null
Add-Text $s '模型的目标：提高实验成功候选的富集率，而不是宣称计算结构等同于真实结构。' 70 450 820 38 16 $true $red 2|Out-Null
Add-Footer $s 11

# 12
$s=New-Slide; Add-Title $s '下一步：把模型变成可评审、可复用、可验证的成果'
Add-Bullets $s @('冻结第一版 Attribution 表与参考文献','发布一键推理接口、示例输入和模型卡','冻结 validation 基线与 acceptance criteria','建立高/中/低评分候选的湿实验设计','在 Wiki 同时展示成功、失败、限制和复现命令') 95 165 760 235 19|Out-Null
$b=Add-Box $s 95 420 760 62 $orange $orange
Add-Text $s '最终证据链：来源透明 → 方法可复现 → 指标可测量 → 实验能验证 → 未来队伍可复用' 112 437 730 30 17 $true 0xFFFFFF 2|Out-Null
Add-Footer $s 12

# 13 references
$s=New-Slide; Add-Title $s '核心方法来源（节选）'
$refs=@('Vaswani et al. Attention Is All You Need. NeurIPS (2017).','Devlin et al. BERT. NAACL (2019). DOI: 10.18653/v1/N19-1423','Jumper et al. AlphaFold2. Nature (2021). DOI: 10.1038/s41586-021-03819-2','Shen et al. RhoFold+. Nature Methods (2024). DOI: 10.1038/s41592-024-02487-0','Parsons et al. NeRF. J. Comput. Chem. (2005). DOI: 10.1002/jcc.20237','Richardson et al. RNA backbone conformers. RNA (2008). DOI: 10.1261/rna.657708','Mariani et al. lDDT. Bioinformatics (2013). DOI: 10.1093/bioinformatics/btt473','Chen et al. MolProbity. Acta Cryst. D (2010). DOI: 10.1107/S0907444909042073','Szikszai et al. RNA3DB. JMB (2024). DOI: 10.1016/j.jmb.2024.168552')
Add-Bullets $s $refs 62 137 840 335 14|Out-Null
Add-Footer $s 13

$pres.SaveAs($out,24)
$pres.Export($preview,'PNG',1600,900)
$pres.Close(); $ppt.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres)|Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt)|Out-Null
Write-Output $out
