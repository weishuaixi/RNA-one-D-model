$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$output = Join-Path $root 'outputs\RNAGym_baseline_comparison.xlsx'
$summary = Import-Csv (Join-Path $root 'outputs\rnagym_baseline_summary.csv')
$details = Import-Csv (Join-Path $root 'outputs\rnagym_baseline_per_target.csv')
$errors = Import-Csv (Join-Path $root 'outputs\rnagym_baseline_errors.csv')
$testRows = Import-Csv (Join-Path $root '.codex_tmp\baseline\monomer_test.csv')
$predictionDir = Join-Path $root '.codex_tmp\baseline\ours_all'

$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
$book = $excel.Workbooks.Add()

function Set-TableStyle($range) {
    $range.Borders.LineStyle = 1
    $range.Borders.Weight = 2
    $range.VerticalAlignment = -4108
}

function Add-MetricChart($sheet, $title, $metricColumn, $left, $top, $higherIsBetter) {
    $chartObject = $sheet.ChartObjects().Add($left, $top, 500, 260)
    $chart = $chartObject.Chart
    $chart.ChartType = 51
    $series = $chart.SeriesCollection().NewSeries()
    $series.Name = $title
    $series.XValues = $sheet.Range("A7:A$($summary.Count + 6)")
    $series.Values = $sheet.Range("$metricColumn`7:$metricColumn$($summary.Count + 6)")
    $chart.HasTitle = $true
    $direction = if ($higherIsBetter) { 'higher is better' } else { 'lower is better' }
    $chart.ChartTitle.Text = "$title ($direction)"
    $chart.HasLegend = $false
    $chart.Axes(1).TickLabels.Font.Size = 9
    $chart.Axes(2).TickLabels.Font.Size = 9
    $series.Format.Fill.ForeColor.RGB = 10053120
    $series.Format.Line.Visible = 0
    $series.Points(1).Format.Fill.ForeColor.RGB = 192
    $series.Points(1).Format.Line.ForeColor.RGB = 192
    $series.HasDataLabels = $true
    $series.DataLabels().NumberFormat = '0.00'
}

$dashboard = $book.Worksheets.Item(1)
$dashboard.Name = 'Summary & Charts'
$dashboard.Cells.Font.Name = 'Arial'
$dashboard.Cells.Font.Size = 10
$dashboard.Range('A1:I1').Merge()
$dashboard.Range('A1').Value2 = 'RNAGym monomer baseline comparison'
$dashboard.Range('A1').Font.Size = 20
$dashboard.Range('A1').Font.Bold = $true
$dashboard.Range('A2:I2').Merge()
$commonN = [int]$summary[0].n
$availableN = [int]$summary[0].available_n
$dashboard.Range('A2').Value2 = "Headline metrics use the common $commonN-target subset available for every model; our model completed $availableN valid AUGC targets."
$dashboard.Range('A2').Font.Color = 5263440
$dashboard.Range('A3:I3').Merge()
$dashboard.Range('A3').Value2 = 'C1′-lDDT measures local distance agreement; Kabsch RMSD measures global fit; distance RMSD measures internal geometry; clash penetration measures steric overlap.'
$dashboard.Range('A4:I4').Merge()
$ours = $summary | Where-Object method -eq 'Our model'
$best = $summary | Sort-Object {[double]$_.c1_lddt_mean} -Descending | Select-Object -First 1
$dashboard.Range('A4').Value2 = ('Result: our model mean C1′-lDDT = {0:N2}; best baseline = {1} ({2:N2}). The model preserves local distances relatively well but global placement and steric clashes remain the main weaknesses.' -f [double]$ours.c1_lddt_mean, $best.method, [double]$best.c1_lddt_mean)
$dashboard.Range('A4').WrapText = $true

$headers = @('Model','Common N','Available N','C1′-lDDT mean ↑','Kabsch RMSD mean ↓','Distance RMSD mean ↓','Clash penetration RMS ↓','Covalent bond RMSE ↓','Base planarity RMS ↓')
for ($c=0; $c -lt $headers.Count; $c++) { $dashboard.Cells.Item(6,$c+1).Value2 = $headers[$c] }
for ($r=0; $r -lt $summary.Count; $r++) {
    $row = $summary[$r]
    $dashboard.Cells.Item($r+7,1).Value2 = [string]$row.method
    $numericValues = @($row.n,$row.available_n,$row.c1_lddt_mean,$row.c1_kabsch_rmsd_mean,$row.c1_distance_rmsd_mean,$row.clash_penetration_rms_mean,$row.covalent_bond_rmse_mean,$row.base_planarity_rms_mean)
    for ($c=0; $c -lt $numericValues.Count; $c++) { $dashboard.Cells.Item($r+7,$c+2).Value2 = [double]$numericValues[$c] }
}
$headerRange = $dashboard.Range('A6:I6')
$headerRange.Font.Bold = $true
$headerRange.Interior.Color = 15132390
$dataRange = $dashboard.Range("A6:I$($summary.Count+6)")
Set-TableStyle $dataRange
$dashboard.Range('A7:I7').Font.Bold = $true
$dashboard.Range('A7:I7').Interior.Color = 13421823
$dashboard.Range('A7').Font.Color = 192
$dashboard.Range("D7:I$($summary.Count+6)").NumberFormat = '0.000'
$dashboard.Columns.Item('A').ColumnWidth = 20
$dashboard.Columns.Item('B:I').ColumnWidth = 18
$dashboard.Rows.Item(4).RowHeight = 34
$dashboard.Application.ActiveWindow.SplitRow = 6
$dashboard.Application.ActiveWindow.FreezePanes = $true
Add-MetricChart $dashboard 'Mean C1′-lDDT' 'D' 15 300 $true
Add-MetricChart $dashboard 'Mean Kabsch RMSD (Å)' 'E' 530 300 $false
Add-MetricChart $dashboard 'Mean distance RMSD (Å)' 'F' 15 575 $false
Add-MetricChart $dashboard 'Mean clash penetration RMS (Å)' 'G' 530 575 $false

$detailSheet = $book.Worksheets.Add()
$detailSheet.Name = 'Per-target metrics'
$detailSheet.Cells.Font.Name = 'Arial'
$detailHeaders = @('target_id','method','length','c1_lddt','c1_kabsch_rmsd','c1_distance_rmsd','clash_penetration_rms','covalent_bond_rmse','backbone_angle_rmse_deg','base_planarity_rms','sugar_closure_rmse','o3_p_bond_rmse','aligned_residues','pred_c1_count','ref_c1_count')
for ($c=0; $c -lt $detailHeaders.Count; $c++) { $detailSheet.Cells.Item(1,$c+1).Value2 = $detailHeaders[$c] }
for ($r=0; $r -lt $details.Count; $r++) {
    for ($c=0; $c -lt $detailHeaders.Count; $c++) {
        $value = $details[$r].($detailHeaders[$c])
        if ($c -ge 2 -and $value -ne '') { $value = [double]$value }
        if ($c -ge 2 -and $value -ne '') { $detailSheet.Cells.Item($r+2,$c+1).Value2 = [double]$value }
        else { $detailSheet.Cells.Item($r+2,$c+1).Value2 = [string]$value }
    }
}
$detailSheet.Range("A1:O$($details.Count+1)").AutoFilter() | Out-Null
$detailSheet.Range('A1:O1').Font.Bold = $true
$detailSheet.Range('A1:O1').Interior.Color = 15132390
$detailSheet.Range("D2:L$($details.Count+1)").NumberFormat = '0.0000'
$detailSheet.Columns.AutoFit() | Out-Null
$detailSheet.Application.ActiveWindow.SplitRow = 1
$detailSheet.Application.ActiveWindow.FreezePanes = $true

$statusSheet = $book.Worksheets.Add()
$statusSheet.Name = 'Coverage & failures'
$statusSheet.Cells.Font.Name = 'Arial'
$statusHeaders = @('Target','Length','Alphabet','Our prediction','All six methods comparable','Note')
for ($c=0; $c -lt $statusHeaders.Count; $c++) { $statusSheet.Cells.Item(1,$c+1).Value2 = $statusHeaders[$c] }
$errorLookup = @{}
foreach ($failure in $errors) { $errorLookup[$failure.target_id] = "$($failure.method): $($failure.error)" }
$rowIndex = 2
foreach ($row in $testRows) {
    $target = "$($row.'PDB ID'.ToLower())_$($row.'Asym. Chain ID')"
    $sequence = $row.'Sequence (unmod.)'.Trim().ToUpper()
    $alphabet = if ($sequence -match '^[AUGC]+$') { 'AUGC' } else { 'contains N/other' }
    $predicted = Test-Path (Join-Path $predictionDir "$target.pdb")
    $comparable = $predicted -and -not $errorLookup.ContainsKey($target)
    $note = if ($errorLookup.ContainsKey($target)) { $errorLookup[$target] } elseif (-not $predicted -and $alphabet -ne 'AUGC') { 'Current model rejects ambiguous bases' } elseif (-not $predicted) { 'Inference not completed' } else { '' }
    $statusSheet.Cells.Item($rowIndex,1).Value2 = [string]$target
    $statusSheet.Cells.Item($rowIndex,2).Value2 = [double]$sequence.Length
    $statusSheet.Cells.Item($rowIndex,3).Value2 = [string]$alphabet
    $statusSheet.Cells.Item($rowIndex,4).Value2 = [string]$predicted
    $statusSheet.Cells.Item($rowIndex,5).Value2 = [string]$comparable
    $statusSheet.Cells.Item($rowIndex,6).Value2 = [string]$note
    $rowIndex++
}
$statusSheet.Range("A1:F$($rowIndex-1)").AutoFilter() | Out-Null
$statusSheet.Range('A1:F1').Font.Bold = $true
$statusSheet.Range('A1:F1').Interior.Color = 15132390
$statusSheet.Columns.AutoFit() | Out-Null

$methodSheet = $book.Worksheets.Add()
$methodSheet.Name = 'Methodology'
$methodSheet.Cells.Font.Name = 'Arial'
$methodSheet.Range('A1:F1').Merge()
$methodSheet.Range('A1').Value2 = 'Evaluation definition and interpretation'
$methodSheet.Range('A1').Font.Size = 18
$methodSheet.Range('A1').Font.Bold = $true
$methodRows = @(
    @('Cohort','RNAGym public monomer targets for which our model produced a PDB. Headline means use only targets with predictions from all six methods.'),
    @('Our input rule','Only unmodified sequences containing A/U/G/C are accepted. Ten of the 88 RNAGym targets contain N or another ambiguous symbol and are excluded.'),
    @('C1′-lDDT ↑','For C1′ atom pairs whose reference distance is below 15 Å, average the fractions with absolute distance error below 0.5, 1, 2 and 4 Å. Higher is better.'),
    @('Kabsch RMSD ↓','Rigidly align predicted and reference C1′ coordinates, then calculate root-mean-square coordinate error. Lower is better; sensitive to global fold.'),
    @('Distance RMSD ↓','RMS error between predicted and reference pairwise C1′ distances below the 15 Å reference cutoff. Lower is better and does not depend on rigid orientation.'),
    @('Clash penetration RMS ↓','RMS amount by which nonbonded atoms overlap beyond the allowed separation. Lower is better.'),
    @('Important limitation','This is a local reproducible comparison, not an official RNAGym leaderboard submission. It does not include RNA modifications, complexes, or external server reruns.'),
    @('Baseline structures','RNAGym public prediction archive: AlphaFold 3, NuFold, RoseTTAFoldNA, RhoFold+, and trRosettaRNA.'),
    @('Reference','RNAGym benchmark repository and monomer test metadata; native structures are the supplied RCSB files.'),
    @('Checkpoint','checkpoints_3d/checkpoints_3d_a800_full/rna_3d_best.pt')
)
for ($r=0; $r -lt $methodRows.Count; $r++) {
    $methodSheet.Cells.Item($r+3,1).Value2 = $methodRows[$r][0]
    $methodSheet.Cells.Item($r+3,2).Value2 = $methodRows[$r][1]
}
$methodSheet.Range("A3:A$($methodRows.Count+2)").Font.Bold = $true
$methodSheet.Range("A3:B$($methodRows.Count+2)").Borders.LineStyle = 1
$methodSheet.Range("B3:B$($methodRows.Count+2)").WrapText = $true
$methodSheet.Columns.Item('A').ColumnWidth = 24
$methodSheet.Columns.Item('B').ColumnWidth = 100
$methodSheet.Rows.AutoFit() | Out-Null

$dashboard.Activate()
$book.SaveAs($output, 51)
$book.Close($true)
$excel.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($book) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel) | Out-Null
Write-Output $output
