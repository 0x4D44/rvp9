param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
$checks=@();$failures=@()
foreach($row in Import-Csv ($Root+'\bdrate.csv')){
 if($row.role -notin @('anchor','test')){continue}
 $file=$Root+'\streams\'+$row.role+'_'+$row.target_kbps+'k.ivf'
 $data=[IO.File]::ReadAllBytes($file);$offset=[BitConverter]::ToUInt16($data,6);$count=0
 while($offset -lt $data.Length){$packet=[BitConverter]::ToUInt32($data,$offset);$offset+=12+$packet;$count++}
 if($count -ne 60){throw "FRAME_AUDIT_ASSERTION expected60 got$count"}
 $expected=$data.Length*8.0/1000.0/(60.0/30.0)
 $actual=[double]::Parse($row.measured_kbps,[Globalization.CultureInfo]::InvariantCulture)
 $pass=[Math]::Abs($expected-$actual) -le 0.00051
 $checks+=@{stream=[IO.Path]::GetFileName($file);bytes=$data.Length;frames=$count;clock_rate=[BitConverter]::ToUInt32($data,16);measured_kbps=$actual;independent_expected_kbps=$expected;pass=$pass;sha256=(Get-FileHash $file -Algorithm SHA256).Hash}
 if(-not $pass){$failures+="BITRATE_AUDIT_ASSERTION stream=$file actual=$actual expected=$expected"}
}
$checks | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 ($Root+'\rate-audit.json')
if($checks.Count -ne 8){throw "AUDIT_SELECTION_ASSERTION expected8 got$($checks.Count)"}
if($failures.Count){throw ($failures -join "`n")}
"RATE_AUDIT_PASS selected=$($checks.Count)"