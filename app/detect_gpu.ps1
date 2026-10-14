# Prints "<mode> <gfx>" for install.bat: mode = cuda | amd | cpu
$names = (Get-CimInstance Win32_VideoController | ForEach-Object Name) -join '; '
$mode = 'cpu'; $gfx = 'none'
if ($names -match 'NVIDIA') { $mode = 'cuda' }
elseif ($names -match 'Radeon VII|Radeon\(TM\) VII|Radeon Pro VII') { $mode = 'amd'; $gfx = 'gfx906' }
elseif ($names -match 'RX\s*(\d{4})') {
    $n = [int]$Matches[1]; $mode = 'amd'
    if     ($n -ge 9070) { $gfx = 'gfx1201' }
    elseif ($n -ge 9060) { $gfx = 'gfx1200' }
    elseif ($n -ge 7900) { $gfx = 'gfx1100' }
    elseif ($n -ge 7700) { $gfx = 'gfx1101' }
    elseif ($n -ge 7600) { $gfx = 'gfx1102' }
    elseif ($n -ge 6800) { $gfx = 'gfx1030' }
    elseif ($n -ge 6700) { $gfx = 'gfx1031' }
    elseif ($n -ge 6600) { $gfx = 'gfx1032' }
    elseif ($n -ge 6400) { $gfx = 'gfx1034' }
    else { $mode = 'cpu' }
}
"$mode $gfx"
