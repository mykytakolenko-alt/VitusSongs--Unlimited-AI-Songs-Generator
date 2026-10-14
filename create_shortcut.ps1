$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath('Desktop')
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path $desktop 'VitusSongs.lnk'))
$lnk.TargetPath = Join-Path $env:WINDIR 'System32\wscript.exe'
$lnk.Arguments = '"' + (Join-Path $root 'VitusSongs.vbs') + '"'
$lnk.WorkingDirectory = $root
$lnk.IconLocation = (Join-Path $root 'app\static\vitussongs.ico')
$lnk.Description = 'VitusSongs - AI song generator'
$lnk.Save()
