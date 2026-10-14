' Starts VitusSongs without a console window and opens it as its own app window.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = root
py = root & "\.venv\Scripts\pythonw.exe"
If Not fso.FileExists(py) Then
  MsgBox "Please run install.bat first.", 48, "VitusSongs"
  WScript.Quit 1
End If
sh.Run """" & py & """ """ & root & "\app\server.py""", 0, False
