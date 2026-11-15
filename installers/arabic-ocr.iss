; Windows installer (Inno Setup 6). Built by installers/build.py, which passes:
;   VERSION  e.g. 1.0.0
;   SOURCE   the PyInstaller folder (dist/arabic-ocr)
;   OUTPUT   where the setup .exe goes (dist)
;
; Installs for the current user by default (no administrator needed); the
; first page lets the user choose "all users" instead. Options: desktop icon,
; and adding the `arabic-ocr` command-line tool to PATH.

#ifndef VERSION
  #define VERSION "0.0.0"
#endif

[Setup]
AppId={{6C1E2F4A-8B7D-4E25-9A3C-2D7F1B5E9A40}
AppName=Arabic OCR
AppVersion={#VERSION}
AppVerName=Arabic OCR {#VERSION}
AppPublisher=arabic-ocr
AppPublisherURL=https://github.com/youssef-mhmd/arabic-ocr
AppSupportURL=https://github.com/youssef-mhmd/arabic-ocr/issues
DefaultDirName={autopf}\Arabic OCR
DefaultGroupName=Arabic OCR
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OUTPUT}
OutputBaseFilename=arabic-ocr-{#VERSION}-windows-x64-setup
SetupIconFile=icons\arabic-ocr.ico
UninstallDisplayIcon={app}\arabic-ocr-gui.exe
UninstallDisplayName=Arabic OCR {#VERSION}
LicenseFile=..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
ChangesEnvironment=yes
ShowLanguageDialog=auto

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "arabic"; MessagesFile: "compiler:Languages\Arabic.isl"

[CustomMessages]
english.AddToPath=Add the arabic-ocr command-line tool to PATH
arabic.AddToPath=إضافة أداة سطر الأوامر arabic-ocr إلى PATH

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addtopath"; Description: "{cm:AddToPath}"; Flags: unchecked

[Files]
Source: "{#SOURCE}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; an upgrade replaces the bundled libraries completely
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\Arabic OCR"; Filename: "{app}\arabic-ocr-gui.exe"
Name: "{autodesktop}\Arabic OCR"; Filename: "{app}\arabic-ocr-gui.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\arabic-ocr-gui.exe"; Description: "{cm:LaunchProgram,Arabic OCR}"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsAddPath(Dir: string): Boolean;
var
  Path: string;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', Path) then
    Path := '';
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Path) + ';') = 0;
end;

procedure RemoveFromPath(Dir: string);
var
  Path: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', Path) then
    exit;
  P := Pos(';' + Uppercase(Dir), Uppercase(Path));
  if P > 0 then
  begin
    Delete(Path, P, Length(Dir) + 1);
    RegWriteExpandStringValue(HKCU, 'Environment', 'Path', Path);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromPath(ExpandConstant('{app}'));
end;
