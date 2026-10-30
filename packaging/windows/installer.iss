; Inno Setup script for the app. build.py passes the values below with /D.
;
; The AppId is fixed on purpose: it is what Windows uses to recognise an
; earlier installation, so a new installer upgrades the old one in place.
; Never change it, not even for the rename from mlgidLAB to mlgidGUI.

#ifndef AppName
  #define AppName "mlgidGUI"
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef AppPublisher
  #define AppPublisher "mlgid-project"
#endif
#ifndef AppUrl
  #define AppUrl "https://github.com/mlgid-project/mlgidGUI"
#endif
#ifndef AppExe
  #define AppExe AppName + ".exe"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\" + AppName
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif
#ifndef OutputBase
  #define OutputBase AppName + "-" + AppVersion + "-windows-x64-setup"
#endif
#ifndef IconFile
  #define IconFile "..\..\mlgidgui\assets\app\mlgidgui.ico"
#endif

[Setup]
AppId={{86E45B58-DF97-49EF-B794-9F53E47E7BD0}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppUrl}
AppSupportURL={#AppUrl}/issues
AppUpdatesURL={#AppUrl}/releases
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\..\LICENSE
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBase}
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#AppExe}
; Per user by default, so no administrator is needed and the app can write
; next to itself; the dialog still offers an all-users install.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

#if AppName != "mlgidLAB"
[InstallDelete]
; An installation made while the app was still called mlgidLAB has the same
; AppId, so this installer upgrades it in place. Its launcher and shortcuts
; carry the old name and would be left behind, pointing at nothing useful.
Type: files; Name: "{app}\mlgidLAB.exe"
Type: files; Name: "{autoprograms}\mlgidLAB.lnk"
Type: files; Name: "{autodesktop}\mlgidLAB.lnk"
#endif

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
