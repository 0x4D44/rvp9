$ProgressPreference='SilentlyContinue'
"host=$env:COMPUTERNAME utc=$([DateTime]::UtcNow.ToString('o'))"
rustup toolchain list
rustc -vV
cargo -vV
"rust195_directory_exists=$(Test-Path 'C:/Users/marti/.rustup/toolchains/1.95.0-x86_64-pc-windows-msvc')"
foreach($name in @('git','deltic','bash','gcc','clang','cl','link','rust-lld','ffmpeg','python','python3','py','pwsh','make')) {
 $cmd=Get-Command $name -ErrorAction SilentlyContinue
 if($cmd) { "path_tool=$name path=$($cmd.Source) type=$($cmd.CommandType)" } else { "path_tool=$name absent=True" }
}
foreach($name in @('git','ffmpeg','python')) { $cmd=Get-Command $name -ErrorAction SilentlyContinue; if($cmd) { "version_begin=$name"; & $cmd.Source --version 2>&1 | Select-Object -First 2 } }
