$ProgressPreference='SilentlyContinue'
'Perl version probe without -e quoting; prior failed -e diagnostic preserved.'
& 'C:/Users/marti/strawberry-portable/perl/bin/perl.exe' -v | Select-Object -First 4
"perl_exit=$LASTEXITCODE"
& ffmpeg -version | Select-Object -First 2
"ffmpeg_exit=$LASTEXITCODE"
