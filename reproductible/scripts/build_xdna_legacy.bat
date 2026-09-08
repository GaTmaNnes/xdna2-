@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d E:\trixdna_test\repos_soket1\build
nmake -f Makefile /nologo ggml-xdna 2>&1 | findstr /I "error C error LNK Build succeeded"
echo LEGACY_BUILD_EXIT=%ERRORLEVEL%