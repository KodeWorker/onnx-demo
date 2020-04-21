:: Copyright (C) 2018-2019 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

@echo off
setlocal enabledelayedexpansion

set TARGET=CPU

set ROOT_DIR=%~dp0
set INTEL_OPENVINO_DIR=C:\Program Files (x86)\IntelSWTools\openvino
set TARGET_PRECISION=FP16

set model_name=opt-efficientnet-b0
set irs_path=%ROOT_DIR%\converted_model\%model_name%
set target_image_path=D:\Datasets\Kaggle\dogs-vs-cats\train\cat\cat.1.jpg

if exist "%INTEL_OPENVINO_DIR%\bin\setupvars.bat" (
    call "%INTEL_OPENVINO_DIR%\bin\setupvars.bat"
) else (
    echo setupvars.bat is not found, INTEL_OPENVINO_DIR can't be set
    goto error
)

echo INTEL_OPENVINO_DIR is set to %INTEL_OPENVINO_DIR%

:: Check if Python is installed
python --version 2>NUL
if errorlevel 1 (
   echo Error^: Python is not installed. Please install Python 3.5 ^(64-bit^) or higher from https://www.python.org/downloads/
   goto error
)

:: Check if Python version is equal or higher 3.4
for /F "tokens=* USEBACKQ" %%F IN (`python --version 2^>^&1`) DO (
   set version=%%F
)
echo %var%

for /F "tokens=1,2,3 delims=. " %%a in ("%version%") do (
   set Major=%%b
   set Minor=%%c
)

if "%Major%" geq "3" (
   if "%Minor%" geq "5" (
	set python_ver=okay
   )
)
if not "%python_ver%"=="okay" (
   echo Unsupported Python version. Please install Python 3.5 ^(64-bit^) or higher from https://www.python.org/downloads/
   goto error
)

set ir_dir=%ROOT_DIR%\converted_model\%model_name%
set downloader_dir=%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\downloader

::echo.
::echo ###############^|^| Install Model Optimizer prerequisites ^|^|###############
::echo.
::timeout 3
::cd "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\install_prerequisites"
::call install_prerequisites_caffe.bat
::if ERRORLEVEL 1 GOTO errorHandling

::timeout 7


:::echo.
:::echo ###############^|^| Run Model Optimizer ^|^|###############
:::echo.
:::timeout 3

::set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
:::echo python "%downloader_dir%\converter.py" --mo "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\mo.py" --input_model "%model_name%.onnx" -o "%irs_path%" --precisions "%TARGET_PRECISION%"
:::python "%downloader_dir%\converter.py" --mo "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\mo.py" --input_model "%model_name%.onnx" -o "%irs_path%" --precisions "%TARGET_PRECISION%"
:::if ERRORLEVEL 1 GOTO errorHandling

:::timeout 7

echo.
echo ###############^|^| Generate VS solution for Inference Engine samples using cmake ^|^|###############
echo.
timeout 3

if "%PROCESSOR_ARCHITECTURE%" == "AMD64" (
   set "PLATFORM=x64"
) else (
   set "PLATFORM=Win32"
)

set VSWHERE="false"
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
   set VSWHERE="true"
   cd "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
) else if exist "%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe" (
      set VSWHERE="true"
      cd "%ProgramFiles%\Microsoft Visual Studio\Installer"
) else (
   echo "vswhere tool is not found"
)

set MSBUILD_BIN=
set VS_PATH=

if !VSWHERE! == "true" (
   for /f "usebackq tokens=*" %%i in (`vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do (
      set VS_PATH=%%i
   )
   if exist "!VS_PATH!\MSBuild\14.0\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=!VS_PATH!\MSBuild\14.0\Bin\MSBuild.exe"
   )
   if exist "!VS_PATH!\MSBuild\15.0\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=!VS_PATH!\MSBuild\15.0\Bin\MSBuild.exe"
   )
   if exist "!VS_PATH!\MSBuild\Current\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=!VS_PATH!\MSBuild\Current\Bin\MSBuild.exe"
   )
)

if "!MSBUILD_BIN!" == "" (
   if exist "C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"
      set "MSBUILD_VERSION=14 2015"
   )
   if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\MSBuild\15.0\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\MSBuild\15.0\Bin\MSBuild.exe"
      set "MSBUILD_VERSION=15 2017"
   )
   if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe"
      set "MSBUILD_VERSION=15 2017"
   )
   if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe"
      set "MSBUILD_VERSION=15 2017"
   )
) else (
   if not "!MSBUILD_BIN:2019=!"=="!MSBUILD_BIN!" set "MSBUILD_VERSION=16 2019"
   if not "!MSBUILD_BIN:2017=!"=="!MSBUILD_BIN!" set "MSBUILD_VERSION=15 2017"
   if not "!MSBUILD_BIN:2015=!"=="!MSBUILD_BIN!" set "MSBUILD_VERSION=14 2015"
)

if "!MSBUILD_BIN!" == "" (
   echo Build tools for Visual Studio 2015 / 2017 / 2019 cannot be found. If you use Visual Studio 2017, please download and install build tools from https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017
   GOTO errorHandling
)

set SOLUTION_DIR64=%ROOT_DIR%\inference_engine_build

echo Creating Visual Studio !MSBUILD_VERSION! %PLATFORM% files in %SOLUTION_DIR64%... && ^
if exist "%SOLUTION_DIR64%\CMakeCache.txt" del "%SOLUTION_DIR64%\CMakeCache.txt"
cd "%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\samples" && cmake -E make_directory "%SOLUTION_DIR64%" && cd "%SOLUTION_DIR64%" && cmake -G "Visual Studio !MSBUILD_VERSION!" -A %PLATFORM% "%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\samples"
if ERRORLEVEL 1 GOTO errorHandling

timeout 7

echo.
echo ###############^|^| Build Inference Engine samples using MS Visual Studio (MSBuild.exe) ^|^|###############
echo.
timeout 3
echo !MSBUILD_BIN!" Samples.sln /p:Configuration=Release /t:classification_sample_async /clp:ErrorsOnly /m
"!MSBUILD_BIN!" Samples.sln /p:Configuration=Release /t:classification_sample_async /clp:ErrorsOnly /m

if ERRORLEVEL 1 GOTO errorHandling

timeout 7

:runSample
echo.
echo ###############^|^| Run Inference Engine classification sample ^|^|###############
echo.
timeout 3
copy /Y "%ROOT_DIR%\imagenet.labels" "%ir_dir%\%model_name%.labels"
cd "%SOLUTION_DIR64%\intel64\Release"

echo classification_sample_async.exe -i "%target_image_path%" -m "%ir_dir%\%model_name%.xml" -d !TARGET! !SAMPLE_OPTIONS!
classification_sample_async.exe -i "%target_image_path%" -m "%ir_dir%\%model_name%.xml" -d !TARGET! !SAMPLE_OPTIONS!

if ERRORLEVEL 1 GOTO errorHandling

echo.
echo ###############^|^| Classification demo completed successfully ^|^|###############

timeout 10
cd "%ROOT_DIR%"

goto :eof

:errorHandling
echo Error
cd "%ROOT_DIR%"