:: Copyright (C) 2018-2019 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

@echo off
setlocal enabledelayedexpansion

set TARGET=CPU

set ROOT_DIR=%~dp0
set INTEL_OPENVINO_DIR=C:\Program Files (x86)\IntelSWTools\openvino
set TARGET_PRECISION=FP16

set model_name=opt-efficientnet-b7
set irs_path=%ROOT_DIR%\converted_model\%model_name%
set target_image_path=D:\Datasets\Kaggle\dogs-vs-cats\train\cat\cat.1.jpg

set ir_dir=%ROOT_DIR%\converted_model\%model_name%
set SOLUTION_DIR64=%ROOT_DIR%\inference_engine_build

if exist "%INTEL_OPENVINO_DIR%\bin\setupvars.bat" (
    call "%INTEL_OPENVINO_DIR%\bin\setupvars.bat"
) else (
    echo setupvars.bat is not found, INTEL_OPENVINO_DIR can't be set
    goto error
)

echo INTEL_OPENVINO_DIR is set to %INTEL_OPENVINO_DIR%

echo.
echo ###############^|^| Run Inference Engine classification sample ^|^|###############
echo.
::timeout 3
copy /Y "%ROOT_DIR%\imagenet.labels" "%ir_dir%\%model_name%.labels"
cd "%SOLUTION_DIR64%\intel64\Release"

echo classification_sample_async.exe -i "%target_image_path%" -m "%ir_dir%\%model_name%.xml" -d !TARGET! !SAMPLE_OPTIONS!
classification_sample_async.exe -i "%target_image_path%" -m "%ir_dir%\%model_name%.xml" -d !TARGET! !SAMPLE_OPTIONS!

if ERRORLEVEL 1 GOTO errorHandling

echo.
echo ###############^|^| Classification demo completed successfully ^|^|###############

::timeout 10
cd "%ROOT_DIR%"

goto :eof

:errorHandling
echo Error
cd "%ROOT_DIR%"