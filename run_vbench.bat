@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat

set VBENCH_JSON=..\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\i2v-bench-info.json
set VBENCH_CROP=..\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\crop

echo Output:  %CD%\results_vbench\videos
echo VBench:  %VBENCH_JSON%
echo Crop:    %VBENCH_CROP%

python run_vbench.py ^
    --vbench_info_json "%VBENCH_JSON%" ^
    --crop_dir "%VBENCH_CROP%" ^
    --output_dir results_vbench/videos ^
    --num_samples 5 ^
    --image_types "indoor,scenery" ^
    --num_ar_steps 50 ^
    --height 480 ^
    --width 832 ^
    --skip_existing True

exit /b %ERRORLEVEL%
