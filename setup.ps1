# Recreates everything that is not stored in git: the conda env, the detector
# weights and the CARLA simulator. Nothing is left in pip's cache and the
# simulator zip is deleted after extraction.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

conda create -y -n carla python=3.12
conda run -n carla python -m pip install --no-cache-dir -r "$root\requirements.txt"

$model = "$root\models\object_detection_yolox_2022nov.onnx"
if (-not (Test-Path $model)) {
    New-Item -ItemType Directory -Force "$root\models" | Out-Null
    curl.exe -L -o $model "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx"
}

$sim = "$root\simulator\CARLA_0.9.16"
if (-not (Test-Path "$sim\CarlaUE4.exe")) {
    New-Item -ItemType Directory -Force $sim | Out-Null
    $zip = "$root\simulator\CARLA_0.9.16.zip"
    curl.exe -L -C - -o $zip "https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip"
    & "$env:SystemRoot\System32\tar.exe" -xf $zip -C $sim
    Remove-Item $zip
    # 4 GB GPU settings: light default map, smaller texture pool
    $ini = "$sim\CarlaUE4\Config\DefaultEngine.ini"
    Copy-Item $ini "$ini.orig"
    (Get-Content $ini) -replace '^(GameDefaultMap|ServerDefaultMap|TransitionMap)=.*', '$1=/Game/Carla/Maps/Town02.Town02' `
        -replace '^r\.Streaming\.PoolSize=.*', 'r.Streaming.PoolSize=1200' | Set-Content $ini
}
Write-Host "Done. Start the simulator with .\start_server.ps1, then: conda activate carla; python main.py"
