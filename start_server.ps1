# Starts the CARLA server in a small window.
# DirectX 12 is the backend that proved stable on this machine (RTX 3050 Ti,
# driver 616.92): DX11 dies as soon as a camera sensor renders, and Vulkan
# crashes after a few minutes. Low quality also crashes with camera sensors,
# so Epic is the default.
param([string]$Quality = "Epic", [int]$Port = 2000, [string]$Api = "dx12")
$exe = Join-Path $PSScriptRoot "simulator\CARLA_0.9.16\CarlaUE4.exe"
Start-Process $exe -ArgumentList "-$Api","-quality-level=$Quality","-windowed","-ResX=800","-ResY=450","-carla-rpc-port=$Port","-nosound"
Write-Host "CARLA starting on port $Port ($Api, quality: $Quality). Wait until the window shows the town."
