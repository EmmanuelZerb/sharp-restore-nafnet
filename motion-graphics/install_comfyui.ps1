# ============================================================
# ComfyUI + AnimateDiff Installation Script for Windows
# Generates AI motion graphics videos from text prompts
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ComfyUI + AnimateDiff Installer" -ForegroundColor Cyan
Write-Host "  AI Motion Graphics Generator" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$INSTALL_DIR = "D:\ComfyUI"
$COMFYUI_URL = "https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z"

# Check for NVIDIA GPU
Write-Host "[1/7] Checking system requirements..." -ForegroundColor Yellow
$gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" }
if (-not $gpu) {
    Write-Host "WARNING: No NVIDIA GPU detected. ComfyUI works best with NVIDIA GPUs." -ForegroundColor Red
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") { exit 1 }
}
else {
    Write-Host "  Found: $($gpu.Name)" -ForegroundColor Green
}

# Check for 7-Zip
$7zipPath = "C:\Program Files\7-Zip\7z.exe"
if (-not (Test-Path $7zipPath)) {
    Write-Host "[2/7] Installing 7-Zip..." -ForegroundColor Yellow
    $7zipInstaller = "$env:TEMP\7z-installer.exe"
    Invoke-WebRequest -Uri "https://www.7-zip.org/a/7z2301-x64.exe" -OutFile $7zipInstaller
    Start-Process -FilePath $7zipInstaller -Args "/S" -Wait
    Remove-Item $7zipInstaller
    Write-Host "  7-Zip installed" -ForegroundColor Green
}
else {
    Write-Host "[2/7] 7-Zip already installed" -ForegroundColor Green
}

# Create installation directory
Write-Host "[3/7] Creating installation directory..." -ForegroundColor Yellow
if (-not (Test-Path $INSTALL_DIR)) {
    New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null
}
Set-Location $INSTALL_DIR
Write-Host "  Directory: $INSTALL_DIR" -ForegroundColor Green

# Download ComfyUI
$comfyuiArchive = "$INSTALL_DIR\ComfyUI_portable.7z"
if (-not (Test-Path "$INSTALL_DIR\ComfyUI_windows_portable")) {
    Write-Host "[4/7] Downloading ComfyUI (~1.5GB)..." -ForegroundColor Yellow
    Write-Host "  This may take a few minutes..." -ForegroundColor Gray

    # Use BITS for better download handling
    Start-BitsTransfer -Source $COMFYUI_URL -Destination $comfyuiArchive -DisplayName "Downloading ComfyUI"

    Write-Host "  Extracting..." -ForegroundColor Yellow
    & $7zipPath x $comfyuiArchive -o"$INSTALL_DIR" -y | Out-Null
    Remove-Item $comfyuiArchive
    Write-Host "  ComfyUI extracted" -ForegroundColor Green
}
else {
    Write-Host "[4/7] ComfyUI already downloaded" -ForegroundColor Green
}

$COMFYUI_DIR = "$INSTALL_DIR\ComfyUI_windows_portable"

# Install ComfyUI Manager
Write-Host "[5/7] Installing ComfyUI Manager..." -ForegroundColor Yellow
$managerDir = "$COMFYUI_DIR\ComfyUI\custom_nodes\ComfyUI-Manager"
if (-not (Test-Path $managerDir)) {
    Set-Location "$COMFYUI_DIR\ComfyUI\custom_nodes"
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    Write-Host "  Manager installed" -ForegroundColor Green
}
else {
    Write-Host "  Manager already installed" -ForegroundColor Green
}

# Install AnimateDiff Evolved
Write-Host "[6/7] Installing AnimateDiff Evolved..." -ForegroundColor Yellow
$animatediffDir = "$COMFYUI_DIR\ComfyUI\custom_nodes\ComfyUI-AnimateDiff-Evolved"
if (-not (Test-Path $animatediffDir)) {
    Set-Location "$COMFYUI_DIR\ComfyUI\custom_nodes"
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git
    Write-Host "  AnimateDiff installed" -ForegroundColor Green
}
else {
    Write-Host "  AnimateDiff already installed" -ForegroundColor Green
}

# Install Video Helper Suite
$videoHelperDir = "$COMFYUI_DIR\ComfyUI\custom_nodes\ComfyUI-VideoHelperSuite"
if (-not (Test-Path $videoHelperDir)) {
    Set-Location "$COMFYUI_DIR\ComfyUI\custom_nodes"
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    Write-Host "  Video Helper Suite installed" -ForegroundColor Green
}

# Download required models
Write-Host "[7/7] Downloading AI models..." -ForegroundColor Yellow

# Create model directories
$modelsDir = "$COMFYUI_DIR\ComfyUI\models"
$checkpointsDir = "$modelsDir\checkpoints"
$animatediffModelsDir = "$animatediffDir\models"
$vaeDir = "$modelsDir\vae"

New-Item -ItemType Directory -Path $checkpointsDir -Force | Out-Null
New-Item -ItemType Directory -Path $animatediffModelsDir -Force | Out-Null
New-Item -ItemType Directory -Path $vaeDir -Force | Out-Null

# Download AnimateDiff motion model
$motionModel = "$animatediffModelsDir\v3_sd15_mm.ckpt"
if (-not (Test-Path $motionModel)) {
    Write-Host "  Downloading AnimateDiff v3 motion model..." -ForegroundColor Gray
    $motionModelUrl = "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt"
    Start-BitsTransfer -Source $motionModelUrl -Destination $motionModel -DisplayName "AnimateDiff Motion Model"
}

# Download Dreamshaper checkpoint (good for stylized videos)
$checkpoint = "$checkpointsDir\dreamshaper_8.safetensors"
if (-not (Test-Path $checkpoint)) {
    Write-Host "  Downloading Dreamshaper 8 checkpoint (~2GB)..." -ForegroundColor Gray
    $checkpointUrl = "https://civitai.com/api/download/models/128713"
    Start-BitsTransfer -Source $checkpointUrl -Destination $checkpoint -DisplayName "Dreamshaper 8"
}

# Download VAE
$vae = "$vaeDir\vae-ft-mse-840000-ema-pruned.safetensors"
if (-not (Test-Path $vae)) {
    Write-Host "  Downloading VAE..." -ForegroundColor Gray
    $vaeUrl = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
    Start-BitsTransfer -Source $vaeUrl -Destination $vae -DisplayName "VAE"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start ComfyUI:" -ForegroundColor Cyan
Write-Host "  1. Navigate to: $COMFYUI_DIR" -ForegroundColor White
Write-Host "  2. Run: run_nvidia_gpu.bat" -ForegroundColor White
Write-Host "  3. Open: http://127.0.0.1:8188" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  - Load the workflow: app_promo_workflow.json" -ForegroundColor White
Write-Host "  - Enter your prompt describing the video" -ForegroundColor White
Write-Host "  - Click 'Queue Prompt' to generate" -ForegroundColor White
Write-Host ""

# Create desktop shortcut
$shortcutPath = "$env:USERPROFILE\Desktop\ComfyUI.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "$COMFYUI_DIR\run_nvidia_gpu.bat"
$shortcut.WorkingDirectory = $COMFYUI_DIR
$shortcut.Save()
Write-Host "Desktop shortcut created!" -ForegroundColor Green
