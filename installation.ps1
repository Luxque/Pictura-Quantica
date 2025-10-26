<#
installation.ps1
PowerShell installer for Pictura Quantica (Windows).
Creates a venv, activates it, installs dependencies, and optionally runs the app.
Run in PowerShell from the repository root:
  .\installation.ps1
To only setup without launching the app:
  .\installation.ps1 -NoRun
#>

param(
    [switch]$NoRun
)

function Abort($msg, $code=1) {
    Write-Error $msg
    exit $code
}

Write-Host "Pictura Quantica installer (PowerShell)" -ForegroundColor Cyan

# Ensure python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Abort "Python executable not found in PATH. Please install Python 3.10+ and ensure 'python' is on PATH."
}

# Create virtual environment
Write-Host "Creating virtual environment 'venv'..."
try {
    python -m venv venv 2>&1 | Write-Verbose
} catch {
    Abort "Failed to create virtual environment: $_"
}

if (-not (Test-Path -Path .\venv)) {
    Abort "Virtual environment directory '.\\venv' was not created."
}

# Allow running the activation script in this process
try {
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force -ErrorAction Stop
} catch {
    Write-Warning "Could not set execution policy for the process. Activation may still work if your policy allows running scripts."
}

$activate = Join-Path -Path (Get-Location) -ChildPath "venv\Scripts\Activate.ps1"
if (-not (Test-Path -Path $activate)) {
    Abort "Activation script not found at: $activate"
}

Write-Host "Activating virtual environment..."
try {
    & $activate
} catch {
    Abort "Failed to activate virtual environment: $_"
}

# Upgrade pip inside venv
Write-Host "Upgrading pip..."
try {
    python -m pip install --upgrade pip
} catch {
    Write-Warning "pip upgrade failed: $_"
}

# Install dependencies
$packages = @(
    'numpy',
    'pillow',
    'opencv-python',
    'scipy',
    'scikit-learn',
    'qiskit',
    'qiskit-machine-learning',
    'matplotlib',
    'pylatexenc',
    'seaborn',
    'joblib',
    'PyQt6'
)

Write-Host "Installing Python packages into venv..." -ForegroundColor Green
try {
    # Use python -m pip to ensure we call the venv's pip
    python -m pip install -U $packages
} catch {
    Write-Warning "Package installation returned an error: $_"
}

# Optional: run the application
if (-not $NoRun) {
    $main = Join-Path -Path (Get-Location) -ChildPath "source\main.py"
    if (Test-Path -Path $main) {
        Write-Host "Launching the application (source/main.py)..." -ForegroundColor Cyan
        try {
            Push-Location -Path .\source
            python main.py
            Pop-Location
        } catch {
            Write-Warning "Failed to run the application: $_"
            if (Get-Location -ErrorAction SilentlyContinue) { Pop-Location }
        }
    } else {
        Write-Warning "Could not find 'source/main.py'. Skipping launch."
    }
} else {
    Write-Host "Setup complete. Skipping app launch because -NoRun was specified." -ForegroundColor Green
}

Write-Host "Done." -ForegroundColor Cyan
