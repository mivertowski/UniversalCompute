# PowerShell script to run all tests with code coverage
param(
    [string]$Framework = "net9.0",
    [string]$Configuration = "Release"
)

Write-Host "=== ILGPU Comprehensive Test Coverage Analysis ===" -ForegroundColor Green
Write-Host "Framework: $Framework" -ForegroundColor Yellow
Write-Host "Configuration: $Configuration" -ForegroundColor Yellow
Write-Host ""

# Define test projects to run
$TestProjects = @(
    "ILGPU.Tests",
    "ILGPU.Tests.CPU", 
    "ILGPU.Tests.Velocity",
    "ILGPU.Algorithms.Tests",
    "ILGPU.Algorithms.Tests.CPU",
    "ILGPU.Tests.Hardware",
    "ILGPU.Tests.AI",
    "ILGPU.Tests.UniversalCompute",
    "ILGPU.Analyzers.Tests"
)

# Optional test projects (may not have hardware available)
$OptionalTestProjects = @(
    "ILGPU.Tests.Cuda",
    "ILGPU.Tests.OpenCL", 
    "ILGPU.Algorithms.Tests.Cuda",
    "ILGPU.Algorithms.Tests.OpenCL"
)

# Clean previous results
if (Test-Path "TestResults") {
    Remove-Item -Recurse -Force TestResults
}
New-Item -ItemType Directory -Force -Path TestResults | Out-Null

$SuccessfulProjects = @()
$FailedProjects = @()
$CoverageFiles = @()

# Function to run tests for a project
function RunTestProject {
    param([string]$Project, [bool]$Optional = $false)
    
    $ProjectPath = "./$Project"
    if (-not (Test-Path "$ProjectPath/$Project.csproj")) {
        if ($Optional) {
            Write-Host "Optional project $Project not found, skipping" -ForegroundColor Yellow
            return
        } else {
            Write-Host "Required project $Project not found!" -ForegroundColor Red
            $script:FailedProjects += $Project
            return
        }
    }

    Write-Host "Running tests for $Project..." -ForegroundColor Cyan
    
    $TestResultsDir = "TestResults/$Project"
    New-Item -ItemType Directory -Force -Path $TestResultsDir | Out-Null
    
    # Run tests with coverage
    $Command = "dotnet test `"$ProjectPath`" --framework $Framework --configuration $Configuration --collect:`"XPlat Code Coverage`" --results-directory `"$TestResultsDir`" --logger trx --logger `"console;verbosity=minimal`""
    
    Write-Host "Executing: $Command" -ForegroundColor DarkGray
    
    try {
        $Result = Invoke-Expression $Command
        $ExitCode = $LASTEXITCODE
        
        if ($ExitCode -eq 0) {
            Write-Host "✓ $Project tests passed" -ForegroundColor Green
            $script:SuccessfulProjects += $Project
            
            # Find coverage files
            $CoverageFile = Get-ChildItem -Path $TestResultsDir -Recurse -Filter "coverage.cobertura.xml" | Select-Object -First 1
            if ($CoverageFile) {
                $script:CoverageFiles += $CoverageFile.FullName
                Write-Host "  Coverage file: $($CoverageFile.FullName)" -ForegroundColor DarkGreen
            }
        } else {
            if ($Optional) {
                Write-Host "⚠ $Project tests failed (optional)" -ForegroundColor Yellow
            } else {
                Write-Host "✗ $Project tests failed" -ForegroundColor Red
                $script:FailedProjects += $Project
            }
        }
    } catch {
        if ($Optional) {
            Write-Host "⚠ $Project tests failed with exception (optional): $($_.Exception.Message)" -ForegroundColor Yellow
        } else {
            Write-Host "✗ $Project tests failed with exception: $($_.Exception.Message)" -ForegroundColor Red
            $script:FailedProjects += $Project
        }
    }
    
    Write-Host ""
}

# Run all required test projects
Write-Host "Running required test projects..." -ForegroundColor Magenta
foreach ($Project in $TestProjects) {
    RunTestProject -Project $Project -Optional $false
}

# Run optional test projects
Write-Host "Running optional test projects..." -ForegroundColor Magenta
foreach ($Project in $OptionalTestProjects) {
    RunTestProject -Project $Project -Optional $true
}

# Generate combined coverage report
Write-Host "=== Test Results Summary ===" -ForegroundColor Green
Write-Host "Successful projects ($($SuccessfulProjects.Count)): $($SuccessfulProjects -join ', ')" -ForegroundColor Green
Write-Host "Failed projects ($($FailedProjects.Count)): $($FailedProjects -join ', ')" -ForegroundColor Red
Write-Host "Coverage files found: $($CoverageFiles.Count)" -ForegroundColor Cyan

# Install reportgenerator if not available
$ReportGeneratorExists = Get-Command "reportgenerator" -ErrorAction SilentlyContinue
if (-not $ReportGeneratorExists) {
    Write-Host "Installing ReportGenerator tool..." -ForegroundColor Yellow
    dotnet tool install -g dotnet-reportgenerator-globaltool
}

# Generate combined coverage report
if ($CoverageFiles.Count -gt 0) {
    Write-Host "Generating combined coverage report..." -ForegroundColor Cyan
    
    $CoverageFilesString = $CoverageFiles -join ";"
    $ReportDir = "TestResults/CoverageReport"
    
    $ReportCommand = "reportgenerator `"-reports:$CoverageFilesString`" `"-targetdir:$ReportDir`" `"-reporttypes:Html;Cobertura;JsonSummary`" `"-assemblyfilters:+ILGPU*;-ILGPU.Tests*;-ILGPU.Benchmarks*`""
    
    Write-Host "Executing: $ReportCommand" -ForegroundColor DarkGray
    try {
        Invoke-Expression $ReportCommand
        
        if (Test-Path "$ReportDir/Summary.json") {
            $Summary = Get-Content "$ReportDir/Summary.json" | ConvertFrom-Json
            $LineCoverage = [math]::Round($Summary.summary.linecoverage, 2)
            $BranchCoverage = [math]::Round($Summary.summary.branchcoverage, 2)
            
            Write-Host ""
            Write-Host "=== COVERAGE RESULTS ===" -ForegroundColor Green
            Write-Host "Line Coverage: $LineCoverage%" -ForegroundColor $(if ($LineCoverage -ge 90) { "Green" } elseif ($LineCoverage -ge 70) { "Yellow" } else { "Red" })
            Write-Host "Branch Coverage: $BranchCoverage%" -ForegroundColor $(if ($BranchCoverage -ge 80) { "Green" } elseif ($BranchCoverage -ge 60) { "Yellow" } else { "Red" })
            Write-Host "Coverage Report: $ReportDir/index.html" -ForegroundColor Cyan
            
            # Check if we meet the 90% target
            if ($LineCoverage -ge 90) {
                Write-Host "🎯 SUCCESS: 90% coverage target achieved!" -ForegroundColor Green
                exit 0
            } else {
                $Remaining = 90 - $LineCoverage
                Write-Host "⚠ Need $Remaining% more coverage to reach 90% target" -ForegroundColor Yellow
                exit 1
            }
        } else {
            Write-Host "❌ Could not find coverage summary" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "❌ Failed to generate coverage report: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ No coverage files found" -ForegroundColor Red
    exit 1
}