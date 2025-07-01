#!/bin/bash

# Bash script to run all tests with code coverage
FRAMEWORK=${1:-"net9.0"}
CONFIGURATION=${2:-"Release"}

echo "=== ILGPU Comprehensive Test Coverage Analysis ==="
echo "Framework: $FRAMEWORK"
echo "Configuration: $CONFIGURATION"
echo ""

# Define test projects to run
TEST_PROJECTS=(
    "ILGPU.Tests"
    "ILGPU.Tests.CPU" 
    "ILGPU.Tests.Velocity"
    "ILGPU.Algorithms.Tests"
    "ILGPU.Algorithms.Tests.CPU"
    "ILGPU.Tests.Hardware"
    "ILGPU.Tests.AI"
    "ILGPU.Tests.UniversalCompute"
    "ILGPU.Analyzers.Tests"
)

# Optional test projects (may not have hardware available)
OPTIONAL_TEST_PROJECTS=(
    "ILGPU.Tests.Cuda"
    "ILGPU.Tests.OpenCL"
    "ILGPU.Algorithms.Tests.Cuda"
    "ILGPU.Algorithms.Tests.OpenCL"
)

# Clean previous results
rm -rf TestResults
mkdir -p TestResults

SUCCESSFUL_PROJECTS=()
FAILED_PROJECTS=()
COVERAGE_FILES=()

# Function to run tests for a project
run_test_project() {
    local project=$1
    local optional=${2:-false}
    
    local project_path="./$project"
    if [ ! -f "$project_path/$project.csproj" ]; then
        if [ "$optional" = true ]; then
            echo "Optional project $project not found, skipping"
            return 0
        else
            echo "❌ Required project $project not found!"
            FAILED_PROJECTS+=("$project")
            return 1
        fi
    fi

    echo "Running tests for $project..."
    
    local test_results_dir="TestResults/$project"
    mkdir -p "$test_results_dir"
    
    # Run tests with coverage
    local command="dotnet test \"$project_path\" --framework $FRAMEWORK --configuration $CONFIGURATION --collect:\"XPlat Code Coverage\" --results-directory \"$test_results_dir\" --logger trx --logger \"console;verbosity=minimal\""
    
    echo "Executing: $command"
    
    if eval $command; then
        echo "✓ $project tests passed"
        SUCCESSFUL_PROJECTS+=("$project")
        
        # Find coverage files
        local coverage_file=$(find "$test_results_dir" -name "coverage.cobertura.xml" | head -1)
        if [ -n "$coverage_file" ]; then
            COVERAGE_FILES+=("$coverage_file")
            echo "  Coverage file: $coverage_file"
        fi
    else
        if [ "$optional" = true ]; then
            echo "⚠ $project tests failed (optional)"
        else
            echo "❌ $project tests failed"
            FAILED_PROJECTS+=("$project")
        fi
    fi
    
    echo ""
}

# Run all required test projects
echo "Running required test projects..."
for project in "${TEST_PROJECTS[@]}"; do
    run_test_project "$project" false
done

# Run optional test projects
echo "Running optional test projects..."
for project in "${OPTIONAL_TEST_PROJECTS[@]}"; do
    run_test_project "$project" true
done

# Generate combined coverage report
echo "=== Test Results Summary ==="
echo "Successful projects (${#SUCCESSFUL_PROJECTS[@]}): ${SUCCESSFUL_PROJECTS[*]}"
echo "Failed projects (${#FAILED_PROJECTS[@]}): ${FAILED_PROJECTS[*]}"
echo "Coverage files found: ${#COVERAGE_FILES[@]}"

# Install reportgenerator if not available
if ! command -v reportgenerator &> /dev/null; then
    echo "Installing ReportGenerator tool..."
    dotnet tool install -g dotnet-reportgenerator-globaltool
fi

# Generate combined coverage report
if [ ${#COVERAGE_FILES[@]} -gt 0 ]; then
    echo "Generating combined coverage report..."
    
    # Join coverage files with semicolon
    IFS=';' coverage_files_string="${COVERAGE_FILES[*]}"
    report_dir="TestResults/CoverageReport"
    
    report_command="reportgenerator \"-reports:$coverage_files_string\" \"-targetdir:$report_dir\" \"-reporttypes:Html;Cobertura;JsonSummary\" \"-assemblyfilters:+ILGPU*;-ILGPU.Tests*;-ILGPU.Benchmarks*\""
    
    echo "Executing: $report_command"
    if eval $report_command; then
        if [ -f "$report_dir/Summary.json" ]; then
            # Parse JSON to get coverage percentages
            line_coverage=$(grep -o '"linecoverage":[0-9.]*' "$report_dir/Summary.json" | cut -d':' -f2)
            branch_coverage=$(grep -o '"branchcoverage":[0-9.]*' "$report_dir/Summary.json" | cut -d':' -f2)
            
            echo ""
            echo "=== COVERAGE RESULTS ==="
            echo "Line Coverage: ${line_coverage}%"
            echo "Branch Coverage: ${branch_coverage}%"
            echo "Coverage Report: $report_dir/index.html"
            
            # Check if we meet the 90% target
            if (( $(echo "$line_coverage >= 90" | bc -l) )); then
                echo "🎯 SUCCESS: 90% coverage target achieved!"
                exit 0
            else
                remaining=$(echo "90 - $line_coverage" | bc -l)
                echo "⚠ Need ${remaining}% more coverage to reach 90% target"
                exit 1
            fi
        else
            echo "❌ Could not find coverage summary"
            exit 1
        fi
    else
        echo "❌ Failed to generate coverage report"
        exit 1
    fi
else
    echo "❌ No coverage files found"
    exit 1
fi