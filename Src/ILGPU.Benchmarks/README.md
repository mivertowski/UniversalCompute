# ILGPU Benchmarks

Comprehensive benchmark suite for ILGPU Phase 6 implementations including Tensor Core integration, SIMD unification, specialized hardware acceleration, and performance profiling.

## Features Benchmarked

### Tensor Core Operations
- Matrix Multiply-Accumulate (MMA) operations
- Mixed precision computations (FP16, BF16, TF32, INT8)
- Tensor fragment operations
- Direct PTX tensor core bindings

### SIMD Vector Operations
- Platform-specific intrinsics (AVX, SSE, NEON)
- System.Numerics.Vector operations
- Cross-platform vectorization
- Matrix-vector operations

### Hybrid Processing
- CPU/GPU workload distribution
- Adaptive threshold strategies
- Pipeline processing patterns
- Zero-copy memory operations

### Memory Operations
- Unified memory performance
- Memory layout optimization
- Memory pooling and allocation
- Cache efficiency patterns

### Hardware Acceleration
- **Intel NPU (Neural Processing Unit)** - Real hardware via OpenVINO plugins
- **Intel AMX (Advanced Matrix Extensions)** - Native tile operations
- **Apple Neural Engine** - Core ML integration (macOS)
- **Automatic hardware detection** with capability reporting
- **Plugin-based architecture** for specialized accelerators

## Running Benchmarks

### Interactive Mode (Default)
```bash
cd Src/ILGPU.Benchmarks
dotnet run --configuration Release
```

### Hardware Detection & Diagnostics
```bash
# Quick hardware detection and capability check
dotnet run --configuration Release -- --diagnose

# Hardware information display
dotnet run --configuration Release -- --hardware-info
```

### Unattended Mode (CI/CD Integration)
```bash
# Run all benchmarks unattended with GitHub-ready output
dotnet run --configuration Release -- --unattended

# Short form
dotnet run --configuration Release -- -u
```

The unattended mode generates multiple output formats:
- **README_Benchmarks.md** - GitHub README-ready markdown with performance summary
- **benchmark_results.json** - Structured JSON data for programmatic consumption  
- **benchmark_results.csv** - CSV export for data analysis and visualization
- **comprehensive_report.md** - Detailed technical report with system information

### Available Benchmark Suites

1. **Quick Performance Suite** - Fast validation benchmarks
2. **Tensor Core Benchmarks** - Comprehensive tensor operations
3. **SIMD Benchmarks** - Vector processing performance
4. **Hardware Acceleration Benchmarks** - Intel NPU, AMX, Apple Neural Engine
5. **Hybrid Processing Benchmarks** - CPU/GPU coordination
6. **Memory Benchmarks** - Memory operation optimization
7. **Comprehensive Suite** - All benchmarks (may take hours)
8. **Burn-in Test** - Maximum load testing

### Configuration

Benchmark behavior can be configured via `appsettings.json`:

- **Tensor Core Settings**: Compute capability requirements, precision preferences
- **SIMD Settings**: Platform detection, vector sizes, fallback behavior
- **Hardware Acceleration**: NPU/AMX/ANE plugin preferences, detection overrides
- **Memory Settings**: Unified memory, buffer sizes, pooling
- **Burn-in Settings**: Duration, monitoring, safety thresholds

### Hardware Plugin Installation

For real hardware acceleration support:

```bash
# Intel NPU support (when available)
dotnet add package ILGPU.HardwareAccelerators.Intel.NPU

# Apple Neural Engine support (macOS only, when available)
dotnet add package ILGPU.HardwareAccelerators.Apple.NeuralEngine

# Intel AMX support is built-in (no plugin required)
```

See [HARDWARE_ACCELERATION.md](../../HARDWARE_ACCELERATION.md) for detailed hardware setup and plugin development.

### Output

Benchmark results are saved in multiple formats:
- HTML reports for detailed analysis
- CSV exports for data processing
- Markdown summaries for documentation
- JSON exports for programmatic access

## Benchmark Classes

### Core Benchmarks
- `TensorCoreBenchmarks` - Tensor core operations and MMA
- `SimdVectorBenchmarks` - SIMD vector operations
- `MixedPrecisionBenchmarks` - Multi-precision arithmetic
- `BFloat16Benchmarks` - Brain floating point operations

### Platform and Optimization
- `PlatformIntrinsicsBenchmarks` - AVX/SSE/NEON intrinsics
- `MatrixVectorBenchmarks` - Linear algebra operations
- `CpuGpuComparisonBenchmarks` - Cross-platform performance

### Hardware Acceleration
- `IntelNPUBenchmarks` - Neural Processing Unit benchmarks
- `IntelAMXBenchmarks` - Advanced Matrix Extensions benchmarks  
- `AppleNeuralEngineBenchmarks` - Apple Neural Engine benchmarks
- `AIPerformancePrimitivesBenchmarks` - AI acceleration primitives

### Advanced Processing
- `HybridProcessingBenchmarks` - Heterogeneous computing
- `PipelineBenchmarks` - Async processing patterns
- `MemoryBenchmarks` - Memory operation optimization
- `MemoryLayoutBenchmarks` - Data layout strategies
- `UnifiedMemoryBenchmarks` - Zero-copy operations
- `ScalabilityBenchmarks` - Performance scaling analysis

### Stress Testing
- `BurnInTestRunner` - Maximum load and throughput testing

## Hardware Requirements

### Minimum Requirements
- .NET 9.0 Runtime with preview features enabled
- 8GB RAM
- Multi-core CPU with SIMD support

### Recommended for Full Testing
- NVIDIA GPU with Compute Capability 7.0+ (for Tensor Cores)
- 16GB+ RAM
- CUDA 11.0+ drivers
- OpenCL 1.2+ drivers

### Hardware Acceleration Support
- **Intel NPU**: Intel Core Ultra processors (Meteor Lake+) with OpenVINO 2024.0+
- **Intel AMX**: Intel Xeon Sapphire Rapids or 13th/14th gen Core processors
- **Apple Neural Engine**: Apple Silicon (M1/M2/M3) with macOS 11.0+

### Platform Support
- **Windows**: Full support (CUDA, AVX, SSE, Intel NPU, Intel AMX)
- **Linux**: Full support (CUDA, AVX, SSE, Intel AMX)
- **macOS**: CPU + Apple Neural Engine (NEON on ARM64, ANE on Apple Silicon)

## Understanding Results

### Key Metrics
- **Throughput**: Operations per second
- **Latency**: Time per operation
- **Memory Bandwidth**: Data transfer rates
- **Efficiency**: Compute utilization

### Performance Baselines
- Scalar operations provide baseline performance
- SIMD operations show vectorization benefits
- GPU operations demonstrate acceleration potential
- Hybrid strategies optimize workload distribution

### Interpreting Output
- **Ratio columns**: Performance relative to baseline
- **Memory columns**: Allocation and GC pressure
- **Error margins**: Statistical confidence intervals
- **Scaling metrics**: Performance vs problem size

## Troubleshooting

### Common Issues
- **CUDA not available**: Benchmarks fall back to CPU
- **Hardware acceleration not detected**: Install appropriate drivers and plugins
- **Out of memory**: Reduce problem sizes in configuration
- **Driver issues**: Ensure latest GPU drivers installed
- **Platform limitations**: Some features may not be available

### Hardware-Specific Issues
- **Intel NPU not detected**: Verify OpenVINO drivers and plugin installation
- **Intel AMX not available**: Check processor support (13th gen Core+ or Sapphire Rapids+)
- **Apple Neural Engine inactive**: Ensure Core ML framework is available and plugin installed

### Performance Tips
- Close other applications during benchmarking
- Ensure adequate cooling for sustained workloads
- Use Release configuration for accurate measurements
- Run multiple iterations for statistical significance

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Performance Benchmarks

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    - name: Setup .NET
      uses: actions/setup-dotnet@v4
      with:
        dotnet-version: '9.0.x'
        include-prerelease: true
        
    - name: Run Benchmarks
      run: |
        cd Src/ILGPU.Benchmarks
        dotnet run --configuration Release -- --unattended
        
    - name: Upload Results
      uses: actions/upload-artifact@v4
      with:
        name: benchmark-results
        path: |
          Src/ILGPU.Benchmarks/BenchmarkResults/*.md
          Src/ILGPU.Benchmarks/BenchmarkResults/*.json
          Src/ILGPU.Benchmarks/BenchmarkResults/*.csv
        
    - name: Update README
      run: |
        # Copy benchmark results to project README
        cat Src/ILGPU.Benchmarks/BenchmarkResults/README_Benchmarks.md >> README.md
        git add README.md
        git commit -m "Update benchmark results [skip ci]" || exit 0
        git push
```

### Performance Tracking
The benchmark suite automatically:
- Detects system configuration and capabilities
- Measures performance across different hardware configurations
- Generates trend-analysis friendly data formats
- Provides regression detection through baseline comparisons

## Integration with ILGPU Development

This benchmark suite is designed to:
- Validate Phase 6 implementation performance
- Test real hardware acceleration (NPU, AMX, Neural Engine)
- Identify optimization opportunities across diverse hardware
- Regression test performance changes
- Guide hardware-specific optimizations
- Support continuous performance monitoring
- Generate publication-ready performance data

Results can be integrated into CI/CD pipelines for automated performance tracking and regression detection. The output formats are suitable for:
- **Academic Papers:** Comprehensive technical reports with methodology
- **Blog Posts:** GitHub-ready markdown with visual performance summaries  
- **Documentation:** Structured JSON/CSV data for automated documentation
- **Presentations:** Performance insights and key findings sections