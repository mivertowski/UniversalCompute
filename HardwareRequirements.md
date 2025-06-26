# ILGPU Hardware Accelerator Requirements

This document outlines the hardware, software, and driver requirements for ILGPU's AI hardware accelerators.

## Intel Advanced Matrix Extensions (AMX)

### Hardware Requirements
- **CPU**: Intel processors with AMX support
  - Sapphire Rapids (Xeon 4th Gen): Model 0x8F and newer
  - Granite Rapids: Model 0x9A and newer
  - Alder Lake (12th Gen): Model 0x97 and newer consumer processors
  - Raptor Lake (13th Gen): Model 0xB7 and newer
- **Architecture**: x86-64 with AVX-512 support
- **Memory**: Minimum 8GB RAM recommended for tile operations

### OS Requirements
- **Windows**: Windows 11 22H2 or newer with AMX kernel support
- **Linux**: Kernel 5.16+ with AMX support enabled
  - Ubuntu 22.04 LTS or newer
  - RHEL 9.0 or newer
  - SUSE Linux Enterprise 15 SP4 or newer

### Software Dependencies
- **.NET**: .NET 8.0 or newer (required for x86 intrinsics)
- **Compiler**: C# 12.0 language features
- **Runtime**: x86-64 with AMX tiles enabled in OS

### Detection Method
AMX support is detected using CPUID instruction:
- CPUID leaf 7, sub-leaf 0, EDX bit 24: AMX-TILE
- CPUID leaf 7, sub-leaf 0, EDX bit 22: AMX-BF16
- CPUID leaf 7, sub-leaf 0, EDX bit 25: AMX-INT8

### Performance Characteristics
- **Sapphire Rapids**: ~400-500 GB/s memory bandwidth
- **Granite Rapids**: ~600-700 GB/s memory bandwidth
- **Consumer (Alder Lake+)**: ~200-300 GB/s memory bandwidth
- **Matrix Operations**: Up to 2048 TOPS for BF16, 1024 TOPS for FP32

---

## Intel Neural Processing Unit (NPU)

### Hardware Requirements
- **CPU**: Intel processors with integrated NPU
  - Meteor Lake (Core Ultra): NPU 2.0 (~10 TOPS)
  - Lunar Lake (Core Ultra 200V): NPU 3.0 (~40 TOPS)
  - Arrow Lake (Core Ultra 200): NPU 4.0 (~45 TOPS)
- **Memory**: Dedicated NPU memory (1-4 GB depending on generation)

### OS Requirements
- **Windows**: Windows 11 22H2 or newer with NPU drivers
- **Linux**: Ubuntu 22.04+ with Intel NPU kernel modules
  - Requires custom kernel compilation for NPU support
  - Intel NPU driver stack (DPDK-based)

### Software Dependencies
- **OpenVINO Runtime**: 2023.0 or newer
  - Download from Intel's OpenVINO toolkit
  - NPU plugin for OpenVINO required
- **Intel NPU Drivers**:
  - Windows: Automatic via Windows Update
  - Linux: Manual installation from Intel driver package
- **.NET**: .NET 8.0 or newer

### Framework Integration
```bash
# Ubuntu/Debian installation
sudo apt update
sudo apt install intel-opencl-icd intel-level-zero-gpu
wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo apt-key add -
sudo add-apt-repository "deb https://apt.repos.intel.com/openvino/2023 ubuntu20 main"
sudo apt install openvino

# Windows installation
# Download OpenVINO toolkit from Intel Developer Zone
# Install Intel NPU drivers via Device Manager or Intel Driver & Support Assistant
```

### Detection Method
NPU detection uses CPUID to identify Intel processor models:
- Meteor Lake: Family 0x6, Model 0xAA
- Lunar Lake: Family 0x6, Model 0xBD
- Arrow Lake: Family 0x6, Model 0xC6

### Performance Characteristics
- **NPU 2.0 (Meteor Lake)**: 10 TOPS, 50 GB/s bandwidth
- **NPU 3.0 (Lunar Lake)**: 40 TOPS, 100 GB/s bandwidth
- **NPU 4.0 (Arrow Lake)**: 45 TOPS, 150 GB/s bandwidth
- **Data Types**: FP16, BF16, INT8, INT4 (generation dependent)

---

## Apple Neural Engine (ANE)

### Hardware Requirements
- **Apple Silicon**: M1, M2, M3, M4 series or A-series with ANE
  - M1: 15.8 TOPS ANE (1st gen)
  - M2: 15.8 TOPS ANE (2nd gen)
  - M3: 18 TOPS ANE (3rd gen)
  - M4: 38 TOPS ANE (4th gen)
- **Memory**: Unified memory architecture (8GB+ recommended)

### OS Requirements
- **macOS**: macOS 11.0 (Big Sur) or newer
- **iOS**: iOS 14.0 or newer (for mobile devices)
- **Architecture**: ARM64 only

### Software Dependencies
- **Core ML Framework**: Built into macOS/iOS
- **Accelerate Framework**: Built into macOS/iOS for optimized operations
- **Xcode**: 12.0+ for development
- **.NET**: .NET 8.0 with macOS ARM64 support

### Framework Integration
```bash
# macOS installation
# Core ML and Accelerate frameworks are pre-installed
# Verify ANE availability:
system_profiler SPHardwareDataType | grep "Neural Engine"

# For .NET development:
dotnet --list-runtimes | grep "Microsoft.NETCore.App 8.0"
```

### Detection Method
ANE detection uses system APIs:
- `MLIsNeuralEngineAvailable()` from Core ML framework
- Hardware profiler checks for ANE in system configuration
- Runtime detection through Metal Performance Shaders

### Performance Characteristics
- **M1 ANE**: 15.8 TOPS, optimized for CNN and RNN
- **M2 ANE**: 15.8 TOPS, improved efficiency
- **M3 ANE**: 18 TOPS, enhanced transformer support
- **M4 ANE**: 38 TOPS, advanced attention mechanisms
- **Data Types**: FP16, INT8, INT4 with dynamic precision

---

## Common Development Requirements

### Build Environment
```xml
<!-- Add to .csproj file -->
<PropertyGroup>
  <TargetFramework>net8.0</TargetFramework>
  <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
  <RuntimeIdentifiers>win-x64;linux-x64;osx-arm64</RuntimeIdentifiers>
</PropertyGroup>

<ItemGroup>
  <PackageReference Include="System.Runtime.Intrinsics" Version="8.0.0" />
  <PackageReference Include="System.Numerics.Tensors" Version="8.0.0" />
</ItemGroup>
```

### Testing Hardware Support
```csharp
// Test AMX support
bool hasAMX = AMXNative.IsAMXSupported();
var amxCaps = AMXNative.QueryCapabilities();

// Test NPU support  
bool hasNPU = NPUNative.IsNPUSupported();
var npuCaps = NPUNative.QueryCapabilities();

// Test ANE support (macOS only)
bool hasANE = ANENative.IsNeuralEngineAvailable();
var aneGen = ANENative.DetectANEGeneration();
```

### Troubleshooting

#### Common Issues

**AMX Not Detected**:
- Verify CPU model supports AMX using `cpuid` tool
- Check OS has AMX enabled: `cat /proc/cpuinfo | grep amx`
- Ensure kernel is 5.16+ on Linux

**NPU Not Available**:
- Install Intel NPU drivers from Device Manager (Windows)
- Verify OpenVINO installation: `python -c "import openvino as ov; print(ov.Core().available_devices)"`
- Check for NPU in device list: should show "NPU" device

**ANE Not Accessible**:
- Verify on Apple Silicon: `uname -m` should show `arm64`
- Check ANE status: `system_profiler SPHardwareDataType | grep Neural`
- Ensure app has proper entitlements for hardware access

#### Performance Optimization

**AMX Optimization**:
- Use tile-friendly matrix dimensions (multiples of 16x64)
- Prefer BF16 over FP32 for 2x throughput
- Minimize tile configuration changes

**NPU Optimization**:
- Batch operations for better throughput
- Use INT8 quantization when possible
- Optimize models with OpenVINO Model Optimizer

**ANE Optimization**:
- Design models for ANE execution (avoid unsupported operations)
- Use Core ML optimization tools
- Prefer tensor operations over scalar math

### License and Support

This implementation targets real hardware and requires:
- Appropriate hardware as specified above
- Valid software licenses (OpenVINO, Xcode, etc.)
- Compliance with vendor terms of service

For production use, verify hardware accelerator availability and fallback to CPU execution when hardware is not present.