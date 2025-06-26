# Installation Guide

This guide will help you install and set up UniversalCompute in your .NET projects.

## 📋 Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|------------|
| **.NET Version** | .NET 9.0 (preview language features enabled) | .NET 9.0 (preview language features enabled) |
| **Operating System** | Windows 10, Linux (Ubuntu 20.04+), macOS 11.0+ | Latest versions |
| **Architecture** | x64 | x64 or ARM64 |
| **Memory** | 4GB RAM | 16GB+ RAM |
| **Storage** | 500MB free space | 2GB+ free space |

### Hardware-Specific Requirements

#### For NVIDIA CUDA Support
- CUDA-compatible GPU (Compute Capability 3.5+)
- CUDA Toolkit 12.0+ installed
- NVIDIA drivers 520.61.05+ (Linux) or 527.41+ (Windows)

#### For Intel Hardware Accelerators
- **Intel AMX**: Intel Xeon Scalable processors (4th Gen+) or Intel Core (12th Gen+)
- **Intel NPU**: Intel Core Ultra processors (Meteor Lake+)
- Intel oneAPI Toolkit (optional, for advanced features)

#### For Apple Neural Engine
- Apple Silicon Mac (M1, M1 Pro, M1 Max, M1 Ultra, M2, M2 Pro, M2 Max, M2 Ultra, M3+)
- macOS 12.0+ (Monterey or later)

## 🚀 Installation Methods

### Method 1: NuGet Package Manager (Recommended)

#### Via Package Manager Console
```powershell
Install-Package UniversalCompute -Version 1.0.0-alpha1
```

#### Via .NET CLI
```bash
dotnet add package UniversalCompute --version 1.0.0-alpha1
```

#### Via PackageReference
Add to your `.csproj` file:
```xml
<PackageReference Include="UniversalCompute" Version="1.0.0-alpha1" />
```

### Method 2: Building from Source

```bash
# Clone the repository
git clone https://github.com/mivertowski/UniversalCompute.git
cd UniversalCompute

# Build the solution
dotnet build Src --configuration=Release

# Run tests (optional)
dotnet test Src/ILGPU.Tests.CPU --configuration=Release --framework=net9.0

# Create local NuGet package
dotnet pack Src/ILGPU --configuration=Release --output ./packages
```

## ⚙️ Project Configuration

### Basic Setup

Create a new console application:
```bash
dotnet new console -n MyUniversalComputeApp
cd MyUniversalComputeApp
dotnet add package UniversalCompute --version 1.0.0-alpha1
```

### Enable Unsafe Code

Add to your `.csproj` file:
```xml
<PropertyGroup>
  <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
</PropertyGroup>
```

### Native AOT Configuration (Optional)

For maximum performance with Native AOT:
```xml
<PropertyGroup>
  <PublishAot>true</PublishAot>
  <InvariantGlobalization>true</InvariantGlobalization>
  <StripSymbols>true</StripSymbols>
</PropertyGroup>
```

## 🧪 Verification

### Basic Functionality Test

Create a simple test to verify installation:

```csharp
using UniversalCompute;
using UniversalCompute.Runtime;

// Create context and auto-detect accelerators
using var context = Context.Create().EnableAllAccelerators();
using var accelerator = context.GetPreferredDevice(preferGPU: true).CreateAccelerator(context);

Console.WriteLine($"Using accelerator: {accelerator.Name}");
Console.WriteLine($"Max threads per group: {accelerator.MaxNumThreadsPerGroup}");
Console.WriteLine($"Max shared memory per group: {accelerator.MaxSharedMemoryPerGroup} bytes");

// Simple kernel test
static void TestKernel(Index1D index, ArrayView<int> data)
{
    data[index] = index * 2;
}

// Allocate memory and run kernel
using var buffer = accelerator.Allocate1D<int>(1024);
var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(TestKernel);

kernel(buffer.Length, buffer.View);
accelerator.Synchronize();

var result = buffer.GetAsArray1D();
Console.WriteLine($"Test successful! First 5 results: [{string.Join(", ", result.Take(5))}]");
```

### Hardware Detection Test

Verify available accelerators:

```csharp
using UniversalCompute;
using UniversalCompute.Runtime.CPU;

var context = Context.Create().EnableAllAccelerators();

Console.WriteLine("Available Accelerators:");
foreach (var device in context.Devices)
{
    Console.WriteLine($"- {device.Name} ({device.AcceleratorType})");
    Console.WriteLine($"  Memory: {device.MemorySize / (1024 * 1024)} MB");
    Console.WriteLine($"  Compute Units: {device.NumMultiProcessors}");
    Console.WriteLine();
}

// Test specific hardware accelerators
try
{
    using var amxAccelerator = context.CreateAMXAccelerator();
    Console.WriteLine("✅ Intel AMX acceleration available");
}
catch
{
    Console.WriteLine("❌ Intel AMX not available");
}

try
{
    using var npuAccelerator = context.CreateNPUAccelerator();
    Console.WriteLine("✅ Intel NPU acceleration available");
}
catch
{
    Console.WriteLine("❌ Intel NPU not available");
}

try
{
    using var aneAccelerator = context.CreateANEAccelerator();
    Console.WriteLine("✅ Apple Neural Engine available");
}
catch
{
    Console.WriteLine("❌ Apple Neural Engine not available");
}
```

## 🔧 IDE Configuration

### Visual Studio 2022

1. **Install required workloads:**
   - .NET desktop development
   - .NET Multi-platform App UI development (for cross-platform)
   - .NET 9.0 preview support

2. **Enable unsafe code globally:**
   - Project Properties → Build → General → Unsafe code ✓

3. **Configure debugging:**
   - Tools → Options → Debugging → General → Enable native code debugging ✓

4. **Enable preview language features:**
   - Project Properties → Build → Advanced → Language version → Preview

### JetBrains Rider

1. **Configure build settings:**
   - File → Settings → Build, Execution, Deployment → Toolset and Build
   - Ensure .NET 9.0 SDK with preview features is selected

2. **Enable unsafe code:**
   - Project settings → Build → Allow unsafe code ✓

3. **Enable preview language features:**
   - Project settings → Build → Language version → Preview

### Visual Studio Code

1. **Install extensions:**
   - C# for Visual Studio Code
   - .NET Install Tool for Extension Authors
   - .NET 9.0 preview support

2. **Configure launch.json:**
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": ".NET Core Launch",
      "type": "coreclr",
      "request": "launch",
      "program": "${workspaceFolder}/bin/Debug/net9.0/MyApp.dll",
      "args": [],
      "cwd": "${workspaceFolder}",
      "console": "internalConsole",
      "stopAtEntry": false
    }
  ]
}
```

## 🐛 Troubleshooting

### Common Issues

#### Issue: "Could not load file or assembly"
**Solution:** Ensure all dependencies are installed and the correct .NET 9.0 runtime with preview features is available.

#### Issue: "No suitable accelerator found"
**Solution:** 
1. Check hardware compatibility
2. Verify drivers are installed
3. Try using CPU accelerator as fallback

#### Issue: Native AOT compilation fails
**Solution:**
1. Ensure `PublishAot` is properly configured for .NET 9.0
2. Check for incompatible dependencies with preview features
3. Use `--self-contained` flag with `--framework=net9.0` during publish

#### Issue: Performance lower than expected
**Solution:**
1. Verify correct accelerator is being used
2. Check memory allocation patterns
3. Enable optimizations in release builds
4. Consider using specialized accelerators (AMX, NPU, ANE)

### Getting Help

- **GitHub Issues**: [Report bugs and request features](https://github.com/mivertowski/UniversalCompute/issues)
- **Discussions**: [Community Q&A](https://github.com/mivertowski/UniversalCompute/discussions)
- **Examples**: [Browse code examples](https://github.com/mivertowski/UniversalCompute/tree/master/Examples)

## ➡️ Next Steps

Now that you have UniversalCompute installed, continue with:

- **[Quick Start Tutorial](Quick-Start-Tutorial)** - Build your first application
- **[Examples Gallery](Examples-Gallery)** - Explore practical examples
- **[API Reference](API-Reference)** - Detailed API documentation
- **[Hardware Accelerators](Hardware-Accelerators)** - Learn about specialized hardware support

---

**Ready to compute? Let's start with the [Quick Start Tutorial](Quick-Start-Tutorial)!**