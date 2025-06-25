# Hardware Acceleration Support

ILGPU benchmarks now include support for specialized AI acceleration hardware through a plugin-based architecture. This system allows for real hardware acceleration while keeping the core benchmark project lightweight.

## Overview

The hardware acceleration system provides:

- **Intel NPU (Neural Processing Unit)** support via OpenVINO Runtime
- **Intel AMX (Advanced Matrix Extensions)** support with native intrinsics
- **Apple Neural Engine** support via Core ML (macOS only)
- **Automatic hardware detection** and capability reporting
- **Plugin-based architecture** to avoid heavy dependencies in the main project

## Quick Start

1. **Run hardware detection**:
   ```bash
   cd Src/ILGPU.Benchmarks
   dotnet run --configuration=Release -- --diagnose
   ```

2. **View available hardware**:
   The benchmark application will automatically detect available hardware and show capabilities.

## Architecture

### Core Components

- **`HardwareDetection`**: Detects available specialized hardware
- **`SpecializedAcceleratorFactory`**: Creates hardware-specific accelerators
- **`ISpecializedAccelerator`**: Interface for all hardware accelerators
- **`IHardwarePlugin`**: Interface for loadable hardware plugins

### Plugin System

The system uses a lightweight plugin architecture:

```
ILGPU.Benchmarks (Core)
├── Hardware Detection
├── Plugin Factory
└── Lightweight Fallbacks

Optional Plugins:
├── ILGPU.HardwareAccelerators.Intel.NPU.dll
├── ILGPU.HardwareAccelerators.Intel.AMX.dll
└── ILGPU.HardwareAccelerators.Apple.NeuralEngine.dll
```

## Supported Hardware

### Intel NPU (Neural Processing Unit)

**Requirements:**
- Intel Core Ultra processors (Meteor Lake, Arrow Lake, Lunar Lake)
- OpenVINO Runtime 2024.0 or later (via plugin)

**Capabilities:**
- 10-40 TOPS performance
- INT8/FP16 inference
- Optimized for AI workloads

**Installation:**
```bash
# Install OpenVINO plugin (when available)
dotnet add package ILGPU.HardwareAccelerators.Intel.NPU
```

**Detection:**
```csharp
var hardwareInfo = HardwareDetection.GetHardwareInfo();
if (hardwareInfo.Capabilities.HasFlag(HardwareCapabilities.IntelNPU))
{
    var npuAccelerator = SpecializedAcceleratorFactory.CreateIntelNPUAccelerator();
    // Use real Intel NPU hardware
}
```

### Intel AMX (Advanced Matrix Extensions)

**Requirements:**
- Intel Xeon Sapphire Rapids or later
- Intel 13th/14th generation Core processors or later

**Capabilities:**
- Hardware matrix acceleration
- BF16/INT8 tile operations
- 8 tile registers (TMM0-TMM7)
- 16x64 byte tile size

**Installation:**
No additional plugins required - uses built-in intrinsics support.

**Detection:**
```csharp
if (HardwareDetection.IsIntelAMXAvailable())
{
    var amxAccelerator = SpecializedAcceleratorFactory.CreateIntelAMXAccelerator();
    // Use Intel AMX instructions
}
```

### Apple Neural Engine

**Requirements:**
- Apple Silicon (M1, M2, M3) processors
- macOS 11.0 or later
- Core ML framework (via plugin)

**Capabilities:**
- M1: 11.5 TOPS
- M2: 15.8 TOPS  
- M3: 18 TOPS
- Optimized for neural network inference

**Installation:**
```bash
# Install Core ML plugin (when available)
dotnet add package ILGPU.HardwareAccelerators.Apple.NeuralEngine
```

**Detection:**
```csharp
if (HardwareDetection.IsAppleNeuralEngineAvailable())
{
    var aneAccelerator = SpecializedAcceleratorFactory.CreateAppleNeuralEngineAccelerator();
    // Use Apple Neural Engine
}
```

## Plugin Development

### Creating Hardware Plugins

To create a new hardware accelerator plugin:

1. **Implement the interfaces**:
   ```csharp
   public class MyHardwarePlugin : IHardwarePlugin
   {
       public string Name => "My Hardware Accelerator";
       public bool IsAvailable => DetectHardware();
       
       public string[] GetAvailableDevices() => new[] { "Device1", "Device2" };
       public Dictionary<string, object> GetDeviceProperties() => new();
       public ISpecializedAccelerator? CreateAccelerator() => new MyAccelerator();
   }
   
   public class MyAccelerator : ISpecializedAccelerator
   {
       public string Name => "My Hardware";
       public HardwareCapabilities SupportedOperations => HardwareCapabilities.MyHardware;
       public bool IsAvailable => true;
       
       public async Task<float[]> ExecuteMatrixMultiplyAsync(float[] a, float[] b, int size)
       {
           // Implement hardware-specific matrix multiplication
       }
       
       // Implement other operations...
   }
   ```

2. **Create plugin assembly**:
   - Assembly name: `ILGPU.HardwareAccelerators.{Vendor}.{Hardware}.dll`
   - Place in benchmark output directory
   - Plugin will be automatically discovered and loaded

3. **Add hardware capability**:
   ```csharp
   [Flags]
   public enum HardwareCapabilities
   {
       // ... existing capabilities
       MyHardware = 1 << 6
   }
   ```

### Plugin Installation

Plugins are loaded automatically from the benchmark application directory:

```
ILGPU.Benchmarks.exe
├── ILGPU.HardwareAccelerators.Intel.NPU.dll
├── ILGPU.HardwareAccelerators.Intel.AMX.dll
└── ILGPU.HardwareAccelerators.Apple.NeuralEngine.dll
```

## Benchmarks

### Intel NPU Benchmarks

- **`IntelNPUBenchmarks`**: Real NPU hardware vs ILGPU simulation
- **Operations**: Matrix multiply, convolution, quantized inference, transformer attention
- **Metrics**: Performance comparison, memory usage, power efficiency

### Intel AMX Benchmarks  

- **`IntelAMXBenchmarks`**: AMX tile operations vs standard SIMD
- **Operations**: Matrix operations with tile registers, BF16 computations
- **Metrics**: Throughput, latency, cache efficiency

### Apple Neural Engine Benchmarks

- **`AppleNeuralEngineBenchmarks`**: ANE vs CPU/GPU inference
- **Operations**: Neural network inference, Core ML model execution
- **Metrics**: TOPS utilization, energy efficiency

## Troubleshooting

### Common Issues

1. **Hardware not detected**:
   ```
   ❌ Intel NPU not detected - using ILGPU simulation
   ```
   **Solution**: Verify processor supports NPU and install appropriate drivers.

2. **Plugin not loaded**:
   ```
   ℹ️ Intel NPU detected but no plugin available
   ```
   **Solution**: Install the hardware-specific plugin package.

3. **Permission errors**:
   ```
   Failed to set up high priority (Permission denied)
   ```
   **Solution**: Run with administrator/root privileges for high-priority benchmarks.

### Hardware Detection Debug

```csharp
var hardwareInfo = HardwareDetection.GetHardwareInfo();
Console.WriteLine($"Processor: {hardwareInfo.ProcessorName}");
Console.WriteLine($"Capabilities: {hardwareInfo.Capabilities}");
Console.WriteLine($"Devices: {string.Join(", ", hardwareInfo.AvailableDevices)}");

foreach (var prop in hardwareInfo.Properties)
{
    Console.WriteLine($"{prop.Key}: {prop.Value}");
}
```

## Performance Guidelines

### Best Practices

1. **Use appropriate hardware for workload**:
   - NPU: Neural network inference, quantized models
   - AMX: Large matrix operations, training workloads  
   - ANE: Mobile/edge inference, low-power scenarios

2. **Optimize data layout**:
   - NPU: Use NHWC format for convolutions
   - AMX: Organize data in 16x64 byte tiles
   - ANE: Use FP16 precision when possible

3. **Minimize CPU-accelerator transfers**:
   - Batch operations when possible
   - Keep data resident on accelerator
   - Use async operations for overlap

### Benchmark Configuration

```json
{
  "hardware": {
    "enableNPU": true,
    "enableAMX": true,
    "enableANE": true,
    "preferRealHardware": true
  },
  "benchmarks": {
    "iterations": 100,
    "warmupIterations": 10,
    "measureMemory": true
  }
}
```

## Development Status

### Completed ✅
- Plugin-based architecture design
- Hardware detection system
- Intel AMX lightweight implementation
- Intel NPU plugin interface
- Apple Neural Engine plugin interface
- Benchmark integration
- Documentation

### In Progress 🚧
- Full OpenVINO NPU plugin
- Complete Core ML ANE plugin
- Performance optimization
- Additional hardware support

### Planned 📋
- NVIDIA TensorRT integration
- AMD ROCm support
- Qualcomm Hexagon DSP
- ARM Ethos-N NPU
- Hardware-specific memory management

## Contributing

To contribute hardware acceleration support:

1. Fork the repository
2. Create hardware detection logic
3. Implement `ISpecializedAccelerator` interface  
4. Add benchmarks for the new hardware
5. Create plugin assembly
6. Add documentation
7. Submit pull request

See `CONTRIBUTING.md` for detailed guidelines.

## References

- [Intel OpenVINO Documentation](https://docs.openvino.ai/)
- [Intel AMX Programming Guide](https://www.intel.com/content/www/us/en/developer/articles/guide/advanced-matrix-extensions-programming-guide.html)
- [Apple Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [ILGPU Documentation](https://github.com/m4rs-mt/ILGPU/wiki)

---

For questions or issues, please file an issue on the [ILGPU GitHub repository](https://github.com/m4rs-mt/ILGPU/issues).