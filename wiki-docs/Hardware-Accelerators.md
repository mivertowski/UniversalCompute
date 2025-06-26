# Hardware Accelerators

Comprehensive guide to hardware accelerator support in UniversalCompute, covering traditional GPU computing and modern specialized accelerators.

## 🎯 Overview

UniversalCompute provides unified access to diverse hardware accelerators through a single API, enabling developers to harness the full power of modern heterogeneous computing systems.

### Supported Accelerator Types

| Accelerator Type | Description | Use Cases | Performance Range |
|------------------|-------------|-----------|-------------------|
| **CPU** | Multi-core processors | General computing, debugging | 10-100 GFLOPS |
| **GPU (CUDA)** | NVIDIA graphics cards | Parallel computing, AI/ML | 100-1000+ GFLOPS |
| **GPU (OpenCL)** | Cross-platform GPUs | Portable computing | 50-800 GFLOPS |
| **Intel AMX** | Advanced Matrix Extensions | Matrix operations, AI | 200-400 GFLOPS |
| **Intel NPU** | Neural Processing Unit | AI inference | 10-50 TOPS |
| **Apple Neural Engine** | Apple Silicon AI accelerator | ML inference on Mac | 15-35 TOPS |
| **Velocity SIMD** | CPU vectorization | High-throughput CPU | 20-80 GFLOPS |

---

## 🖥️ CPU Accelerators

### Standard CPU Acceleration

CPU accelerators provide reliable, debuggable computation that works on any system.

```csharp
using UniversalCompute;
using UniversalCompute.Runtime.CPU;

// Create CPU context
using var context = Context.Create().CPU();
using var cpuAccelerator = context.CreateCPUAccelerator();

Console.WriteLine($"CPU: {cpuAccelerator.Name}");
Console.WriteLine($"Max threads: {cpuAccelerator.MaxNumThreadsPerGroup}");
Console.WriteLine($"Cores: {cpuAccelerator.NumMultiProcessors}");
```

#### CPU Accelerator Modes

```csharp
// Auto mode (recommended) - automatically selects optimal threading
using var autoAccelerator = context.CreateCPUAccelerator(0, CPUAcceleratorMode.Auto);

// Sequential mode - single-threaded execution (debugging)
using var seqAccelerator = context.CreateCPUAccelerator(0, CPUAcceleratorMode.Sequential);

// Parallel mode - explicit multi-threading
using var parallelAccelerator = context.CreateCPUAccelerator(0, CPUAcceleratorMode.Parallel);
```

### Velocity SIMD Acceleration

Velocity accelerators use CPU SIMD instructions for vectorized operations.

```csharp
// Enable Velocity accelerators
using var context = Context.Create()
    .CPU()                    // Standard CPU
    .EnableVelocity()        // SIMD-accelerated CPU
    .ToContext();

// Get Velocity accelerator
var velocityDevice = context.GetDevices<VelocityDevice>().FirstOrDefault();
if (velocityDevice != null)
{
    using var velocityAccelerator = velocityDevice.CreateAccelerator(context);
    
    Console.WriteLine($"Velocity: {velocityAccelerator.Name}");
    Console.WriteLine($"SIMD width: {velocityAccelerator.WarpSize}");
}
```

#### SIMD-Optimized Kernels

```csharp
// Kernel optimized for SIMD execution
static void SIMDVectorAddKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
{
    // This kernel will be automatically vectorized by Velocity
    result[index] = a[index] + b[index] * 2.0f + Math.Sin(a[index]);
}

// Load kernel on Velocity accelerator
var kernel = velocityAccelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(SIMDVectorAddKernel);
```

---

## 🎮 GPU Accelerators

### NVIDIA CUDA Support

Full CUDA support with PTX backend for maximum performance on NVIDIA GPUs.

```csharp
using UniversalCompute.Runtime.Cuda;

// Create CUDA context
using var context = Context.Create().CUDA();

// Get CUDA devices
var cudaDevices = context.GetCudaDevices();
foreach (var device in cudaDevices)
{
    Console.WriteLine($"CUDA Device: {device.Name}");
    Console.WriteLine($"Compute Capability: {device.ComputeCapability}");
    Console.WriteLine($"Memory: {device.MemorySize / (1024 * 1024)} MB");
    Console.WriteLine($"SMs: {device.NumMultiProcessors}");
    Console.WriteLine();
}

// Create CUDA accelerator
using var cudaAccelerator = context.CreateCudaAccelerator(0);
```

#### CUDA-Specific Features

```csharp
// Access CUDA-specific properties
var cudaAccel = accelerator as CudaAccelerator;
if (cudaAccel != null)
{
    Console.WriteLine($"Warp size: {cudaAccel.WarpSize}");
    Console.WriteLine($"Max blocks per SM: {cudaAccel.MaxNumThreadsPerMultiprocessor}");
    Console.WriteLine($"Shared memory per block: {cudaAccel.MaxSharedMemoryPerGroup} bytes");
    
    // Get detailed device information
    var deviceInfo = cudaAccel.GetDeviceInfo();
    Console.WriteLine($"Texture alignment: {deviceInfo.TextureAlignment}");
    Console.WriteLine($"Global memory bandwidth: {deviceInfo.MemoryBandwidth} GB/s");
}
```

#### CUDA Memory Management

```csharp
// Allocate different types of CUDA memory
using var globalMem = cudaAccelerator.Allocate1D<float>(1024 * 1024);      // Global memory
using var sharedMem = cudaAccelerator.AllocateShared<float>(1024);         // Shared memory
using var constantMem = cudaAccelerator.AllocateConstant<float>(256);      // Constant memory

// Page-locked memory for faster transfers
using var pinnedMem = cudaAccelerator.AllocatePageLocked<float>(1024);
```

### OpenCL Support

Cross-platform GPU acceleration supporting NVIDIA, AMD, Intel, and other vendors.

```csharp
using UniversalCompute.Runtime.OpenCL;

// Create OpenCL context
using var context = Context.Create().OpenCL();

// Get OpenCL devices
var openCLDevices = context.GetOpenCLDevices();
foreach (var device in openCLDevices)
{
    Console.WriteLine($"OpenCL Device: {device.Name}");
    Console.WriteLine($"Vendor: {device.Vendor}");
    Console.WriteLine($"Type: {device.DeviceType}");
    Console.WriteLine($"Compute Units: {device.ComputeUnits}");
    Console.WriteLine($"Max Work Group Size: {device.MaxWorkGroupSize}");
    Console.WriteLine();
}

// Create OpenCL accelerator
using var openCLAccelerator = context.CreateOpenCLAccelerator(0);
```

#### OpenCL Device Selection

```csharp
// Select specific device types
var gpuDevices = context.GetOpenCLDevices().Where(d => d.DeviceType == OpenCLDeviceType.GPU);
var cpuDevices = context.GetOpenCLDevices().Where(d => d.DeviceType == OpenCLDeviceType.CPU);

// Select by vendor
var nvidiaDevices = context.GetOpenCLDevices().Where(d => d.Vendor.Contains("NVIDIA"));
var amdDevices = context.GetOpenCLDevices().Where(d => d.Vendor.Contains("AMD"));
var intelDevices = context.GetOpenCLDevices().Where(d => d.Vendor.Contains("Intel"));
```

---

## 🧠 Intel Hardware Accelerators

### Intel AMX (Advanced Matrix Extensions)

High-performance matrix operations using Intel's specialized hardware.

```csharp
using UniversalCompute.Intel.AMX;

// Check AMX availability
if (AMXCapabilities.IsAMXSupported())
{
    var capabilities = AMXCapabilities.Query();
    Console.WriteLine($"AMX Supported: {capabilities.IsSupported}");
    Console.WriteLine($"Max Tiles: {capabilities.MaxTiles}");
    Console.WriteLine($"Tile Size: {capabilities.MaxTileRows}x{capabilities.MaxTileColumns}");
    Console.WriteLine($"BF16 Support: {capabilities.SupportsBF16}");
    Console.WriteLine($"INT8 Support: {capabilities.SupportsInt8}");
    Console.WriteLine($"Estimated Bandwidth: {capabilities.EstimatedBandwidthGBps:F1} GB/s");
    
    // Create AMX accelerator
    using var context = Context.Create().CPU();
    using var amxAccelerator = context.CreateAMXAccelerator();
    
    // AMX-optimized matrix multiplication
    await DemoAMXMatrixMultiplication(amxAccelerator, capabilities);
}
else
{
    Console.WriteLine("Intel AMX not supported on this hardware");
}
```

#### AMX Matrix Multiplication Example

```csharp
static async Task DemoAMXMatrixMultiplication(Accelerator amxAccelerator, AMXCapabilities capabilities)
{
    const int matrixSize = 1024;
    
    // Allocate matrices
    using var matrixA = amxAccelerator.Allocate2D<float>(new LongIndex2D(matrixSize, matrixSize));
    using var matrixB = amxAccelerator.Allocate2D<float>(new LongIndex2D(matrixSize, matrixSize));
    using var matrixC = amxAccelerator.Allocate2D<float>(new LongIndex2D(matrixSize, matrixSize));
    
    // Initialize matrices with test data
    var dataA = new float[matrixSize, matrixSize];
    var dataB = new float[matrixSize, matrixSize];
    var random = new Random(42);
    
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            dataA[i, j] = random.NextSingle();
            dataB[i, j] = random.NextSingle();
        }
    }
    
    matrixA.CopyFromCPU(dataA);
    matrixB.CopyFromCPU(dataB);
    
    // AMX-optimized matrix multiplication kernel
    var kernel = amxAccelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(AMXMatMulKernel);
    
    // Execute with timing
    var stopwatch = Stopwatch.StartNew();
    kernel(matrixC.Extent, matrixA.View, matrixB.View, matrixC.View);
    amxAccelerator.Synchronize();
    stopwatch.Stop();
    
    // Calculate performance
    var gflops = (2.0 * matrixSize * matrixSize * matrixSize) / (stopwatch.ElapsedMilliseconds / 1000.0) / 1e9;
    Console.WriteLine($"AMX Matrix Multiplication Performance: {gflops:F2} GFLOPS");
    
    // Get optimal tile configuration
    var (tileM, tileN, tileK) = capabilities.GetOptimalTileSize(matrixSize, matrixSize, matrixSize, AMXDataType.Float32);
    Console.WriteLine($"Optimal tile size: {tileM}x{tileN}x{tileK}");
}

// AMX-optimized kernel using tile operations
static void AMXMatMulKernel(Index2D index, ArrayView2D<float, Stride2D.DenseX> a, ArrayView2D<float, Stride2D.DenseX> b, ArrayView2D<float, Stride2D.DenseX> c)
{
    var row = index.X;
    var col = index.Y;
    
    if (row >= a.Extent.X || col >= b.Extent.Y)
        return;
    
    float sum = 0;
    
    // This will be optimized to use AMX tile operations
    for (int k = 0; k < a.Extent.Y; k++)
    {
        sum += a[row, k] * b[k, col];
    }
    
    c[row, col] = sum;
}
```

### Intel NPU (Neural Processing Unit)

Dedicated AI inference acceleration on Intel Core Ultra processors.

```csharp
using UniversalCompute.Intel.NPU;

// Check NPU availability
if (NPUCapabilities.IsNPUSupported())
{
    var capabilities = NPUCapabilities.Query();
    Console.WriteLine($"NPU Device: {capabilities.DeviceName}");
    Console.WriteLine($"Max Batch Size: {capabilities.MaxBatchSize}");
    Console.WriteLine($"Max Memory: {capabilities.MaxMemorySize / (1024 * 1024)} MB");
    Console.WriteLine($"Peak TOPS: {capabilities.PeakTOPS:F1}");
    Console.WriteLine($"Supported Precisions: {string.Join(", ", capabilities.SupportedPrecisions)}");
    
    // Create NPU accelerator
    using var context = Context.Create().CPU();
    using var npuAccelerator = context.CreateNPUAccelerator();
    
    // Demo NPU inference
    await DemoNPUInference(npuAccelerator);
}
else
{
    Console.WriteLine("Intel NPU not supported on this hardware");
}
```

#### NPU Inference Example

```csharp
static async Task DemoNPUInference(Accelerator npuAccelerator)
{
    // Create NPU inference engine
    using var inferenceEngine = new NPUInferenceEngine(npuAccelerator);
    
    // Load a pre-trained model (example path)
    // Note: In a real application, you would load an actual ONNX or OpenVINO model
    try
    {
        // inferenceEngine.LoadModel("path/to/model.onnx");
        Console.WriteLine("Model loading simulated (actual model file needed for real inference)");
        
        // Prepare input data
        const int inputSize = 224 * 224 * 3; // Example: ImageNet input
        using var inputBuffer = npuAccelerator.Allocate1D<float>(inputSize);
        
        // Initialize with sample data
        var inputData = new float[inputSize];
        var random = new Random(42);
        for (int i = 0; i < inputSize; i++)
        {
            inputData[i] = random.NextSingle();
        }
        
        inputBuffer.CopyFromCPU(inputData);
        
        // Run inference (simulated)
        var stopwatch = Stopwatch.StartNew();
        // var result = await inferenceEngine.RunInferenceAsync(inputBuffer.View);
        stopwatch.Stop();
        
        Console.WriteLine($"NPU Inference Time: {stopwatch.ElapsedMilliseconds} ms");
        Console.WriteLine("NPU inference demonstration completed (model loading required for full functionality)");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"NPU inference demo: {ex.Message}");
    }
}
```

---

## 🍎 Apple Neural Engine

Hardware-accelerated AI inference on Apple Silicon Macs.

```csharp
using UniversalCompute.Apple.NeuralEngine;

// Check ANE availability (only on Apple Silicon)
if (ANECapabilities.IsANESupported())
{
    var capabilities = ANECapabilities.Query();
    Console.WriteLine($"Apple Neural Engine: {capabilities.DeviceName}");
    Console.WriteLine($"Max Network Size: {capabilities.MaxNetworkSize}");
    Console.WriteLine($"Peak TOPS: {capabilities.PeakTOPS:F1}");
    Console.WriteLine($"Supported Precisions: {string.Join(", ", capabilities.SupportedPrecisions)}");
    
    // Create ANE accelerator
    using var context = Context.Create().CPU();
    using var aneAccelerator = context.CreateANEAccelerator();
    
    // Demo ANE inference
    await DemoANEInference(aneAccelerator);
}
else
{
    Console.WriteLine("Apple Neural Engine not available (requires Apple Silicon Mac)");
}
```

#### ANE Model Runner Example

```csharp
static async Task DemoANEInference(Accelerator aneAccelerator)
{
    // Create ANE model runner
    using var modelRunner = new ANEModelRunner(aneAccelerator);
    
    try
    {
        // Compile model for ANE (example - requires actual Core ML model)
        // modelRunner.CompileModel("path/to/model.mlmodel");
        Console.WriteLine("Model compilation simulated (actual Core ML model needed)");
        
        // Prepare input tensor
        var inputShape = TensorShape.Create4D(1, 3, 224, 224); // Batch, Channels, Height, Width
        using var inputTensor = new UnifiedTensor<float>(inputShape, aneAccelerator);
        
        // Initialize with sample image data
        var inputData = new float[1, 3, 224, 224];
        var random = new Random(42);
        for (int b = 0; b < 1; b++)
        {
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < 224; h++)
                {
                    for (int w = 0; w < 224; w++)
                    {
                        inputData[b, c, h, w] = random.NextSingle();
                    }
                }
            }
        }
        
        inputTensor.CopyFromCPU(inputData);
        
        // Run prediction (simulated)
        var stopwatch = Stopwatch.StartNew();
        // var result = await modelRunner.PredictAsync(inputTensor);
        stopwatch.Stop();
        
        Console.WriteLine($"ANE Inference Time: {stopwatch.ElapsedMilliseconds} ms");
        Console.WriteLine("ANE inference demonstration completed (Core ML model required for full functionality)");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"ANE inference demo: {ex.Message}");
    }
}
```

---

## 🔧 Accelerator Selection and Management

### Automatic Accelerator Selection

```csharp
// Let UniversalCompute choose the best accelerator
using var context = Context.Create().EnableAllAccelerators();
using var accelerator = context.GetPreferredDevice(preferGPU: true).CreateAccelerator(context);

Console.WriteLine($"Selected: {accelerator.Name} ({accelerator.AcceleratorType})");
```

### Manual Accelerator Selection

```csharp
// Select by type
var gpuDevice = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
var cpuDevice = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.CPU);

// Select by performance criteria
var highMemoryDevice = context.Devices.OrderByDescending(d => d.MemorySize).First();
var mostCoresDevice = context.Devices.OrderByDescending(d => d.NumMultiProcessors).First();

// Create accelerators
if (gpuDevice != null)
{
    using var gpuAccelerator = gpuDevice.CreateAccelerator(context);
    Console.WriteLine($"GPU: {gpuAccelerator.Name}");
}
```

### Multi-Accelerator Workflows

```csharp
// Use multiple accelerators simultaneously
var allDevices = context.Devices.ToList();
var accelerators = new List<Accelerator>();

try
{
    foreach (var device in allDevices)
    {
        try
        {
            var accelerator = device.CreateAccelerator(context);
            accelerators.Add(accelerator);
            Console.WriteLine($"Initialized: {accelerator.Name}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to initialize {device.Name}: {ex.Message}");
        }
    }
    
    // Distribute work across accelerators
    await DistributeWorkload(accelerators);
}
finally
{
    // Clean up all accelerators
    foreach (var accelerator in accelerators)
    {
        accelerator.Dispose();
    }
}
```

### Performance Monitoring

```csharp
// Enable profiling for performance analysis
accelerator.EnableProfiling();

// Run workload
var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(MyKernel);
kernel(1024, buffer.View);
accelerator.Synchronize();

// Get profiling information
var profilingInfo = accelerator.GetProfilingInfo();
Console.WriteLine($"Kernel execution time: {profilingInfo.KernelExecutionTime.TotalMilliseconds} ms");
Console.WriteLine($"Memory transfer time: {profilingInfo.MemoryTransferTime.TotalMilliseconds} ms");
Console.WriteLine($"Throughput: {profilingInfo.ThroughputGBps:F2} GB/s");
Console.WriteLine($"Compute utilization: {profilingInfo.ComputeUtilization:P1}");
```

---

## 📊 Hardware Comparison and Benchmarking

### Performance Comparison Utility

```csharp
public class AcceleratorBenchmark
{
    public static async Task CompareAccelerators(Context context, int workloadSize = 1_000_000)
    {
        Console.WriteLine($"🏁 Accelerator Performance Comparison (workload size: {workloadSize:N0})");
        Console.WriteLine("=" + new string('=', 70));
        
        var results = new List<(string Name, double Time, double Throughput)>();
        
        foreach (var device in context.Devices)
        {
            try
            {
                using var accelerator = device.CreateAccelerator(context);
                var (time, throughput) = await BenchmarkAccelerator(accelerator, workloadSize);
                results.Add((accelerator.Name, time, throughput));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ {device.Name}: Failed ({ex.Message.Split('.')[0]})");
            }
        }
        
        // Sort by performance
        results.Sort((a, b) => a.Time.CompareTo(b.Time));
        
        // Display results
        var fastest = results.First().Time;
        foreach (var (name, time, throughput) in results)
        {
            var speedup = fastest / time;
            Console.WriteLine($"📊 {name}");
            Console.WriteLine($"   Time: {time:F2} ms");
            Console.WriteLine($"   Throughput: {throughput:F2} GB/s");
            Console.WriteLine($"   Speedup: {speedup:F1}x");
            Console.WriteLine();
        }
    }
    
    private static async Task<(double Time, double Throughput)> BenchmarkAccelerator(Accelerator accelerator, int workloadSize)
    {
        // Prepare test data
        using var bufferA = accelerator.Allocate1D<float>(workloadSize);
        using var bufferB = accelerator.Allocate1D<float>(workloadSize);
        using var bufferResult = accelerator.Allocate1D<float>(workloadSize);
        
        var dataA = new float[workloadSize];
        var dataB = new float[workloadSize];
        var random = new Random(42);
        
        for (int i = 0; i < workloadSize; i++)
        {
            dataA[i] = random.NextSingle();
            dataB[i] = random.NextSingle();
        }
        
        bufferA.CopyFromCPU(dataA);
        bufferB.CopyFromCPU(dataB);
        
        // Load kernel
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(BenchmarkKernel);
        
        // Warm-up
        kernel(workloadSize, bufferA.View, bufferB.View, bufferResult.View);
        accelerator.Synchronize();
        
        // Benchmark
        const int iterations = 5;
        var stopwatch = Stopwatch.StartNew();
        
        for (int i = 0; i < iterations; i++)
        {
            kernel(workloadSize, bufferA.View, bufferB.View, bufferResult.View);
        }
        
        accelerator.Synchronize();
        stopwatch.Stop();
        
        var avgTime = stopwatch.ElapsedMilliseconds / (double)iterations;
        var throughput = (workloadSize * sizeof(float) * 3) / (avgTime / 1000.0) / (1024 * 1024 * 1024);
        
        return (avgTime, throughput);
    }
    
    static void BenchmarkKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
    {
        result[index] = a[index] * b[index] + Math.Sin(a[index]) * Math.Cos(b[index]);
    }
}
```

### Usage

```csharp
// Run comprehensive accelerator comparison
using var context = Context.Create().EnableAllAccelerators();
await AcceleratorBenchmark.CompareAccelerators(context);
```

---

## 🎯 Best Practices

### Accelerator Selection Guidelines

1. **CPU**: General computing, debugging, small datasets
2. **GPU (CUDA)**: Large parallel workloads, AI/ML training
3. **GPU (OpenCL)**: Cross-platform GPU computing
4. **Intel AMX**: Matrix operations, neural network inference
5. **Intel NPU**: AI inference with power efficiency
6. **Apple Neural Engine**: ML inference on Apple Silicon
7. **Velocity SIMD**: High-throughput CPU vectorization

### Performance Optimization Tips

```csharp
// 1. Choose appropriate accelerator for workload
var accelerator = workloadType switch
{
    WorkloadType.MatrixMultiplication => context.CreateAMXAccelerator(),
    WorkloadType.AIInference => context.CreateNPUAccelerator(),
    WorkloadType.GeneralParallel => context.GetPreferredDevice(preferGPU: true).CreateAccelerator(context),
    _ => context.CreateCPUAccelerator()
};

// 2. Batch operations to amortize overhead
const int batchSize = 1024;
var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(MyKernel);
kernel(batchSize, buffer.View);

// 3. Reuse kernels and memory buffers
var kernelCache = new Dictionary<string, Action<Index1D, ArrayView<float>>>();
var memoryPool = new MemoryPool<float>(accelerator);

// 4. Use async operations when possible
var copyTask = buffer.CopyFromCPUAsync(data);
var kernelTask = Task.Run(() => kernel(size, buffer.View));
await Task.WhenAll(copyTask, kernelTask);
```

---

## 🔗 Related Topics

- **[Performance Tuning](Performance-Tuning)** - Optimize accelerator performance
- **[Memory Management](Memory-Management)** - Efficient memory usage patterns
- **[FFT Operations](FFT-Operations)** - Hardware-accelerated signal processing
- **[API Reference](API-Reference)** - Complete hardware accelerator API documentation

---

**⚡ Harness the full power of modern hardware with UniversalCompute!**