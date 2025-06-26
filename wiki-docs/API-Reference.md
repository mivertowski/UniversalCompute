# API Reference

Complete API documentation for UniversalCompute framework.

## 📚 Core Namespaces

### `UniversalCompute`
Core types and context management for the UniversalCompute framework.

### `UniversalCompute.Runtime`
Runtime services, accelerator management, and memory operations.

### `UniversalCompute.Runtime.CPU`
CPU-specific accelerator implementations and extensions.

### `UniversalCompute.FFT`
Fast Fourier Transform operations and accelerator abstractions.

### `UniversalCompute.Intel.AMX`
Intel Advanced Matrix Extensions support.

### `UniversalCompute.Intel.NPU`
Intel Neural Processing Unit integration.

### `UniversalCompute.Apple.NeuralEngine`
Apple Neural Engine hardware acceleration.

---

## 🏗️ Core Classes

### Context Class

The central orchestrator for UniversalCompute operations.

```csharp
namespace UniversalCompute
{
    public sealed class Context : IDisposable
    {
        // Static factory methods
        public static Context.Builder Create();
        public static Context CreateDefault();
        
        // Device management
        public IReadOnlyList<Device> Devices { get; }
        public Device GetPreferredDevice(bool preferGPU = true);
        public T GetDevice<T>(int deviceIndex) where T : Device;
        
        // Accelerator creation
        public CPUAccelerator CreateCPUAccelerator(int deviceIndex = 0);
        public Accelerator CreateAccelerator(Device device);
        
        // Hardware-specific accelerators
        public CPUAccelerator CreateAMXAccelerator(int deviceIndex = 0);
        public CPUAccelerator CreateNPUAccelerator(int deviceIndex = 0);
        public CPUAccelerator CreateANEAccelerator(int deviceIndex = 0);
        
        // Memory management
        public MemoryBuffer1D<T> AllocateGlobal<T>(long extent) where T : unmanaged;
        
        // Cleanup
        public void Dispose();
    }
}
```

#### Context.Builder Class

Fluent builder for configuring UniversalCompute contexts.

```csharp
public sealed class Builder
{
    // Accelerator enablement
    public Builder EnableAllAccelerators();
    public Builder DefaultCPU();
    public Builder CPU();
    public Builder CUDA();
    public Builder OpenCL();
    
    // Hardware-specific enablement
    public Builder AppleNeuralEngine();
    public Builder IntelNPU();
    public Builder IntelAMX();
    
    // Optimization settings
    public Builder Optimize(OptimizationLevel level);
    public Builder EnableNativeAOT();
    public Builder EnableProfiling();
    
    // Build context
    public Context ToContext();
}
```

### Accelerator Class

Abstract base class for all compute accelerators.

```csharp
namespace UniversalCompute.Runtime
{
    public abstract class Accelerator : IDisposable
    {
        // Properties
        public string Name { get; }
        public AcceleratorType AcceleratorType { get; }
        public long MemorySize { get; }
        public int MaxNumThreadsPerGroup { get; }
        public int MaxSharedMemoryPerGroup { get; }
        public int NumMultiProcessors { get; }
        
        // Memory allocation
        public MemoryBuffer1D<T> Allocate1D<T>(long extent) where T : unmanaged;
        public MemoryBuffer2D<T> Allocate2D<T>(LongIndex2D extent) where T : unmanaged;
        public MemoryBuffer3D<T> Allocate3D<T>(LongIndex3D extent) where T : unmanaged;
        
        // Kernel loading and execution
        public Action<Index1D> LoadAutoGroupedStreamKernel<TDelegate>(TDelegate kernel);
        public Action<Index1D, T1> LoadAutoGroupedStreamKernel<T1, TDelegate>(TDelegate kernel);
        public Action<Index1D, T1, T2> LoadAutoGroupedStreamKernel<T1, T2, TDelegate>(TDelegate kernel);
        
        // Stream management
        public AcceleratorStream CreateStream();
        public void Synchronize();
        
        // Performance and profiling
        public void EnableProfiling();
        public ProfilingInfo GetProfilingInfo();
        
        // Cleanup
        public void Dispose();
    }
}
```

### Device Class

Represents a compute device available in the system.

```csharp
namespace UniversalCompute.Runtime
{
    public abstract class Device
    {
        // Properties
        public string Name { get; }
        public AcceleratorType AcceleratorType { get; }
        public long MemorySize { get; }
        public int NumMultiProcessors { get; }
        public DeviceCapabilities Capabilities { get; }
        
        // Accelerator creation
        public Accelerator CreateAccelerator(Context context);
        public Accelerator CreateAccelerator(Context context, AcceleratorMode mode);
        
        // Capability queries
        public bool Supports<T>() where T : Capability;
        public T GetCapability<T>() where T : Capability;
        
        // Device information
        public DeviceInfo GetDeviceInfo();
        public override string ToString();
    }
}
```

---

## 🧮 Memory Management

### MemoryBuffer Classes

Type-safe memory buffers for GPU/accelerator memory.

```csharp
namespace UniversalCompute.Runtime
{
    // 1D Memory Buffer
    public sealed class MemoryBuffer1D<T> : MemoryBuffer<T> where T : unmanaged
    {
        public long Length { get; }
        public ArrayView<T> View { get; }
        
        // Data transfer
        public void CopyFromCPU(T[] data);
        public void CopyFromCPU(ReadOnlySpan<T> data);
        public void CopyToCPU(T[] target);
        public void CopyToCPU(Span<T> target);
        
        // Async operations
        public Task CopyFromCPUAsync(T[] data);
        public Task CopyToCPUAsync(T[] target);
        
        // Conversion
        public T[] GetAsArray1D();
        public ArrayView<T> GetArrayView();
    }
    
    // 2D Memory Buffer
    public sealed class MemoryBuffer2D<T> : MemoryBuffer<T> where T : unmanaged
    {
        public LongIndex2D Extent { get; }
        public ArrayView2D<T, Stride2D.DenseX> View { get; }
        
        // Data transfer
        public void CopyFromCPU(T[,] data);
        public T[,] GetAsArray2D();
    }
    
    // 3D Memory Buffer
    public sealed class MemoryBuffer3D<T> : MemoryBuffer<T> where T : unmanaged
    {
        public LongIndex3D Extent { get; }
        public ArrayView3D<T, Stride3D.DenseXY> View { get; }
        
        // Data transfer
        public void CopyFromCPU(T[,,] data);
        public T[,,] GetAsArray3D();
    }
}
```

### ArrayView Classes

High-performance views for accessing accelerator memory.

```csharp
namespace UniversalCompute
{
    // 1D Array View
    public readonly struct ArrayView<T> where T : unmanaged
    {
        public long Length { get; }
        public ref T this[long index] { get; }
        public ref T this[Index1D index] { get; }
        
        // Slicing operations
        public ArrayView<T> SubView(long offset, long length);
        public ArrayView<T> GetSubView(Range range);
        
        // Casting and conversion
        public ArrayView<TOther> Cast<TOther>() where TOther : unmanaged;
        public ArrayView2D<T, Stride2D.DenseX> As2DView(long width, long height);
    }
    
    // 2D Array View
    public readonly struct ArrayView2D<T, TStride> 
        where T : unmanaged 
        where TStride : struct, IStride2D
    {
        public LongIndex2D Extent { get; }
        public ref T this[LongIndex2D index] { get; }
        public ref T this[long x, long y] { get; }
        
        // Row/column access
        public ArrayView<T> GetRowView(long row);
        public ArrayView<T> GetColumnView(long column);
        
        // Slicing
        public ArrayView2D<T, TStride> GetSubView(LongIndex2D offset, LongIndex2D extent);
    }
}
```

---

## ⚡ Hardware Accelerators

### Intel AMX Support

Advanced Matrix Extensions for high-performance matrix operations.

```csharp
namespace UniversalCompute.Intel.AMX
{
    public readonly struct AMXCapabilities
    {
        public bool IsSupported { get; }
        public int MaxTiles { get; }
        public int MaxTileRows { get; }
        public int MaxTileColumns { get; }
        public int MaxTileBytes { get; }
        public bool SupportsBF16 { get; }
        public bool SupportsInt8 { get; }
        public bool SupportsFloat32 { get; }
        public double EstimatedBandwidthGBps { get; }
        
        // Static methods
        public static bool IsAMXSupported();
        public static AMXCapabilities Query();
        
        // Performance estimation
        public double GetEstimatedPerformance(AMXDataType dataType);
        public (int tileM, int tileN, int tileK) GetOptimalTileSize(int m, int n, int k, AMXDataType dataType);
    }
    
    public enum AMXDataType
    {
        BFloat16,
        Int8,
        Float32
    }
    
    public sealed class AMXTileConfiguration
    {
        public AMXDataType DataType { get; set; }
        public int TileRows { get; set; }
        public int TileColumns { get; set; }
        public byte Palette { get; set; }
        
        // Factory methods
        public static AMXTileConfiguration CreateDefault(AMXCapabilities capabilities);
        public static AMXTileConfiguration CreateForDataType(AMXDataType dataType, AMXCapabilities capabilities);
        
        // Configuration methods
        public AMXTileConfiguration WithDataType(AMXDataType dataType);
    }
}
```

### Intel NPU Support

Neural Processing Unit integration for AI workloads.

```csharp
namespace UniversalCompute.Intel.NPU
{
    public readonly struct NPUCapabilities
    {
        public bool IsSupported { get; }
        public string DeviceName { get; }
        public int MaxBatchSize { get; }
        public long MaxMemorySize { get; }
        public NPUPrecision[] SupportedPrecisions { get; }
        public double PeakTOPS { get; }
        
        // Static methods
        public static bool IsNPUSupported();
        public static NPUCapabilities Query();
    }
    
    public enum NPUPrecision
    {
        Int8,
        Int16,
        Float16,
        BFloat16
    }
    
    public sealed class NPUInferenceEngine : IDisposable
    {
        // Model loading
        public void LoadModel(string modelPath);
        public void LoadModel(byte[] modelData);
        
        // Inference execution
        public Task<ArrayView<float>> RunInferenceAsync(ArrayView<float> input);
        public ArrayView<float> RunInference(ArrayView<float> input);
        
        // Batch processing
        public Task<ArrayView<float>[]> RunBatchInferenceAsync(ArrayView<float>[] inputs);
        
        // Performance monitoring
        public InferenceMetrics GetMetrics();
        
        public void Dispose();
    }
}
```

### Apple Neural Engine Support

Hardware-accelerated AI inference on Apple Silicon.

```csharp
namespace UniversalCompute.Apple.NeuralEngine
{
    public readonly struct ANECapabilities
    {
        public bool IsSupported { get; }
        public string DeviceName { get; }
        public int MaxNetworkSize { get; }
        public ANEPrecision[] SupportedPrecisions { get; }
        public double PeakTOPS { get; }
        
        // Static methods
        public static bool IsANESupported();
        public static ANECapabilities Query();
    }
    
    public enum ANEPrecision
    {
        Float16,
        Int8,
        Int16
    }
    
    public sealed class ANEModelRunner : IDisposable
    {
        // Model compilation and loading
        public void CompileModel(string modelPath);
        public void LoadCompiledModel(byte[] compiledModel);
        
        // Prediction
        public Task<Tensor<float>> PredictAsync(Tensor<float> input);
        public Tensor<float> Predict(Tensor<float> input);
        
        // Batch operations
        public Task<Tensor<float>[]> PredictBatchAsync(Tensor<float>[] inputs);
        
        // Resource management
        public void Dispose();
    }
}
```

---

## 📊 FFT Operations

Fast Fourier Transform support with hardware acceleration.

```csharp
namespace UniversalCompute.FFT
{
    public abstract class FFTAccelerator : IDisposable
    {
        public string Name { get; }
        public bool IsAvailable { get; }
        public FFTCapabilities Capabilities { get; }
        
        // 1D FFT operations
        public abstract void FFT1D(ArrayView<Complex> input, ArrayView<Complex> output, bool forward = true, AcceleratorStream stream = null);
        public abstract void FFT1DReal(ArrayView<float> input, ArrayView<Complex> output, AcceleratorStream stream = null);
        public abstract void IFFT1DReal(ArrayView<Complex> input, ArrayView<float> output, AcceleratorStream stream = null);
        
        // 2D FFT operations
        public abstract void FFT2D(ArrayView2D<Complex, Stride2D.DenseX> input, ArrayView2D<Complex, Stride2D.DenseX> output, bool forward = true, AcceleratorStream stream = null);
        
        // Performance estimation
        public abstract FFTPerformanceEstimate EstimatePerformance(int size, bool is2D);
        public abstract int GetOptimalFFTSize(int requestedSize);
        
        public void Dispose();
    }
    
    public sealed class FFTManager : IDisposable
    {
        public FFTManager(Context context);
        
        // Accelerator selection
        public FFTAccelerator GetBestAccelerator(int size, bool is2D, bool allowFallback = true);
        public IReadOnlyList<FFTAccelerator> GetAvailableAccelerators();
        
        // Direct FFT operations
        public void FFT1D(ArrayView<Complex> input, ArrayView<Complex> output, bool forward = true);
        public void FFT2D(ArrayView2D<Complex, Stride2D.DenseX> input, ArrayView2D<Complex, Stride2D.DenseX> output, bool forward = true);
        
        // Plan-based operations
        public FFTPlan1D CreatePlan1D(int length, bool isReal = false);
        public FFTPlan2D CreatePlan2D(int width, int height);
        
        public void Dispose();
    }
    
    public sealed class FFTPlan1D : FFTPlan
    {
        public int Length { get; }
        public bool IsReal { get; }
        public int OptimalLength { get; }
        
        // Execution methods
        public void Forward(ArrayView<Complex> input, ArrayView<Complex> output, AcceleratorStream stream = null);
        public void Inverse(ArrayView<Complex> input, ArrayView<Complex> output, AcceleratorStream stream = null);
        public void ForwardReal(ArrayView<float> input, ArrayView<Complex> output, AcceleratorStream stream = null);
        public void InverseReal(ArrayView<Complex> input, ArrayView<float> output, AcceleratorStream stream = null);
    }
}
```

---

## 🎯 Index Types

Multi-dimensional indexing for kernel operations.

```csharp
namespace UniversalCompute
{
    // 1D Index
    public readonly struct Index1D : IIndex
    {
        public int X { get; }
        
        // Operators
        public static implicit operator int(Index1D index);
        public static implicit operator Index1D(int value);
        public static Index1D operator +(Index1D left, Index1D right);
        public static Index1D operator -(Index1D left, Index1D right);
        public static Index1D operator *(Index1D left, int right);
    }
    
    // 2D Index
    public readonly struct Index2D : IIndex
    {
        public int X { get; }
        public int Y { get; }
        
        // Construction
        public Index2D(int x, int y);
        public Index2D(Index1D linearIndex, int width);
        
        // Conversion
        public Index1D ToLinearIndex(int width);
        
        // Operators
        public static Index2D operator +(Index2D left, Index2D right);
        public static Index2D operator -(Index2D left, Index2D right);
        public static Index2D operator *(Index2D left, int right);
    }
    
    // 3D Index
    public readonly struct Index3D : IIndex
    {
        public int X { get; }
        public int Y { get; }
        public int Z { get; }
        
        // Construction
        public Index3D(int x, int y, int z);
        public Index3D(Index1D linearIndex, int width, int height);
        
        // Conversion
        public Index1D ToLinearIndex(int width, int height);
        public Index2D ToIndex2D();
        
        // Operators
        public static Index3D operator +(Index3D left, Index3D right);
        public static Index3D operator -(Index3D left, Index3D right);
        public static Index3D operator *(Index3D left, int right);
    }
}
```

---

## 🧠 Tensor Operations

Multi-dimensional tensor support for AI/ML workloads.

```csharp
namespace UniversalCompute.Core
{
    public sealed class UnifiedTensor<T> : IDisposable where T : unmanaged
    {
        public TensorShape Shape { get; }
        public int Rank { get; }
        public long[] Dimensions { get; }
        public long TotalElements { get; }
        
        // Construction
        public UnifiedTensor(TensorShape shape, Accelerator accelerator);
        public UnifiedTensor(long[] dimensions, Accelerator accelerator);
        
        // Data access
        public ArrayView<T> GetLinearView();
        public ArrayView2D<T, Stride2D.DenseX> Get2DView();
        public ArrayView3D<T, Stride3D.DenseXY> Get3DView();
        
        // Element access
        public ref T this[params long[] indices] { get; }
        public ref T this[long linearIndex] { get; }
        
        // Operations
        public UnifiedTensor<T> Add(UnifiedTensor<T> other);
        public UnifiedTensor<T> Multiply(UnifiedTensor<T> other);
        public UnifiedTensor<T> MatrixMultiply(UnifiedTensor<T> other);
        public UnifiedTensor<T> Transpose(int axis1, int axis2);
        public UnifiedTensor<T> Reshape(TensorShape newShape);
        
        // Reduction operations
        public T Sum();
        public T Mean();
        public T Max();
        public T Min();
        public UnifiedTensor<T> Sum(int axis);
        public UnifiedTensor<T> Mean(int axis);
        
        // Data transfer
        public void CopyFromCPU(Array data);
        public Array CopyToCPU();
        public Task CopyFromCPUAsync(Array data);
        public Task<Array> CopyToCPUAsync();
        
        public void Dispose();
    }
    
    public readonly struct TensorShape
    {
        public int Rank { get; }
        public long[] Dimensions { get; }
        public long TotalElements { get; }
        
        // Construction
        public TensorShape(params long[] dimensions);
        public static TensorShape Create1D(long length);
        public static TensorShape Create2D(long height, long width);
        public static TensorShape Create3D(long depth, long height, long width);
        public static TensorShape Create4D(long batch, long channels, long height, long width);
        
        // Operations
        public TensorShape Transpose(int axis1, int axis2);
        public TensorShape Reshape(params long[] newDimensions);
        public TensorShape Squeeze(int axis = -1);
        public TensorShape Unsqueeze(int axis);
        
        // Queries
        public bool IsCompatibleWith(TensorShape other);
        public bool CanBroadcastWith(TensorShape other);
        public TensorShape GetBroadcastShape(TensorShape other);
    }
}
```

---

## 🔧 Utility Classes

### Performance Monitoring

```csharp
namespace UniversalCompute.Runtime
{
    public sealed class ProfilingInfo
    {
        public TimeSpan KernelExecutionTime { get; }
        public TimeSpan MemoryTransferTime { get; }
        public long MemoryAllocated { get; }
        public long MemoryFreed { get; }
        public int KernelLaunches { get; }
        
        // Performance metrics
        public double ThroughputGBps { get; }
        public double ComputeUtilization { get; }
        public double MemoryUtilization { get; }
        
        // Reset counters
        public void Reset();
        
        // Export data
        public string ToJson();
        public Dictionary<string, object> ToDictionary();
    }
    
    public sealed class AcceleratorStream : IDisposable
    {
        public bool IsValid { get; }
        public Accelerator Accelerator { get; }
        
        // Synchronization
        public void Synchronize();
        public Task SynchronizeAsync();
        
        // Memory operations
        public void MemcpyAsync<T>(ArrayView<T> destination, ArrayView<T> source) where T : unmanaged;
        
        // Events and timing
        public void RecordEvent(AcceleratorEvent evt);
        public TimeSpan GetElapsedTime(AcceleratorEvent start, AcceleratorEvent end);
        
        public void Dispose();
    }
}
```

### Error Handling

```csharp
namespace UniversalCompute
{
    public class AcceleratorException : Exception
    {
        public AcceleratorException(string message) : base(message) { }
        public AcceleratorException(string message, Exception innerException) : base(message, innerException) { }
    }
    
    public class InsufficientMemoryException : AcceleratorException
    {
        public long RequestedMemory { get; }
        public long AvailableMemory { get; }
        
        public InsufficientMemoryException(long requested, long available) 
            : base($"Insufficient memory: requested {requested} bytes, only {available} bytes available")
        {
            RequestedMemory = requested;
            AvailableMemory = available;
        }
    }
    
    public class KernelCompilationException : AcceleratorException
    {
        public string KernelName { get; }
        public string CompilerOutput { get; }
        
        public KernelCompilationException(string kernelName, string compilerOutput)
            : base($"Failed to compile kernel '{kernelName}': {compilerOutput}")
        {
            KernelName = kernelName;
            CompilerOutput = compilerOutput;
        }
    }
}
```

---

## 📖 Usage Examples

### Basic Kernel Example

```csharp
using UniversalCompute;
using UniversalCompute.Runtime;

// Create context and get accelerator
using var context = Context.Create().EnableAllAccelerators();
using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);

// Define kernel
static void VectorAddKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
{
    result[index] = a[index] + b[index];
}

// Allocate memory
const int size = 1024;
using var bufferA = accelerator.Allocate1D<float>(size);
using var bufferB = accelerator.Allocate1D<float>(size);
using var bufferResult = accelerator.Allocate1D<float>(size);

// Initialize data
var dataA = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
var dataB = Enumerable.Range(0, size).Select(i => (float)i * 2).ToArray();

bufferA.CopyFromCPU(dataA);
bufferB.CopyFromCPU(dataB);

// Load and execute kernel
var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);
kernel(size, bufferA.View, bufferB.View, bufferResult.View);

// Synchronize and get results
accelerator.Synchronize();
var result = bufferResult.GetAsArray1D();
```

### FFT Example

```csharp
using UniversalCompute;
using UniversalCompute.FFT;
using System.Numerics;

// Create context and FFT manager
using var context = Context.Create().EnableAllAccelerators();
using var fftManager = new FFTManager(context);

// Create input signal
const int N = 1024;
var signal = new Complex[N];
for (int i = 0; i < N; i++)
{
    signal[i] = new Complex(Math.Sin(2 * Math.PI * 5 * i / N), 0); // 5 Hz sine wave
}

// Allocate GPU memory
using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);
using var inputBuffer = accelerator.Allocate1D<Complex>(N);
using var outputBuffer = accelerator.Allocate1D<Complex>(N);

// Copy data and perform FFT
inputBuffer.CopyFromCPU(signal);
fftManager.FFT1D(inputBuffer.View, outputBuffer.View, forward: true);

// Get results
var fftResult = outputBuffer.GetAsArray1D();
```

### Tensor Operations Example

```csharp
using UniversalCompute;
using UniversalCompute.Core;

// Create context and accelerator
using var context = Context.Create().EnableAllAccelerators();
using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);

// Create tensors
var shape = TensorShape.Create2D(1024, 1024);
using var tensorA = new UnifiedTensor<float>(shape, accelerator);
using var tensorB = new UnifiedTensor<float>(shape, accelerator);

// Initialize with data
var dataA = new float[1024, 1024];
var dataB = new float[1024, 1024];
// ... populate data arrays ...

tensorA.CopyFromCPU(dataA);
tensorB.CopyFromCPU(dataB);

// Perform operations
using var result = tensorA.MatrixMultiply(tensorB);
var sum = result.Sum();
var mean = result.Mean();

// Get result data
var resultArray = (float[,])result.CopyToCPU();
```

---

For more detailed examples and advanced usage patterns, see the [Examples Gallery](Examples-Gallery) and [Quick Start Tutorial](Quick-Start-Tutorial).

---

**📖 Continue exploring:** [Hardware Accelerators](Hardware-Accelerators) | [Performance Tuning](Performance-Tuning) | [Examples Gallery](Examples-Gallery)