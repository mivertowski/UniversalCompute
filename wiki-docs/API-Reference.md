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

### `UniversalCompute.DependencyInjection`
Service registration, resolution, and lifetime management for UniversalCompute components.

### `UniversalCompute.Attributes`
Kernel attributes system for method marking and optimization configuration.

### `UniversalCompute.SourceGeneration`
Source generators for automatic kernel launcher generation and type discovery.

### `UniversalCompute.Memory.Unified`
Advanced unified memory management across accelerators with optimization strategies.

### `UniversalCompute.Performance`
Performance monitoring, profiling, and measurement capabilities.

### `UniversalCompute.AOT`
Native Ahead-of-Time compilation support for .NET 9.0 with preview features and cross-platform deployment.

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
    
    // Dependency injection
    public Builder ConfigureServices(Action<IServiceContainer> configure);
    public Builder AddSingleton<TService, TImplementation>() where TImplementation : class, TService;
    public Builder AddTransient<TService, TImplementation>() where TImplementation : class, TService;
    public Builder AddScoped<TService, TImplementation>() where TImplementation : class, TService;
    
    // Source generation
    public Builder EnableSourceGeneration();
    public Builder ConfigureSourceGeneration(SourceGenerationOptions options);
    
    // Memory management
    public Builder ConfigureMemoryManager(UnifiedMemoryOptions options);
    public Builder EnableMemoryPooling();
    
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

## 🔧 Dependency Injection Framework

Comprehensive dependency injection system for service registration and resolution.

```csharp
namespace UniversalCompute.DependencyInjection
{
    public interface IServiceContainer
    {
        // Service registration
        IServiceContainer AddSingleton<TService, TImplementation>() where TImplementation : class, TService;
        IServiceContainer AddSingleton<TService>(TService instance) where TService : class;
        IServiceContainer AddSingleton<TService>(Func<IServiceProvider, TService> factory) where TService : class;
        
        IServiceContainer AddTransient<TService, TImplementation>() where TImplementation : class, TService;
        IServiceContainer AddTransient<TService>(Func<IServiceProvider, TService> factory) where TService : class;
        
        IServiceContainer AddScoped<TService, TImplementation>() where TImplementation : class, TService;
        IServiceContainer AddScoped<TService>(Func<IServiceProvider, TService> factory) where TService : class;
        
        // Build service provider
        IServiceProvider BuildServiceProvider();
    }
    
    public interface IServiceProvider
    {
        // Service resolution
        TService GetRequiredService<TService>();
        TService GetService<TService>();
        object GetRequiredService(Type serviceType);
        object GetService(Type serviceType);
        
        // Service enumeration
        IEnumerable<TService> GetServices<TService>();
        IEnumerable<object> GetServices(Type serviceType);
        
        // Scope management
        IServiceScope CreateScope();
    }
    
    public interface IServiceScope : IDisposable
    {
        IServiceProvider ServiceProvider { get; }
    }
    
    public enum ServiceLifetime
    {
        Singleton,
        Transient,
        Scoped
    }
    
    public sealed class ServiceDescriptor
    {
        public Type ServiceType { get; }
        public Type ImplementationType { get; }
        public ServiceLifetime Lifetime { get; }
        public Func<IServiceProvider, object> Factory { get; }
        public object Instance { get; }
        
        // Factory methods
        public static ServiceDescriptor Singleton<TService, TImplementation>() 
            where TImplementation : class, TService;
        public static ServiceDescriptor Transient<TService, TImplementation>() 
            where TImplementation : class, TService;
        public static ServiceDescriptor Scoped<TService, TImplementation>() 
            where TImplementation : class, TService;
    }
}
```

### Service Registration Examples

```csharp
using UniversalCompute;
using UniversalCompute.DependencyInjection;

// Create context with dependency injection
using var context = Context.Create()
    .ConfigureServices(services =>
    {
        // Register accelerator services
        services.AddSingleton<IAcceleratorManager, AcceleratorManager>();
        services.AddScoped<IMemoryManager, UnifiedMemoryManager>();
        services.AddTransient<IKernelCompiler, JITKernelCompiler>();
        
        // Register custom services
        services.AddSingleton<IDataProcessor, GPUDataProcessor>();
        services.AddScoped<IPerformanceMonitor, RealTimeMonitor>();
    })
    .EnableAllAccelerators()
    .ToContext();

// Resolve services
var serviceProvider = context.ServiceProvider;
var acceleratorManager = serviceProvider.GetRequiredService<IAcceleratorManager>();
var memoryManager = serviceProvider.GetRequiredService<IMemoryManager>();

// Use scoped services
using var scope = serviceProvider.CreateScope();
var scopedService = scope.ServiceProvider.GetRequiredService<IPerformanceMonitor>();
```

---

## 🏷️ Kernel Attributes System

Attribute-based kernel discovery, registration, and optimization configuration.

```csharp
namespace UniversalCompute.Attributes
{
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class KernelMethodAttribute : Attribute
    {
        public string Name { get; set; }
        public bool AutoGenerate { get; set; } = true;
        public KernelOptimization Optimization { get; set; } = KernelOptimization.Default;
        public int SharedMemorySize { get; set; } = 0;
        public int MaxThreadsPerGroup { get; set; } = 0;
        
        public KernelMethodAttribute() { }
        public KernelMethodAttribute(string name) { Name = name; }
    }
    
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class OptimizedKernelAttribute : Attribute
    {
        public KernelOptimization OptimizationLevel { get; }
        public bool EnableVectorization { get; set; } = true;
        public bool EnableLoopUnrolling { get; set; } = true;
        public bool EnableMemoryCoalescing { get; set; } = true;
        public int PreferredBlockSize { get; set; } = 0;
        
        public OptimizedKernelAttribute(KernelOptimization level)
        {
            OptimizationLevel = level;
        }
    }
    
    [AttributeUsage(AttributeTargets.Parameter)]
    public sealed class SharedMemoryAttribute : Attribute
    {
        public int Size { get; }
        public bool DynamicSize { get; set; } = false;
        
        public SharedMemoryAttribute(int size)
        {
            Size = size;
        }
    }
    
    [AttributeUsage(AttributeTargets.Parameter)]
    public sealed class ConstantMemoryAttribute : Attribute
    {
        public bool ReadOnly { get; set; } = true;
        public int CacheHint { get; set; } = 0;
    }
    
    public enum KernelOptimization
    {
        None,
        Default,
        Aggressive,
        MemoryOptimized,
        ComputeOptimized,
        Custom
    }
    
    public sealed class KernelDiscovery
    {
        // Discover kernels in assembly
        public static IEnumerable<KernelInfo> DiscoverKernels(Assembly assembly);
        public static IEnumerable<KernelInfo> DiscoverKernels<T>();
        
        // Register discovered kernels
        public static void RegisterKernels(Accelerator accelerator, Assembly assembly);
        public static void RegisterKernels<T>(Accelerator accelerator);
        
        // Get kernel metadata
        public static KernelMetadata GetKernelMetadata(MethodInfo method);
        public static bool IsKernelMethod(MethodInfo method);
    }
    
    public sealed class KernelInfo
    {
        public MethodInfo Method { get; }
        public KernelMethodAttribute KernelAttribute { get; }
        public OptimizedKernelAttribute OptimizationAttribute { get; }
        public string Name { get; }
        public Type[] ParameterTypes { get; }
        public KernelMetadata Metadata { get; }
    }
}
```

### Kernel Attributes Usage Examples

```csharp
using UniversalCompute;
using UniversalCompute.Attributes;

public static class MyKernels
{
    [KernelMethod("VectorAdd")]
    [OptimizedKernel(KernelOptimization.MemoryOptimized)]
    public static void VectorAddKernel(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> result)
    {
        result[index] = a[index] + b[index];
    }
    
    [KernelMethod("MatrixMultiply")]
    [OptimizedKernel(KernelOptimization.ComputeOptimized, 
        PreferredBlockSize = 256,
        EnableLoopUnrolling = true)]
    public static void MatrixMultiplyKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> result,
        [SharedMemory(1024)] ArrayView<float> sharedA,
        [SharedMemory(1024)] ArrayView<float> sharedB)
    {
        // Matrix multiplication with shared memory optimization
        var row = index.X;
        var col = index.Y;
        // ... implementation
    }
    
    [KernelMethod(AutoGenerate = false)]
    public static void CustomKernel(
        Index1D index,
        [ConstantMemory] ArrayView<float> constants,
        ArrayView<float> data)
    {
        data[index] *= constants[0];
    }
}

// Auto-discovery and registration
using var context = Context.Create().EnableAllAccelerators();
using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);

// Register all kernels from type
KernelDiscovery.RegisterKernels<MyKernels>(accelerator);

// Manual kernel loading
var kernelInfo = KernelDiscovery.DiscoverKernels<MyKernels>()
    .First(k => k.Name == "VectorAdd");
```

---

## 🔄 Source Generators

Automatic kernel launcher generation and type discovery for compile-time optimization.

```csharp
namespace UniversalCompute.SourceGeneration
{
    public sealed class SourceGenerationOptions
    {
        public bool GenerateKernelLaunchers { get; set; } = true;
        public bool GenerateTypeDiscovery { get; set; } = true;
        public bool GenerateNativeBindings { get; set; } = false;
        public bool EnableNullableAnnotations { get; set; } = true;
        public string OutputNamespace { get; set; } = "UniversalCompute.Generated";
        public KernelLauncherOptions LauncherOptions { get; set; } = new();
        public TypeDiscoveryOptions DiscoveryOptions { get; set; } = new();
    }
    
    public sealed class KernelLauncherOptions
    {
        public bool GenerateAsyncVariants { get; set; } = true;
        public bool GenerateStreamOverloads { get; set; } = true;
        public bool GenerateProfiledVariants { get; set; } = false;
        public bool IncludeParameterValidation { get; set; } = true;
        public bool GenerateXmlDocumentation { get; set; } = true;
        public NamingConvention NamingConvention { get; set; } = NamingConvention.Pascal;
    }
    
    public sealed class TypeDiscoveryOptions
    {
        public bool ScanReferencedAssemblies { get; set; } = false;
        public string[] IncludeNamespaces { get; set; } = Array.Empty<string>();
        public string[] ExcludeNamespaces { get; set; } = Array.Empty<string>();
        public bool GenerateRegistrationMethods { get; set; } = true;
        public bool GenerateFactoryMethods { get; set; } = true;
    }
    
    public enum NamingConvention
    {
        Pascal,
        Camel,
        Snake,
        Kebab
    }
    
    // Generated launcher interface
    public interface IKernelLauncher<TKernel>
    {
        void Launch(int extent, params object[] args);
        void LaunchAsync(int extent, AcceleratorStream stream, params object[] args);
        Task LaunchTaskAsync(int extent, params object[] args);
        
        // Profiled variants
        KernelExecutionResult LaunchWithProfiling(int extent, params object[] args);
        Task<KernelExecutionResult> LaunchWithProfilingAsync(int extent, params object[] args);
    }
    
    public readonly struct KernelExecutionResult
    {
        public TimeSpan ExecutionTime { get; }
        public long MemoryTransferred { get; }
        public double ThroughputGBps { get; }
        public bool WasSuccessful { get; }
        public string ErrorMessage { get; }
    }
}
```

### Source Generator Usage Examples

```csharp
// Mark assembly for source generation
[assembly: UniversalCompute.SourceGeneration.GenerateKernelLaunchers]
[assembly: UniversalCompute.SourceGeneration.GenerateTypeDiscovery]

// Define kernels - launchers will be auto-generated
public static partial class MyKernels
{
    [KernelMethod]
    public static void VectorAdd(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
    {
        result[index] = a[index] + b[index];
    }
}

// Generated launcher code (automatically created by source generator)
public static partial class MyKernels
{
    private static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _vectorAddKernel;
    
    public static void LaunchVectorAdd(
        Accelerator accelerator,
        int extent,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> result)
    {
        _vectorAddKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAdd);
        
        _vectorAddKernel(extent, a, b, result);
    }
    
    public static async Task LaunchVectorAddAsync(
        Accelerator accelerator,
        AcceleratorStream stream,
        int extent,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> result)
    {
        _vectorAddKernel ??= accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAdd);
        
        _vectorAddKernel(extent, a, b, result);
        await stream.SynchronizeAsync();
    }
}

// Usage with generated launchers
using var context = Context.Create()
    .EnableSourceGeneration()
    .EnableAllAccelerators();
using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);

// Allocate memory
using var bufferA = accelerator.Allocate1D<float>(1024);
using var bufferB = accelerator.Allocate1D<float>(1024);
using var result = accelerator.Allocate1D<float>(1024);

// Use generated launcher
MyKernels.LaunchVectorAdd(accelerator, 1024, bufferA.View, bufferB.View, result.View);

// Async variant
using var stream = accelerator.CreateStream();
await MyKernels.LaunchVectorAddAsync(accelerator, stream, 1024, bufferA.View, bufferB.View, result.View);
```

---

## 🧠 Advanced Memory Management

Unified memory management system with cross-accelerator optimization and pooling strategies.

```csharp
namespace UniversalCompute.Memory.Unified
{
    public sealed class UniversalMemoryManager : IDisposable
    {
        public MemoryPlacementStrategy PlacementStrategy { get; set; }
        public bool EnableMemoryPooling { get; set; } = true;
        public long TotalManagedMemory { get; }
        public long AvailableMemory { get; }
        public MemoryStatistics Statistics { get; }
        
        // Memory allocation
        public UnifiedMemoryBuffer<T> Allocate<T>(long size, MemoryPlacement placement = MemoryPlacement.Auto) where T : unmanaged;
        public UnifiedMemoryBuffer<T> AllocatePageLocked<T>(long size) where T : unmanaged;
        public UnifiedMemoryBuffer<T> AllocateUnified<T>(long size) where T : unmanaged;
        
        // Memory pool management
        public void ConfigurePool(MemoryPoolConfiguration config);
        public void WarmUpPool(Type elementType, long[] sizes);
        public void TrimPool();
        
        // Cross-accelerator transfers
        public void CopyBetweenAccelerators<T>(
            ArrayView<T> source, 
            ArrayView<T> destination,
            Accelerator sourceAccelerator,
            Accelerator destinationAccelerator) where T : unmanaged;
        
        // Memory bandwidth optimization
        public MemoryTransferPlan OptimizeTransfer<T>(
            T[] source,
            Accelerator[] targetAccelerators,
            MemoryAccess accessPattern) where T : unmanaged;
        
        public void Dispose();
    }
    
    public sealed class UnifiedMemoryBuffer<T> : IDisposable where T : unmanaged
    {
        public long Size { get; }
        public MemoryPlacement Placement { get; }
        public bool IsPageLocked { get; }
        public bool IsUnified { get; }
        public ArrayView<T> View { get; }
        
        // Multi-accelerator views
        public ArrayView<T> GetView(Accelerator accelerator);
        public void MigrateToAccelerator(Accelerator target);
        public void PrefetchToAccelerator(Accelerator target);
        
        // Synchronization
        public void Synchronize();
        public Task SynchronizeAsync();
        
        // Copy operations with optimization
        public void CopyFromCPU(T[] data, CopyHints hints = CopyHints.None);
        public void CopyToCPU(T[] target, CopyHints hints = CopyHints.None);
        public Task CopyFromCPUAsync(T[] data, CopyHints hints = CopyHints.None);
        public Task CopyToCPUAsync(T[] target, CopyHints hints = CopyHints.None);
        
        public void Dispose();
    }
    
    public enum MemoryPlacement
    {
        Auto,
        Host,
        Device,
        Unified,
        PageLocked
    }
    
    public enum MemoryPlacementStrategy
    {
        Performance,
        Memory,
        Balanced,
        Custom
    }
    
    public enum CopyHints
    {
        None = 0,
        Sequential = 1,
        Random = 2,
        WriteOnly = 4,
        ReadOnly = 8,
        Streaming = 16
    }
    
    public sealed class MemoryPoolConfiguration
    {
        public long MaxPoolSize { get; set; } = 1024 * 1024 * 1024; // 1GB
        public int MaxBuffersPerSize { get; set; } = 64;
        public TimeSpan BufferLifetime { get; set; } = TimeSpan.FromMinutes(5);
        public bool EnableSizeRounding { get; set; } = true;
        public double GrowthFactor { get; set; } = 1.5;
        public long[] PreallocatedSizes { get; set; } = Array.Empty<long>();
    }
    
    public readonly struct MemoryStatistics
    {
        public long TotalAllocated { get; }
        public long TotalFreed { get; }
        public long PeakUsage { get; }
        public int ActiveAllocations { get; }
        public int PoolHits { get; }
        public int PoolMisses { get; }
        public double PoolHitRate { get; }
        public long FragmentedMemory { get; }
        
        // Performance metrics
        public double AverageAllocationTime { get; }
        public double AverageDeallocationTime { get; }
        public double TotalTransferredGB { get; }
        public double AverageTransferBandwidthGBps { get; }
    }
}
```

### Advanced Memory Management Examples

```csharp
using UniversalCompute;
using UniversalCompute.Memory.Unified;

// Create context with unified memory management
var memoryOptions = new UnifiedMemoryOptions
{
    PlacementStrategy = MemoryPlacementStrategy.Performance,
    EnablePooling = true,
    MaxPoolSize = 2L * 1024 * 1024 * 1024, // 2GB pool
    PreferUnifiedMemory = true
};

using var context = Context.Create()
    .ConfigureMemoryManager(memoryOptions)
    .EnableMemoryPooling()
    .EnableAllAccelerators();

var memoryManager = context.MemoryManager;

// Allocate unified memory that can be accessed by multiple accelerators
using var unifiedBuffer = memoryManager.AllocateUnified<float>(1_000_000);

// Use across multiple accelerators
var cpuAccelerator = context.CreateCPUAccelerator();
var gpuAccelerator = context.CreateAccelerator(context.GetPreferredDevice(preferGPU: true));

// Page-locked memory for fast transfers
using var fastBuffer = memoryManager.AllocatePageLocked<float>(1_000_000);

// Optimized multi-accelerator workflow
var data = new float[1_000_000];
var plan = memoryManager.OptimizeTransfer(data, 
    new[] { cpuAccelerator, gpuAccelerator }, 
    MemoryAccess.Sequential);

// Execute optimized transfer
plan.Execute();

// Monitor memory statistics
var stats = memoryManager.Statistics;
Console.WriteLine($"Pool hit rate: {stats.PoolHitRate:P2}");
Console.WriteLine($"Average bandwidth: {stats.AverageTransferBandwidthGBps:F2} GB/s");
```

---

## 📈 Performance Monitoring and Profiling

Comprehensive performance monitoring with built-in counters and real-time metrics.

```csharp
namespace UniversalCompute.Performance
{
    public sealed class PerformanceMonitor : IDisposable
    {
        public bool IsEnabled { get; set; }
        public TimeSpan SamplingInterval { get; set; } = TimeSpan.FromMilliseconds(100);
        public PerformanceCounters Counters { get; }
        public event EventHandler<PerformanceEventArgs> ThresholdExceeded;
        
        // Monitoring control
        public void Start();
        public void Stop();
        public void Reset();
        
        // Profiling sessions
        public ProfilingSession StartProfiling(string sessionName);
        public ProfilingResult GetProfilingResult(string sessionName);
        public IReadOnlyList<ProfilingSession> GetActiveSessions();
        
        // Real-time metrics
        public PerformanceSnapshot GetSnapshot();
        public Task<PerformanceSnapshot> GetSnapshotAsync();
        
        // Threshold monitoring
        public void SetThreshold(PerformanceMetric metric, double threshold, ThresholdDirection direction);
        public void RemoveThreshold(PerformanceMetric metric);
        
        public void Dispose();
    }
    
    public sealed class ProfilingSession : IDisposable
    {
        public string Name { get; }
        public DateTime StartTime { get; }
        public TimeSpan Elapsed { get; }
        public bool IsActive { get; }
        public ProfilingOptions Options { get; }
        
        // Kernel profiling
        public KernelProfilingResult ProfileKernel<TDelegate>(
            TDelegate kernel,
            int iterations = 1,
            bool warmup = true) where TDelegate : Delegate;
        
        // Memory profiling
        public MemoryProfilingResult ProfileMemoryOperation(Action operation);
        public Task<MemoryProfilingResult> ProfileMemoryOperationAsync(Func<Task> operation);
        
        // Custom events
        public void RecordEvent(string eventName, Dictionary<string, object> data = null);
        public void StartTimer(string timerName);
        public TimeSpan StopTimer(string timerName);
        
        public void Dispose();
    }
    
    public readonly struct PerformanceCounters
    {
        // Execution metrics
        public long KernelLaunches { get; }
        public TimeSpan TotalKernelTime { get; }
        public TimeSpan AverageKernelTime { get; }
        public double KernelThroughput { get; } // kernels/second
        
        // Memory metrics
        public long BytesAllocated { get; }
        public long BytesFreed { get; }
        public long BytesTransferred { get; }
        public double MemoryBandwidthGBps { get; }
        public double MemoryUtilization { get; } // percentage
        
        // Compute metrics
        public double ComputeUtilization { get; } // percentage
        public long FlopsExecuted { get; }
        public double FlopsPerSecond { get; }
        public double EfficiencyRatio { get; }
        
        // System metrics
        public double PowerConsumptionWatts { get; }
        public double TemperatureCelsius { get; }
        public long PageFaults { get; }
        public long CacheMisses { get; }
    }
    
    public readonly struct PerformanceSnapshot
    {
        public DateTime Timestamp { get; }
        public PerformanceCounters Counters { get; }
        public AcceleratorMetrics[] AcceleratorMetrics { get; }
        public SystemMetrics SystemMetrics { get; }
        
        // Analysis methods
        public bool IsPerformingWell(PerformanceBaseline baseline);
        public PerformanceAnalysis AnalyzeAgainst(PerformanceSnapshot baseline);
        public double GetMetric(PerformanceMetric metric);
    }
    
    public readonly struct KernelProfilingResult
    {
        public string KernelName { get; }
        public int Iterations { get; }
        public TimeSpan TotalTime { get; }
        public TimeSpan AverageTime { get; }
        public TimeSpan MinTime { get; }
        public TimeSpan MaxTime { get; }
        public double StandardDeviation { get; }
        public double ThroughputGBps { get; }
        public double FlopsPerSecond { get; }
        public KernelMetrics DetailedMetrics { get; }
    }
    
    public enum PerformanceMetric
    {
        KernelExecutionTime,
        MemoryBandwidth,
        ComputeUtilization,
        MemoryUtilization,
        PowerConsumption,
        Temperature,
        Throughput
    }
    
    public enum ThresholdDirection
    {
        Above,
        Below
    }
}
```

### Performance Monitoring Examples

```csharp
using UniversalCompute;
using UniversalCompute.Performance;

// Create context with performance monitoring
using var context = Context.Create()
    .EnableProfiling()
    .EnableAllAccelerators();

using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);
var monitor = context.PerformanceMonitor;

// Set up monitoring thresholds
monitor.SetThreshold(PerformanceMetric.MemoryUtilization, 0.9, ThresholdDirection.Above);
monitor.SetThreshold(PerformanceMetric.Temperature, 80.0, ThresholdDirection.Above);

monitor.ThresholdExceeded += (sender, args) =>
{
    Console.WriteLine($"Threshold exceeded: {args.Metric} = {args.Value}");
};

// Start monitoring
monitor.Start();

// Profile kernel performance
using var session = monitor.StartProfiling("VectorAddBenchmark");

// Define and profile kernel
static void VectorAddKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
{
    result[index] = a[index] + b[index];
}

var kernelDelegate = accelerator.LoadAutoGroupedStreamKernel<
    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);

// Allocate test data
using var bufferA = accelerator.Allocate1D<float>(1_000_000);
using var bufferB = accelerator.Allocate1D<float>(1_000_000);
using var result = accelerator.Allocate1D<float>(1_000_000);

// Profile kernel execution
var profilingResult = session.ProfileKernel(() => 
{
    kernelDelegate(1_000_000, bufferA.View, bufferB.View, result.View);
    accelerator.Synchronize();
}, iterations: 100, warmup: true);

Console.WriteLine($"Average execution time: {profilingResult.AverageTime.TotalMicroseconds:F2} μs");
Console.WriteLine($"Throughput: {profilingResult.ThroughputGBps:F2} GB/s");
Console.WriteLine($"Performance deviation: {profilingResult.StandardDeviation:F2}%");

// Get real-time performance snapshot
var snapshot = monitor.GetSnapshot();
Console.WriteLine($"Current memory utilization: {snapshot.Counters.MemoryUtilization:P2}");
Console.WriteLine($"Current compute utilization: {snapshot.Counters.ComputeUtilization:P2}");

// Memory operation profiling
var memoryResult = session.ProfileMemoryOperation(() =>
{
    var data = new float[1_000_000];
    bufferA.CopyFromCPU(data);
});

Console.WriteLine($"Memory transfer: {memoryResult.BandwidthGBps:F2} GB/s");
```

---

## 🔧 Cross-Platform Native AOT

Native Ahead-of-Time compilation support for minimal deployment footprint.

```csharp
namespace UniversalCompute.AOT
{
    public sealed class NativeAOTCompiler : IDisposable
    {
        public CompilationTarget Target { get; }
        public AOTCompilerOptions Options { get; set; }
        public bool IsSupported { get; }
        
        // Compilation methods
        public CompilationResult CompileKernel<TDelegate>(TDelegate kernel, CompilationTarget target);
        public CompilationResult CompileAssembly(Assembly assembly, CompilationTarget target);
        public Task<CompilationResult> CompileKernelAsync<TDelegate>(TDelegate kernel, CompilationTarget target);
        
        // Optimization
        public OptimizedBinary OptimizeForTarget(CompiledKernel kernel, HardwareProfile profile);
        public BinaryMetrics AnalyzeBinary(byte[] binary);
        
        // Deployment
        public DeploymentPackage CreateDeploymentPackage(CompilationResult[] results);
        public void ExtractToDirectory(DeploymentPackage package, string directory);
        
        public void Dispose();
    }
    
    public sealed class AOTCompilerOptions
    {
        public OptimizationLevel Optimization { get; set; } = OptimizationLevel.Release;
        public bool EnableDebugging { get; set; } = false;
        public bool StripSymbols { get; set; } = true;
        public bool EnableLTO { get; set; } = true; // Link-Time Optimization
        public bool MinimizeSize { get; set; } = true;
        public string[] AdditionalFlags { get; set; } = Array.Empty<string>();
        public Dictionary<string, string> Defines { get; set; } = new();
        
        // Platform-specific options
        public WindowsAOTOptions Windows { get; set; } = new();
        public LinuxAOTOptions Linux { get; set; } = new();
        public MacOSAOTOptions MacOS { get; set; } = new();
        public AndroidAOTOptions Android { get; set; } = new();
        public iOSAOTOptions iOS { get; set; } = new();
    }
    
    public readonly struct CompilationTarget
    {
        public Platform Platform { get; }
        public Architecture Architecture { get; }
        public string RuntimeIdentifier { get; }
        public TargetFramework Framework { get; }
        
        // Predefined targets
        public static CompilationTarget Windows_x64 { get; }
        public static CompilationTarget Windows_ARM64 { get; }
        public static CompilationTarget Linux_x64 { get; }
        public static CompilationTarget Linux_ARM64 { get; }
        public static CompilationTarget MacOS_x64 { get; }
        public static CompilationTarget MacOS_ARM64 { get; }
        public static CompilationTarget Android_ARM64 { get; }
        public static CompilationTarget iOS_ARM64 { get; }
        
        // Factory methods
        public static CompilationTarget Create(Platform platform, Architecture arch);
        public static CompilationTarget Parse(string rid);
    }
    
    public sealed class CompilationResult
    {
        public CompilationTarget Target { get; }
        public byte[] Binary { get; }
        public BinaryMetadata Metadata { get; }
        public CompilationDiagnostics Diagnostics { get; }
        public bool IsSuccessful { get; }
        public TimeSpan CompilationTime { get; }
        
        // Binary analysis
        public long BinarySize { get; }
        public Dictionary<string, long> SectionSizes { get; }
        public string[] ExportedSymbols { get; }
        public string[] Dependencies { get; }
        
        // Save/load
        public void SaveToFile(string path);
        public static CompilationResult LoadFromFile(string path);
        public Task SaveToFileAsync(string path);
        public static Task<CompilationResult> LoadFromFileAsync(string path);
    }
    
    public readonly struct HardwareProfile
    {
        public string DeviceName { get; }
        public Architecture Architecture { get; }
        public long MemorySize { get; }
        public int ComputeUnits { get; }
        public double ClockSpeedGHz { get; }
        public string[] SupportedFeatures { get; }
        public Dictionary<string, object> CustomProperties { get; }
        
        // Predefined profiles
        public static HardwareProfile GetForDevice(Device device);
        public static HardwareProfile[] GetCommonProfiles(Platform platform);
        public static HardwareProfile CreateCustom(Dictionary<string, object> properties);
    }
    
    public enum Platform
    {
        Windows,
        Linux,
        MacOS,
        Android,
        iOS,
        FreeBSD,
        WebAssembly
    }
    
    public enum Architecture
    {
        x86,
        x64,
        ARM,
        ARM64,
        RISCV64,
        WASM
    }
}
```

### Native AOT Compilation Examples

```csharp
using UniversalCompute;
using UniversalCompute.AOT;

// Create context with AOT compilation enabled
using var context = Context.Create()
    .EnableNativeAOT()
    .EnableAllAccelerators();

// Configure AOT compiler
var compiler = new NativeAOTCompiler();
compiler.Options = new AOTCompilerOptions
{
    Optimization = OptimizationLevel.Release,
    MinimizeSize = true,
    EnableLTO = true,
    StripSymbols = true
};

// Define kernel for compilation
static void MatrixMultiplyKernel(
    Index2D index,
    ArrayView2D<float, Stride2D.DenseX> a,
    ArrayView2D<float, Stride2D.DenseX> b,
    ArrayView2D<float, Stride2D.DenseX> result)
{
    var row = index.X;
    var col = index.Y;
    var sum = 0.0f;
    
    for (int k = 0; k < a.IntExtent.Y; k++)
    {
        sum += a[row, k] * b[k, col];
    }
    
    result[row, col] = sum;
}

// Compile for multiple targets
var targets = new[]
{
    CompilationTarget.Windows_x64,
    CompilationTarget.Linux_x64,
    CompilationTarget.MacOS_ARM64
};

var compilationTasks = targets.Select(async target =>
{
    var result = await compiler.CompileKernelAsync(MatrixMultiplyKernel, target);
    Console.WriteLine($"{target.RuntimeIdentifier}: {result.BinarySize} bytes, {result.CompilationTime.TotalSeconds:F2}s");
    return result;
});

var results = await Task.WhenAll(compilationTasks);

// Create deployment package
var package = compiler.CreateDeploymentPackage(results);

// Hardware-specific optimization
var device = context.GetPreferredDevice();
var profile = HardwareProfile.GetForDevice(device);
var optimizedBinary = compiler.OptimizeForTarget(results[0].Binary, profile);

Console.WriteLine($"Original size: {results[0].BinarySize} bytes");
Console.WriteLine($"Optimized size: {optimizedBinary.Size} bytes");
Console.WriteLine($"Size reduction: {(1.0 - (double)optimizedBinary.Size / results[0].BinarySize):P2}");

// Deploy to target directory
compiler.ExtractToDirectory(package, "./deployment");
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

### Advanced Integration Example

Comprehensive example showcasing dependency injection, attributes, source generation, unified memory, performance monitoring, and AOT compilation working together:

```csharp
using UniversalCompute;
using UniversalCompute.DependencyInjection;
using UniversalCompute.Attributes;
using UniversalCompute.SourceGeneration;
using UniversalCompute.Memory.Unified;
using UniversalCompute.Performance;
using UniversalCompute.AOT;

// Mark for source generation
[assembly: GenerateKernelLaunchers]
[assembly: GenerateTypeDiscovery]

// Define high-performance kernels with attributes
public static partial class AdvancedKernels
{
    [KernelMethod("OptimizedMatMul")]
    [OptimizedKernel(KernelOptimization.ComputeOptimized, 
        PreferredBlockSize = 256,
        EnableLoopUnrolling = true,
        EnableMemoryCoalescing = true)]
    public static void MatrixMultiplyKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> result,
        [SharedMemory(4096)] ArrayView<float> sharedA,
        [SharedMemory(4096)] ArrayView<float> sharedB)
    {
        var row = index.X;
        var col = index.Y;
        var sum = 0.0f;
        
        // Optimized matrix multiplication with shared memory
        for (int k = 0; k < a.IntExtent.Y; k++)
        {
            sum += a[row, k] * b[k, col];
        }
        
        result[row, col] = sum;
    }
    
    [KernelMethod("VectorReduce")]
    [OptimizedKernel(KernelOptimization.MemoryOptimized)]
    public static void ReductionKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        [SharedMemory(1024, DynamicSize = true)] ArrayView<float> shared)
    {
        // High-performance reduction with shared memory
        var tid = index.X;
        shared[tid] = input[tid];
        
        // Reduction logic
        for (int stride = 1; stride < shared.Length; stride *= 2)
        {
            if (tid % (2 * stride) == 0)
            {
                shared[tid] += shared[tid + stride];
            }
        }
        
        if (tid == 0)
        {
            output[0] = shared[0];
        }
    }
}

// Service interfaces for dependency injection
public interface IComputeService
{
    Task<float[,]> ComputeMatrixMultiplyAsync(float[,] a, float[,] b);
    Task<float> ComputeReductionAsync(float[] data);
}

public interface IPerformanceTracker
{
    void TrackExecution(string operation, TimeSpan duration, double throughput);
    PerformanceReport GenerateReport();
}

// Service implementations
public class AdvancedComputeService : IComputeService
{
    private readonly Accelerator _accelerator;
    private readonly UniversalMemoryManager _memoryManager;
    private readonly IPerformanceTracker _performanceTracker;
    
    public AdvancedComputeService(
        Accelerator accelerator, 
        UniversalMemoryManager memoryManager,
        IPerformanceTracker performanceTracker)
    {
        _accelerator = accelerator;
        _memoryManager = memoryManager;
        _performanceTracker = performanceTracker;
    }
    
    public async Task<float[,]> ComputeMatrixMultiplyAsync(float[,] a, float[,] b)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        var rows = a.GetLength(0);
        var cols = b.GetLength(1);
        var inner = a.GetLength(1);
        
        // Use unified memory for optimal performance
        using var bufferA = _memoryManager.AllocateUnified<float>(rows * inner);
        using var bufferB = _memoryManager.AllocateUnified<float>(inner * cols);
        using var result = _memoryManager.AllocateUnified<float>(rows * cols);
        
        // Copy data with optimization hints
        bufferA.CopyFromCPU(a.Cast<float>().ToArray(), CopyHints.Sequential);
        bufferB.CopyFromCPU(b.Cast<float>().ToArray(), CopyHints.Sequential);
        
        // Use generated launcher (automatically created by source generator)
        await AdvancedKernels.LaunchOptimizedMatMulAsync(
            _accelerator,
            _accelerator.CreateStream(),
            (rows, cols),
            bufferA.Get2DView(rows, inner),
            bufferB.Get2DView(inner, cols),
            result.Get2DView(rows, cols));
        
        var resultArray = new float[rows, cols];
        await result.CopyToCPUAsync(resultArray.Cast<float>().ToArray());
        
        stopwatch.Stop();
        var throughput = (rows * cols * inner * 2.0) / (stopwatch.Elapsed.TotalSeconds * 1e9); // GFLOPS
        _performanceTracker.TrackExecution("MatrixMultiply", stopwatch.Elapsed, throughput);
        
        return resultArray;
    }
    
    public async Task<float> ComputeReductionAsync(float[] data)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        using var input = _memoryManager.AllocatePageLocked<float>(data.Length);
        using var output = _memoryManager.AllocatePageLocked<float>(1);
        
        input.CopyFromCPU(data, CopyHints.Sequential | CopyHints.ReadOnly);
        
        // Use generated launcher with profiling
        var result = await AdvancedKernels.LaunchVectorReduceWithProfilingAsync(
            _accelerator, data.Length, input.View, output.View);
        
        var sum = new float[1];
        await output.CopyToCPUAsync(sum);
        
        stopwatch.Stop();
        var throughput = data.Length / (stopwatch.Elapsed.TotalSeconds * 1e6); // Elements/μs
        _performanceTracker.TrackExecution("Reduction", stopwatch.Elapsed, throughput);
        
        return sum[0];
    }
}

public class RealTimePerformanceTracker : IPerformanceTracker
{
    private readonly List<PerformanceEntry> _entries = new();
    
    public void TrackExecution(string operation, TimeSpan duration, double throughput)
    {
        _entries.Add(new PerformanceEntry(operation, duration, throughput, DateTime.Now));
    }
    
    public PerformanceReport GenerateReport()
    {
        return new PerformanceReport(_entries.ToArray());
    }
}

// Main application using all features
class Program
{
    static async Task Main(string[] args)
    {
        // Configure unified memory management
        var memoryOptions = new UnifiedMemoryOptions
        {
            PlacementStrategy = MemoryPlacementStrategy.Performance,
            EnablePooling = true,
            MaxPoolSize = 4L * 1024 * 1024 * 1024, // 4GB
            PreferUnifiedMemory = true
        };
        
        // Create context with all features enabled
        using var context = Context.Create()
            .ConfigureServices(services =>
            {
                // Register services with proper lifetimes
                services.AddSingleton<IPerformanceTracker, RealTimePerformanceTracker>();
                services.AddScoped<IComputeService, AdvancedComputeService>();
                
                // Register accelerator and memory manager
                services.AddSingleton(provider => 
                    provider.GetRequiredService<Context>().GetPreferredDevice().CreateAccelerator(provider.GetRequiredService<Context>()));
                services.AddSingleton(provider => 
                    provider.GetRequiredService<Context>().MemoryManager);
            })
            .ConfigureMemoryManager(memoryOptions)
            .EnableMemoryPooling()
            .EnableSourceGeneration()
            .EnableProfiling()
            .EnableNativeAOT()
            .EnableAllAccelerators()
            .ToContext();
        
        // Get services through dependency injection
        var serviceProvider = context.ServiceProvider;
        var computeService = serviceProvider.GetRequiredService<IComputeService>();
        var performanceTracker = serviceProvider.GetRequiredService<IPerformanceTracker>();
        var monitor = context.PerformanceMonitor;
        
        // Set up performance monitoring
        monitor.SetThreshold(PerformanceMetric.MemoryUtilization, 0.9, ThresholdDirection.Above);
        monitor.SetThreshold(PerformanceMetric.ComputeUtilization, 0.95, ThresholdDirection.Above);
        
        monitor.ThresholdExceeded += (sender, args) =>
        {
            Console.WriteLine($"⚠️ Performance threshold exceeded: {args.Metric} = {args.Value:F2}");
        };
        
        monitor.Start();
        
        // Perform computations using services
        Console.WriteLine("🚀 Starting advanced computation workflow...");
        
        // Matrix multiplication example
        var matrixA = new float[1024, 1024];
        var matrixB = new float[1024, 1024];
        
        // Initialize matrices with test data
        for (int i = 0; i < 1024; i++)
        {
            for (int j = 0; j < 1024; j++)
            {
                matrixA[i, j] = (float)(i + j);
                matrixB[i, j] = (float)(i * j + 1);
            }
        }
        
        var result = await computeService.ComputeMatrixMultiplyAsync(matrixA, matrixB);
        Console.WriteLine($"✅ Matrix multiplication completed: {result.GetLength(0)}x{result.GetLength(1)}");
        
        // Vector reduction example
        var vectorData = Enumerable.Range(0, 1_000_000).Select(i => (float)i).ToArray();
        var sum = await computeService.ComputeReductionAsync(vectorData);
        Console.WriteLine($"✅ Vector reduction completed: sum = {sum:F2}");
        
        // Get performance statistics
        var snapshot = await monitor.GetSnapshotAsync();
        var report = performanceTracker.GenerateReport();
        
        Console.WriteLine("\n📊 Performance Summary:");
        Console.WriteLine($"Memory utilization: {snapshot.Counters.MemoryUtilization:P2}");
        Console.WriteLine($"Compute utilization: {snapshot.Counters.ComputeUtilization:P2}");
        Console.WriteLine($"Total kernel launches: {snapshot.Counters.KernelLaunches}");
        Console.WriteLine($"Average kernel time: {snapshot.Counters.AverageKernelTime.TotalMicroseconds:F2} μs");
        Console.WriteLine($"Memory bandwidth: {snapshot.Counters.MemoryBandwidthGBps:F2} GB/s");
        
        // AOT compilation for deployment
        if (args.Contains("--compile"))
        {
            Console.WriteLine("\n🔧 Compiling for deployment...");
            
            var compiler = new NativeAOTCompiler();
            compiler.Options = new AOTCompilerOptions
            {
                Optimization = OptimizationLevel.Release,
                MinimizeSize = true,
                EnableLTO = true,
                StripSymbols = true
            };
            
            var targets = new[]
            {
                CompilationTarget.Windows_x64,
                CompilationTarget.Linux_x64,
                CompilationTarget.MacOS_ARM64
            };
            
            var compilationTasks = targets.Select(async target =>
            {
                var compilationResult = await compiler.CompileKernelAsync(
                    AdvancedKernels.MatrixMultiplyKernel, target);
                Console.WriteLine($"📦 {target.RuntimeIdentifier}: {compilationResult.BinarySize:N0} bytes");
                return compilationResult;
            });
            
            var results = await Task.WhenAll(compilationTasks);
            var package = compiler.CreateDeploymentPackage(results);
            
            Console.WriteLine($"✅ Deployment package created with {results.Length} target binaries");
        }
        
        Console.WriteLine("\n🎉 Advanced workflow completed successfully!");
    }
}

// Supporting types
public record PerformanceEntry(string Operation, TimeSpan Duration, double Throughput, DateTime Timestamp);

public class PerformanceReport
{
    private readonly PerformanceEntry[] _entries;
    
    public PerformanceReport(PerformanceEntry[] entries)
    {
        _entries = entries;
    }
    
    public double GetAverageThroughput(string operation) =>
        _entries.Where(e => e.Operation == operation).Average(e => e.Throughput);
    
    public TimeSpan GetAverageDuration(string operation) =>
        TimeSpan.FromTicks((long)_entries.Where(e => e.Operation == operation).Average(e => e.Duration.Ticks));
}
```

This comprehensive example demonstrates:

1. **Dependency Injection**: Services registered with different lifetimes and resolved automatically
2. **Kernel Attributes**: Performance optimization hints and metadata
3. **Source Generation**: Automatic launcher generation for type-safe kernel execution
4. **Unified Memory**: Cross-accelerator memory management with optimization hints
5. **Performance Monitoring**: Real-time metrics, thresholds, and profiling
6. **AOT Compilation**: Cross-platform native binary generation for deployment

---

For more detailed examples and advanced usage patterns, see the [Examples Gallery](Examples-Gallery) and [Quick Start Tutorial](Quick-Start-Tutorial).

---

**📖 Continue exploring:** [Hardware Accelerators](Hardware-Accelerators) | [Performance Tuning](Performance-Tuning) | [Examples Gallery](Examples-Gallery)