// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowsi/ILGPU/blob/main/LICENSE
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// NOTICE: This software is NOT licensed for commercial or production use.
// Change Date: 2029-06-24
// Change License: Apache License, Version 2.0

# ILGPU Technical Architecture Reference

## Executive Summary

ILGPU is a high-performance .NET library providing just-in-time compilation for GPU-accelerated computing across multiple hardware platforms. This document provides comprehensive architectural guidance for enterprise integration, covering system design patterns, performance optimization strategies, and production deployment considerations.

## System Architecture

### Core Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│              ILGPU Universal Computing API                  │
├───────────────┬─────────────────┬───────────────┬───────────┤
│   Memory Mgmt │   Kernel Comp   │   Scheduling  │  Backend  │
│   Universal   │   Multi-target  │   Adaptive    │  Abstract │
│   Manager     │   JIT           │   Resource    │  Layer    │
├───────────────┼─────────────────┼───────────────┼───────────┤
│     CPU       │      CUDA       │    OpenCL     │   Metal   │
│   Backend     │    Backend      │   Backend     │  Backend  │
└───────────────┴─────────────────┴───────────────┴───────────┘
```

### 1. Context and Device Management

The Context serves as the central orchestrator for all ILGPU operations:

```csharp
// Enterprise-grade context initialization
public class ILGPUContextManager : IDisposable
{
    private readonly Context context;
    private readonly Dictionary<string, IAccelerator> accelerators;
    private readonly ILogger logger;
    
    public ILGPUContextManager(ILogger logger, ContextConfiguration config = null)
    {
        this.logger = logger;
        
        // Configure context with enterprise settings
        var builder = Context.CreateBuilder()
            .OptimizationLevel(OptimizationLevel.O2)
            .EnableAssertions(config?.EnableDebugMode ?? false)
            .AllAccelerators();
            
        if (config?.EnableCaching ?? true)
        {
            builder.EnableKernelCaching();
        }
        
        this.context = builder.ToContext();
        this.accelerators = new Dictionary<string, IAccelerator>();
        
        InitializeAccelerators();
    }
    
    private void InitializeAccelerators()
    {
        // Discover and initialize all available accelerators
        foreach (var device in context)
        {
            try
            {
                var accelerator = device.CreateAccelerator(context);
                var key = $"{device.AcceleratorType}_{device.DeviceId}";
                accelerators[key] = accelerator;
                
                logger.LogInformation($"Initialized {device.AcceleratorType} accelerator: {device.Name}");
                logger.LogInformation($"  Memory: {device.MemorySize / (1024 * 1024)} MB");
                logger.LogInformation($"  Max threads per group: {device.MaxNumThreadsPerGroup}");
            }
            catch (Exception ex)
            {
                logger.LogWarning($"Failed to initialize {device.AcceleratorType}: {ex.Message}");
            }
        }
    }
    
    public IAccelerator GetOptimalAccelerator(WorkloadCharacteristics workload)
    {
        return workload.WorkloadType switch
        {
            WorkloadType.Compute => GetAcceleratorByType(AcceleratorType.Cuda) ?? 
                                  GetAcceleratorByType(AcceleratorType.OpenCL) ?? 
                                  GetAcceleratorByType(AcceleratorType.CPU),
            WorkloadType.Memory => GetHighestBandwidthAccelerator(),
            WorkloadType.Debug => GetAcceleratorByType(AcceleratorType.CPU),
            _ => GetPreferredAccelerator()
        };
    }
    
    private IAccelerator GetAcceleratorByType(AcceleratorType type)
    {
        return accelerators.Values.FirstOrDefault(a => a.AcceleratorType == type);
    }
    
    public void Dispose()
    {
        foreach (var accelerator in accelerators.Values)
        {
            accelerator?.Dispose();
        }
        context?.Dispose();
    }
}

public enum WorkloadType
{
    Compute,
    Memory,
    Debug,
    Production
}

public class WorkloadCharacteristics
{
    public WorkloadType WorkloadType { get; set; }
    public int DataSize { get; set; }
    public AccessPattern AccessPattern { get; set; }
    public bool RequiresHighPrecision { get; set; }
    public TimeSpan LatencyRequirement { get; set; }
}
```

### 2. Universal Memory Management Architecture

```csharp
public interface IUniversalMemoryManager : IDisposable
{
    IUniversalBuffer<T> AllocateUniversal<T>(
        long size,
        MemoryPlacement placement = MemoryPlacement.Auto,
        AccessPattern accessPattern = AccessPattern.Unknown) where T : unmanaged;
    
    void EnableMemoryPooling(long poolSize);
    MemoryStatistics GetMemoryStatistics();
    Task<bool> TryDefragmentAsync();
}

public class UniversalMemoryManager : IUniversalMemoryManager
{
    private readonly Context context;
    private readonly Dictionary<AcceleratorType, IAccelerator> accelerators;
    private readonly MemoryPlacementOptimizer optimizer;
    private readonly MemoryPool pool;
    private readonly ILogger logger;
    
    public UniversalMemoryManager(Context context, ILogger logger)
    {
        this.context = context;
        this.logger = logger;
        this.accelerators = DiscoverAccelerators();
        this.optimizer = new MemoryPlacementOptimizer(accelerators);
        this.pool = new MemoryPool(logger);
    }
    
    public IUniversalBuffer<T> AllocateUniversal<T>(
        long size,
        MemoryPlacement placement = MemoryPlacement.Auto,
        AccessPattern accessPattern = AccessPattern.Unknown) where T : unmanaged
    {
        var optimalAccelerator = placement == MemoryPlacement.Auto
            ? optimizer.SelectOptimalAccelerator(size, accessPattern, accelerators.Values)
            : SelectAcceleratorByPlacement(placement);
            
        return new UniversalBuffer<T>(optimalAccelerator, size, placement, accessPattern, logger);
    }
    
    private Dictionary<AcceleratorType, IAccelerator> DiscoverAccelerators()
    {
        var discovered = new Dictionary<AcceleratorType, IAccelerator>();
        
        foreach (var device in context)
        {
            try
            {
                var accelerator = device.CreateAccelerator(context);
                discovered[device.AcceleratorType] = accelerator;
                logger.LogDebug($"Discovered {device.AcceleratorType} accelerator");
            }
            catch (Exception ex)
            {
                logger.LogWarning($"Failed to create accelerator for {device.AcceleratorType}: {ex.Message}");
            }
        }
        
        return discovered;
    }
    
    public void Dispose()
    {
        pool?.Dispose();
        foreach (var accelerator in accelerators.Values)
        {
            accelerator?.Dispose();
        }
    }
}
```

### 3. Kernel Compilation and Execution Pipeline

```csharp
public interface IKernelManager
{
    Task<ICompiledKernel<T>> CompileKernelAsync<T>(
        string kernelName,
        Delegate kernelMethod,
        CompilationOptions options = null) where T : struct;
    
    Task<ExecutionResult> ExecuteAsync<T>(
        ICompiledKernel<T> kernel,
        KernelConfig config,
        params object[] arguments) where T : struct;
}

public class KernelManager : IKernelManager
{
    private readonly IAccelerator accelerator;
    private readonly ConcurrentDictionary<string, object> compiledKernels;
    private readonly ILogger logger;
    private readonly KernelCache cache;
    
    public KernelManager(IAccelerator accelerator, ILogger logger)
    {
        this.accelerator = accelerator;
        this.logger = logger;
        this.compiledKernels = new ConcurrentDictionary<string, object>();
        this.cache = new KernelCache();
    }
    
    public async Task<ICompiledKernel<T>> CompileKernelAsync<T>(
        string kernelName,
        Delegate kernelMethod,
        CompilationOptions options = null) where T : struct
    {
        var cacheKey = GenerateCacheKey(kernelName, kernelMethod, options);
        
        if (compiledKernels.TryGetValue(cacheKey, out var cached))
        {
            return (ICompiledKernel<T>)cached;
        }
        
        logger.LogInformation($"Compiling kernel: {kernelName}");
        
        try
        {
            var kernel = await Task.Run(() => 
            {
                var loadedKernel = accelerator.LoadAutoGroupedStreamKernel<T>(kernelMethod);
                return new CompiledKernel<T>(loadedKernel, kernelName, accelerator, logger);
            });
            
            compiledKernels.TryAdd(cacheKey, kernel);
            logger.LogInformation($"Successfully compiled kernel: {kernelName}");
            
            return kernel;
        }
        catch (Exception ex)
        {
            logger.LogError($"Kernel compilation failed for {kernelName}: {ex.Message}");
            throw new KernelCompilationException($"Failed to compile {kernelName}", ex);
        }
    }
    
    private string GenerateCacheKey(string kernelName, Delegate kernelMethod, CompilationOptions options)
    {
        var hash = new StringBuilder();
        hash.Append(kernelName);
        hash.Append(kernelMethod.Method.GetHashCode());
        hash.Append(accelerator.AcceleratorType);
        
        if (options != null)
        {
            hash.Append(options.OptimizationLevel);
            hash.Append(options.EnableDebugInfo);
        }
        
        return hash.ToString();
    }
}

public class CompiledKernel<T> : ICompiledKernel<T> where T : struct
{
    private readonly object loadedKernel;
    private readonly string kernelName;
    private readonly IAccelerator accelerator;
    private readonly ILogger logger;
    private readonly PerformanceMetrics metrics;
    
    public CompiledKernel(object loadedKernel, string kernelName, IAccelerator accelerator, ILogger logger)
    {
        this.loadedKernel = loadedKernel;
        this.kernelName = kernelName;
        this.accelerator = accelerator;
        this.logger = logger;
        this.metrics = new PerformanceMetrics();
    }
    
    public async Task<ExecutionResult> ExecuteAsync(KernelConfig config, params object[] arguments)
    {
        var stopwatch = Stopwatch.StartNew();
        
        try
        {
            // Execute kernel based on its signature
            await ExecuteKernelDynamic(config, arguments);
            
            // Ensure completion
            accelerator.Synchronize();
            
            stopwatch.Stop();
            
            var result = new ExecutionResult
            {
                Success = true,
                ExecutionTime = stopwatch.Elapsed,
                KernelName = kernelName,
                Configuration = config
            };
            
            metrics.RecordExecution(result);
            
            return result;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            logger.LogError($"Kernel execution failed for {kernelName}: {ex.Message}");
            
            return new ExecutionResult
            {
                Success = false,
                ExecutionTime = stopwatch.Elapsed,
                Error = ex,
                KernelName = kernelName,
                Configuration = config
            };
        }
    }
    
    private async Task ExecuteKernelDynamic(KernelConfig config, object[] arguments)
    {
        // Dynamic kernel execution based on signature
        var method = loadedKernel.GetType().GetMethod("Invoke");
        var parameters = new object[] { config }.Concat(arguments).ToArray();
        
        if (method.ReturnType == typeof(Task))
        {
            await (Task)method.Invoke(loadedKernel, parameters);
        }
        else
        {
            method.Invoke(loadedKernel, parameters);
        }
    }
}
```

## Integration Patterns

### 1. Enterprise Service Integration

```csharp
// Dependency injection integration
public class ILGPUService : IILGPUService, IDisposable
{
    private readonly IUniversalMemoryManager memoryManager;
    private readonly IKernelManager kernelManager;
    private readonly ILogger<ILGPUService> logger;
    private readonly ILGPUContextManager contextManager;
    
    public ILGPUService(
        ILogger<ILGPUService> logger,
        IOptions<ILGPUConfiguration> config)
    {
        this.logger = logger;
        this.contextManager = new ILGPUContextManager(logger, config.Value.Context);
        
        var accelerator = contextManager.GetOptimalAccelerator(new WorkloadCharacteristics
        {
            WorkloadType = WorkloadType.Production
        });
        
        this.memoryManager = new UniversalMemoryManager(contextManager.Context, logger);
        this.kernelManager = new KernelManager(accelerator, logger);
        
        logger.LogInformation($"ILGPU service initialized with {accelerator.AcceleratorType} accelerator");
    }
    
    public async Task<TResult[]> ProcessBatchAsync<TInput, TResult>(
        TInput[] data,
        Func<Index1D, ArrayView<TInput>, ArrayView<TResult>> kernel)
        where TInput : unmanaged
        where TResult : unmanaged
    {
        using var inputBuffer = memoryManager.AllocateUniversal<TInput>(data.Length);
        using var outputBuffer = memoryManager.AllocateUniversal<TResult>(data.Length);
        
        inputBuffer.CopyFromCPU(data);
        
        var compiledKernel = await kernelManager.CompileKernelAsync<Index1D>(
            kernel.Method.Name, kernel);
        
        var config = new KernelConfig(data.Length, 256);
        await compiledKernel.ExecuteAsync(config, inputBuffer.View, outputBuffer.View);
        
        return outputBuffer.GetAsArray1D();
    }
    
    public void Dispose()
    {
        memoryManager?.Dispose();
        contextManager?.Dispose();
    }
}

// Service registration
public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddILGPU(
        this IServiceCollection services,
        Action<ILGPUConfiguration> configure = null)
    {
        services.Configure<ILGPUConfiguration>(config =>
        {
            config.Context = new ContextConfiguration
            {
                EnableDebugMode = false,
                EnableCaching = true,
                OptimizationLevel = OptimizationLevel.O2
            };
            configure?.Invoke(config);
        });
        
        services.AddSingleton<IILGPUService, ILGPUService>();
        
        return services;
    }
}
```

### 2. High-Availability Pattern

```csharp
public class ResilientILGPUService : IILGPUService
{
    private readonly List<IAccelerator> accelerators;
    private readonly ICircuitBreaker circuitBreaker;
    private readonly IHealthChecker healthChecker;
    private readonly ILogger logger;
    
    public ResilientILGPUService(
        ILGPUContextManager contextManager,
        ICircuitBreaker circuitBreaker,
        ILogger logger)
    {
        this.circuitBreaker = circuitBreaker;
        this.logger = logger;
        this.accelerators = InitializeAccelerators(contextManager);
        this.healthChecker = new AcceleratorHealthChecker(accelerators, logger);
        
        // Start health monitoring
        _ = Task.Run(MonitorHealth);
    }
    
    public async Task<TResult[]> ProcessBatchAsync<TInput, TResult>(
        TInput[] data,
        Func<Index1D, ArrayView<TInput>, ArrayView<TResult>> kernel)
        where TInput : unmanaged
        where TResult : unmanaged
    {
        var attempts = 0;
        var maxAttempts = accelerators.Count + 1; // +1 for CPU fallback
        
        while (attempts < maxAttempts)
        {
            var accelerator = GetHealthyAccelerator();
            
            try
            {
                return await circuitBreaker.ExecuteAsync(async () =>
                {
                    return await ExecuteOnAccelerator(accelerator, data, kernel);
                });
            }
            catch (Exception ex)
            {
                logger.LogWarning($"Execution failed on {accelerator.AcceleratorType}: {ex.Message}");
                healthChecker.MarkUnhealthy(accelerator);
                attempts++;
                
                if (attempts < maxAttempts)
                {
                    logger.LogInformation($"Retrying with different accelerator (attempt {attempts + 1})");
                    await Task.Delay(TimeSpan.FromMilliseconds(100 * attempts)); // Exponential backoff
                }
            }
        }
        
        throw new ILGPUExecutionException("All accelerators failed to execute the kernel");
    }
    
    private async Task MonitorHealth()
    {
        while (true)
        {
            await healthChecker.CheckAllAsync();
            await Task.Delay(TimeSpan.FromSeconds(30));
        }
    }
}
```

### 3. Microservices Integration Pattern

```csharp
// gRPC service for ILGPU compute
[Authorize]
public class ComputeService : Compute.ComputeBase
{
    private readonly IILGPUService ilgpuService;
    private readonly ILogger<ComputeService> logger;
    
    public ComputeService(IILGPUService ilgpuService, ILogger<ComputeService> logger)
    {
        this.ilgpuService = ilgpuService;
        this.logger = logger;
    }
    
    public override async Task<ComputeResponse> ProcessArray(
        ComputeRequest request,
        ServerCallContext context)
    {
        try
        {
            var inputData = request.Data.ToArray();
            var operation = ParseOperation(request.Operation);
            
            var result = await ilgpuService.ProcessBatchAsync(inputData, operation);
            
            return new ComputeResponse
            {
                Success = true,
                Data = { result },
                ProcessingTime = DateTime.UtcNow.Ticks
            };
        }
        catch (Exception ex)
        {
            logger.LogError($"Compute request failed: {ex.Message}");
            
            return new ComputeResponse
            {
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }
    
    public override async Task ProcessStream(
        IAsyncStreamReader<StreamComputeRequest> requestStream,
        IServerStreamWriter<StreamComputeResponse> responseStream,
        ServerCallContext context)
    {
        await foreach (var request in requestStream.ReadAllAsync())
        {
            var response = await ProcessStreamChunk(request);
            await responseStream.WriteAsync(response);
        }
    }
}
```

## Performance Architecture

### 1. Adaptive Performance Management

```csharp
public class PerformanceManager
{
    private readonly Dictionary<string, PerformanceProfile> profiles;
    private readonly IMetricsCollector metricsCollector;
    private readonly ILogger logger;
    
    public PerformanceManager(IMetricsCollector metricsCollector, ILogger logger)
    {
        this.metricsCollector = metricsCollector;
        this.logger = logger;
        this.profiles = LoadPerformanceProfiles();
    }
    
    public async Task<KernelConfig> OptimizeConfigurationAsync(
        string kernelName,
        int dataSize,
        IAccelerator accelerator)
    {
        var profile = GetOrCreateProfile(kernelName, accelerator);
        
        if (profile.NeedsCalibration)
        {
            await CalibrateKernel(profile, dataSize, accelerator);
        }
        
        return profile.GetOptimalConfig(dataSize);
    }
    
    private async Task CalibrateKernel(
        PerformanceProfile profile,
        int dataSize,
        IAccelerator accelerator)
    {
        var blockSizes = new[] { 32, 64, 128, 256, 512, 1024 };
        var results = new List<CalibrationResult>();
        
        foreach (var blockSize in blockSizes)
        {
            if (blockSize > accelerator.MaxNumThreadsPerGroup)
                continue;
            
            var config = new KernelConfig(
                (dataSize + blockSize - 1) / blockSize,
                blockSize);
            
            var metrics = await BenchmarkConfiguration(profile.KernelName, config, dataSize, accelerator);
            
            results.Add(new CalibrationResult
            {
                Configuration = config,
                ExecutionTime = metrics.ExecutionTime,
                Occupancy = CalculateOccupancy(config, accelerator)
            });
        }
        
        profile.UpdateCalibration(results);
        await SaveProfile(profile);
    }
    
    private double CalculateOccupancy(KernelConfig config, IAccelerator accelerator)
    {
        var threadsPerBlock = config.GroupSize.Size;
        var maxThreadsPerSM = accelerator.MaxNumThreadsPerMultiprocessor;
        var blocksPerSM = maxThreadsPerSM / threadsPerBlock;
        var activeThreads = Math.Min(blocksPerSM * threadsPerBlock, maxThreadsPerSM);
        
        return (double)activeThreads / maxThreadsPerSM;
    }
}

public class PerformanceProfile
{
    public string KernelName { get; set; }
    public AcceleratorType AcceleratorType { get; set; }
    public Dictionary<int, KernelConfig> OptimalConfigs { get; set; }
    public DateTime LastCalibration { get; set; }
    public bool NeedsCalibration => DateTime.UtcNow - LastCalibration > TimeSpan.FromDays(7);
    
    public KernelConfig GetOptimalConfig(int dataSize)
    {
        // Find the closest calibrated size
        var closestSize = OptimalConfigs.Keys
            .OrderBy(size => Math.Abs(size - dataSize))
            .FirstOrDefault();
        
        if (closestSize == 0)
        {
            // Return default configuration
            return new KernelConfig(Math.Min(dataSize, 65535), 256);
        }
        
        var baseConfig = OptimalConfigs[closestSize];
        
        // Scale configuration for actual data size
        var scaleFactor = (double)dataSize / closestSize;
        var scaledBlocks = (int)(baseConfig.GridSize.Size * scaleFactor);
        
        return new KernelConfig(scaledBlocks, baseConfig.GroupSize.Size);
    }
}
```

### 2. Resource Management

```csharp
public class ResourceManager : IDisposable
{
    private readonly ConcurrentDictionary<string, ResourcePool> pools;
    private readonly SemaphoreSlim semaphore;
    private readonly ILogger logger;
    private readonly Timer cleanupTimer;
    
    public ResourceManager(ILogger logger, int maxConcurrentOperations = 10)
    {
        this.logger = logger;
        this.pools = new ConcurrentDictionary<string, ResourcePool>();
        this.semaphore = new SemaphoreSlim(maxConcurrentOperations);
        this.cleanupTimer = new Timer(CleanupResources, null, TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(5));
    }
    
    public async Task<T> UseResourceAsync<T>(
        string poolName,
        Func<IAccelerator, Task<T>> operation)
    {
        await semaphore.WaitAsync();
        
        try
        {
            var pool = pools.GetOrAdd(poolName, name => new ResourcePool(name, logger));
            var accelerator = await pool.AcquireAsync();
            
            try
            {
                return await operation(accelerator);
            }
            finally
            {
                pool.Release(accelerator);
            }
        }
        finally
        {
            semaphore.Release();
        }
    }
    
    private void CleanupResources(object state)
    {
        foreach (var pool in pools.Values)
        {
            pool.Cleanup();
        }
    }
    
    public void Dispose()
    {
        cleanupTimer?.Dispose();
        semaphore?.Dispose();
        
        foreach (var pool in pools.Values)
        {
            pool.Dispose();
        }
    }
}

public class ResourcePool : IDisposable
{
    private readonly string name;
    private readonly ConcurrentQueue<IAccelerator> available;
    private readonly HashSet<IAccelerator> inUse;
    private readonly ILogger logger;
    private readonly object lockObject = new object();
    
    public ResourcePool(string name, ILogger logger)
    {
        this.name = name;
        this.logger = logger;
        this.available = new ConcurrentQueue<IAccelerator>();
        this.inUse = new HashSet<IAccelerator>();
    }
    
    public async Task<IAccelerator> AcquireAsync()
    {
        if (available.TryDequeue(out var accelerator))
        {
            lock (lockObject)
            {
                inUse.Add(accelerator);
            }
            return accelerator;
        }
        
        // Create new accelerator if none available
        // Implementation would create and configure new accelerator
        accelerator = CreateNewAccelerator();
        
        lock (lockObject)
        {
            inUse.Add(accelerator);
        }
        
        return accelerator;
    }
    
    public void Release(IAccelerator accelerator)
    {
        lock (lockObject)
        {
            if (inUse.Remove(accelerator))
            {
                available.Enqueue(accelerator);
            }
        }
    }
    
    public void Dispose()
    {
        while (available.TryDequeue(out var accelerator))
        {
            accelerator.Dispose();
        }
        
        lock (lockObject)
        {
            foreach (var accelerator in inUse)
            {
                accelerator.Dispose();
            }
            inUse.Clear();
        }
    }
}
```

## Configuration and Deployment

### 1. Configuration Management

```csharp
public class ILGPUConfiguration
{
    public ContextConfiguration Context { get; set; } = new();
    public MemoryConfiguration Memory { get; set; } = new();
    public PerformanceConfiguration Performance { get; set; } = new();
    public MonitoringConfiguration Monitoring { get; set; } = new();
}

public class ContextConfiguration
{
    public bool EnableDebugMode { get; set; } = false;
    public bool EnableCaching { get; set; } = true;
    public OptimizationLevel OptimizationLevel { get; set; } = OptimizationLevel.O2;
    public bool EnableAssertions { get; set; } = false;
    public TimeSpan CompilationTimeout { get; set; } = TimeSpan.FromMinutes(5);
}

public class MemoryConfiguration
{
    public long DefaultPoolSize { get; set; } = 256 * 1024 * 1024; // 256MB
    public bool EnableMemoryPooling { get; set; } = true;
    public bool EnableMemoryCompression { get; set; } = false;
    public double MemoryPressureThreshold { get; set; } = 0.8;
}

public class PerformanceConfiguration
{
    public bool EnablePerformanceMonitoring { get; set; } = true;
    public int MaxConcurrentOperations { get; set; } = 10;
    public TimeSpan HealthCheckInterval { get; set; } = TimeSpan.FromSeconds(30);
    public bool EnableAdaptiveOptimization { get; set; } = true;
}
```

This architecture document provides the foundation for enterprise ILGPU integration. Would you like me to continue with the remaining technical reference documents covering specific implementation patterns, security considerations, and operational guidance?
