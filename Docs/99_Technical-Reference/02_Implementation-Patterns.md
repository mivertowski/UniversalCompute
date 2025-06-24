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

# ILGPU Implementation Patterns

## Kernel Design Patterns

### 1. Universal Kernel Pattern

```csharp
// Template for platform-agnostic kernels
[UniversalKernel]
[NvidiaOptimization(UseTensorCores = true)]
[IntelOptimization(UseAVX512 = true)]
[AppleOptimization(UseNeuralEngine = true)]
public static class UniversalKernels
{
    // Pattern: Data-parallel computation
    public static void VectorOperation(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        float scalar)
    {
        if (index < input.Length)
        {
            output[index] = input[index] * scalar;
        }
    }
    
    // Pattern: Reduction operation
    public static void ParallelReduction(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        var sharedData = SharedMemory.Allocate1D<float>(Group.DimX);
        var tid = Group.IdxX;
        var globalId = Grid.GlobalIndex.X;
        
        // Load data into shared memory
        sharedData[tid] = globalId < input.Length ? input[globalId] : 0.0f;
        Group.Barrier();
        
        // Tree reduction
        for (int stride = Group.DimX / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                sharedData[tid] += sharedData[tid + stride];
            }
            Group.Barrier();
        }
        
        // Write result
        if (tid == 0)
        {
            output[Group.IdxY] = sharedData[0];
        }
    }
    
    // Pattern: Matrix computation with tiling
    public static void TiledMatrixMultiply(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> matrixA,
        ArrayView2D<float, Stride2D.DenseX> matrixB,
        ArrayView2D<float, Stride2D.DenseX> result)
    {
        const int tileSize = 16;
        var sharedA = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
            new Index2D(tileSize, tileSize));
        var sharedB = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
            new Index2D(tileSize, tileSize));
        
        var row = index.Y;
        var col = index.X;
        var localRow = Group.IdxY;
        var localCol = Group.IdxX;
        
        float accumulator = 0.0f;
        var numTiles = (matrixA.Width + tileSize - 1) / tileSize;
        
        for (int tile = 0; tile < numTiles; tile++)
        {
            // Collaborative loading
            var globalCol = tile * tileSize + localCol;
            var globalRow = tile * tileSize + localRow;
            
            sharedA[localRow, localCol] = (row < matrixA.Height && globalCol < matrixA.Width) ?
                matrixA[row, globalCol] : 0.0f;
            
            sharedB[localRow, localCol] = (globalRow < matrixB.Height && col < matrixB.Width) ?
                matrixB[globalRow, col] : 0.0f;
            
            Group.Barrier();
            
            // Compute partial dot product
            for (int k = 0; k < tileSize; k++)
            {
                accumulator += sharedA[localRow, k] * sharedB[k, localCol];
            }
            
            Group.Barrier();
        }
        
        if (row < result.Height && col < result.Width)
        {
            result[row, col] = accumulator;
        }
    }
}
```

### 2. Adaptive Algorithm Pattern

```csharp
public class AdaptiveAlgorithmExecutor
{
    private readonly IAccelerator accelerator;
    private readonly PerformanceProfiler profiler;
    
    public AdaptiveAlgorithmExecutor(IAccelerator accelerator)
    {
        this.accelerator = accelerator;
        this.profiler = new PerformanceProfiler(accelerator);
    }
    
    public async Task<T[]> ExecuteAdaptiveSort<T>(T[] data) 
        where T : unmanaged, IComparable<T>
    {
        var characteristics = AnalyzeDataCharacteristics(data);
        
        return characteristics.DataSize switch
        {
            < 1000 => await ExecuteInsertionSort(data),
            < 100000 => await ExecuteQuickSort(data),
            _ => await ExecuteMergeSort(data)
        };
    }
    
    private DataCharacteristics AnalyzeDataCharacteristics<T>(T[] data) 
        where T : unmanaged, IComparable<T>
    {
        var sampleSize = Math.Min(1000, data.Length);
        var sample = data.Take(sampleSize).ToArray();
        
        var sortedness = CalculateSortedness(sample);
        var uniqueness = CalculateUniqueness(sample);
        
        return new DataCharacteristics
        {
            DataSize = data.Length,
            Sortedness = sortedness,
            Uniqueness = uniqueness,
            ElementSize = Marshal.SizeOf<T>()
        };
    }
    
    private async Task<T[]> ExecuteQuickSort<T>(T[] data) 
        where T : unmanaged, IComparable<T>
    {
        var optimalConfig = await profiler.GetOptimalConfiguration(
            "QuickSort", data.Length);
        
        using var inputBuffer = accelerator.Allocate1D(data);
        using var outputBuffer = accelerator.Allocate1D<T>(data.Length);
        
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<T>, ArrayView<T>>(QuickSortKernel);
        
        kernel(optimalConfig, inputBuffer.View, outputBuffer.View);
        accelerator.Synchronize();
        
        return outputBuffer.GetAsArray1D();
    }
    
    public static void QuickSortKernel<T>(
        Index1D index,
        ArrayView<T> input,
        ArrayView<T> output) where T : unmanaged, IComparable<T>
    {
        // Parallel quicksort implementation
        var length = input.Length;
        if (length <= 1) return;
        
        // Implementation details...
    }
}

public class DataCharacteristics
{
    public int DataSize { get; set; }
    public double Sortedness { get; set; }
    public double Uniqueness { get; set; }
    public int ElementSize { get; set; }
}
```

### 3. Pipeline Pattern for Complex Workloads

```csharp
public interface IPipelineStage<TInput, TOutput>
{
    Task<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default);
    string StageName { get; }
}

public class ComputePipeline<TInput, TOutput> : IDisposable
{
    private readonly List<IPipelineStage<object, object>> stages;
    private readonly IAccelerator accelerator;
    private readonly ILogger logger;
    private readonly MemoryPool memoryPool;
    
    public ComputePipeline(IAccelerator accelerator, ILogger logger)
    {
        this.accelerator = accelerator;
        this.logger = logger;
        this.stages = new List<IPipelineStage<object, object>>();
        this.memoryPool = new MemoryPool(accelerator);
    }
    
    public ComputePipeline<TInput, TOutput> AddStage<TStageInput, TStageOutput>(
        IPipelineStage<TStageInput, TStageOutput> stage)
    {
        stages.Add(new StageWrapper<TStageInput, TStageOutput>(stage));
        return this;
    }
    
    public async Task<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
    {
        object currentData = input;
        var stopwatch = Stopwatch.StartNew();
        
        foreach (var stage in stages)
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            var stageStart = stopwatch.Elapsed;
            currentData = await stage.ExecuteAsync(currentData, cancellationToken);
            var stageTime = stopwatch.Elapsed - stageStart;
            
            logger.LogDebug($"Stage {stage.StageName} completed in {stageTime.TotalMilliseconds:F2}ms");
        }
        
        logger.LogInformation($"Pipeline completed in {stopwatch.Elapsed.TotalMilliseconds:F2}ms");
        return (TOutput)currentData;
    }
    
    public void Dispose()
    {
        memoryPool?.Dispose();
    }
}

// Example: Image Processing Pipeline
public class ImageProcessingPipeline
{
    public static ComputePipeline<byte[], DetectionResult> CreatePipeline(
        IAccelerator accelerator, ILogger logger)
    {
        return new ComputePipeline<byte[], DetectionResult>(accelerator, logger)
            .AddStage(new PreprocessingStage(accelerator))
            .AddStage(new FeatureExtractionStage(accelerator))
            .AddStage(new ObjectDetectionStage(accelerator))
            .AddStage(new PostprocessingStage());
    }
}

public class PreprocessingStage : IPipelineStage<byte[], float[]>
{
    private readonly IAccelerator accelerator;
    
    public string StageName => "Preprocessing";
    
    public PreprocessingStage(IAccelerator accelerator)
    {
        this.accelerator = accelerator;
    }
    
    public async Task<float[]> ExecuteAsync(byte[] input, CancellationToken cancellationToken = default)
    {
        using var inputBuffer = accelerator.Allocate1D(input);
        using var outputBuffer = accelerator.Allocate1D<float>(input.Length);
        
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<float>>(NormalizeKernel);
        
        kernel(input.Length, inputBuffer.View, outputBuffer.View);
        accelerator.Synchronize();
        
        return outputBuffer.GetAsArray1D();
    }
    
    static void NormalizeKernel(Index1D index, ArrayView<byte> input, ArrayView<float> output)
    {
        if (index < input.Length)
        {
            output[index] = input[index] / 255.0f;
        }
    }
}
```

## Memory Management Patterns

### 1. Object Pool Pattern

```csharp
public class ILGPUObjectPool<T> : IDisposable where T : class, IDisposable
{
    private readonly ConcurrentQueue<T> objects;
    private readonly Func<T> objectFactory;
    private readonly int maxPoolSize;
    private readonly ILogger logger;
    private int currentCount;
    
    public ILGPUObjectPool(Func<T> objectFactory, int maxPoolSize = 10, ILogger logger = null)
    {
        this.objectFactory = objectFactory;
        this.maxPoolSize = maxPoolSize;
        this.logger = logger;
        this.objects = new ConcurrentQueue<T>();
    }
    
    public PooledObject<T> Rent()
    {
        if (objects.TryDequeue(out var obj))
        {
            Interlocked.Decrement(ref currentCount);
            return new PooledObject<T>(obj, this);
        }
        
        obj = objectFactory();
        logger?.LogDebug($"Created new {typeof(T).Name} instance");
        return new PooledObject<T>(obj, this);
    }
    
    public void Return(T obj)
    {
        if (obj == null) return;
        
        if (currentCount < maxPoolSize)
        {
            objects.Enqueue(obj);
            Interlocked.Increment(ref currentCount);
        }
        else
        {
            obj.Dispose();
            logger?.LogDebug($"Disposed excess {typeof(T).Name} instance");
        }
    }
    
    public void Dispose()
    {
        while (objects.TryDequeue(out var obj))
        {
            obj.Dispose();
        }
    }
}

public struct PooledObject<T> : IDisposable where T : class, IDisposable
{
    private readonly T obj;
    private readonly ILGPUObjectPool<T> pool;
    private bool returned;
    
    public PooledObject(T obj, ILGPUObjectPool<T> pool)
    {
        this.obj = obj;
        this.pool = pool;
        this.returned = false;
    }
    
    public T Value => returned ? throw new ObjectDisposedException(nameof(PooledObject<T>)) : obj;
    
    public void Dispose()
    {
        if (!returned)
        {
            pool.Return(obj);
            returned = true;
        }
    }
}

// Usage example
public class BufferPoolManager
{
    private readonly ILGPUObjectPool<MemoryBuffer1D<float, Stride1D.Dense>> floatBufferPool;
    private readonly IAccelerator accelerator;
    
    public BufferPoolManager(IAccelerator accelerator)
    {
        this.accelerator = accelerator;
        this.floatBufferPool = new ILGPUObjectPool<MemoryBuffer1D<float, Stride1D.Dense>>(
            () => accelerator.Allocate1D<float>(1024 * 1024), // 1M floats
            maxPoolSize: 5);
    }
    
    public async Task<float[]> ProcessData(float[] input)
    {
        using var buffer = floatBufferPool.Rent();
        
        // Use buffer.Value for computations
        var view = buffer.Value.SubView(0, input.Length);
        view.CopyFromCPU(input);
        
        // Process data...
        
        return view.GetAsArray1D();
    }
}
```

### 2. Memory Streaming Pattern

```csharp
public class StreamingProcessor<T> where T : unmanaged
{
    private readonly IAccelerator accelerator;
    private readonly int chunkSize;
    private readonly ILogger logger;
    
    public StreamingProcessor(IAccelerator accelerator, int chunkSize = 1024 * 1024)
    {
        this.accelerator = accelerator;
        this.chunkSize = chunkSize;
        this.logger = logger;
    }
    
    public async Task<T[]> ProcessLargeDatasetAsync(
        T[] data,
        Func<Index1D, ArrayView<T>, ArrayView<T>> kernel)
    {
        var result = new T[data.Length];
        var numChunks = (data.Length + chunkSize - 1) / chunkSize;
        
        // Use multiple streams for overlapping computation and memory transfer
        using var stream1 = accelerator.CreateStream();
        using var stream2 = accelerator.CreateStream();
        
        var buffers = new[]
        {
            accelerator.Allocate1D<T>(chunkSize),
            accelerator.Allocate1D<T>(chunkSize)
        };
        
        var compiledKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<T>, ArrayView<T>>(kernel);
        
        try
        {
            for (int chunk = 0; chunk < numChunks; chunk++)
            {
                var currentStream = chunk % 2 == 0 ? stream1 : stream2;
                var currentBuffer = buffers[chunk % 2];
                
                var startIdx = chunk * chunkSize;
                var actualChunkSize = Math.Min(chunkSize, data.Length - startIdx);
                
                // Asynchronous memory transfer
                var chunkData = new Span<T>(data, startIdx, actualChunkSize);
                await currentBuffer.CopyFromCPUAsync(currentStream, chunkData.ToArray());
                
                // Execute kernel on current stream
                compiledKernel(
                    currentStream,
                    new KernelConfig(actualChunkSize, 256),
                    currentBuffer.View.SubView(0, actualChunkSize),
                    currentBuffer.View.SubView(0, actualChunkSize));
                
                // Asynchronous result retrieval
                var chunkResult = await currentBuffer.GetAsArray1DAsync(currentStream, 0, actualChunkSize);
                Array.Copy(chunkResult, 0, result, startIdx, actualChunkSize);
                
                logger?.LogDebug($"Processed chunk {chunk + 1}/{numChunks}");
            }
            
            // Synchronize all streams
            stream1.Synchronize();
            stream2.Synchronize();
            
            return result;
        }
        finally
        {
            foreach (var buffer in buffers)
            {
                buffer.Dispose();
            }
        }
    }
}
```

## Error Handling and Resilience Patterns

### 1. Circuit Breaker Pattern

```csharp
public class ILGPUCircuitBreaker
{
    private readonly int threshold;
    private readonly TimeSpan timeout;
    private readonly ILogger logger;
    private int failureCount;
    private DateTime lastFailureTime;
    private CircuitState state;
    private readonly object lockObject = new object();
    
    public ILGPUCircuitBreaker(int threshold = 5, TimeSpan? timeout = null, ILogger logger = null)
    {
        this.threshold = threshold;
        this.timeout = timeout ?? TimeSpan.FromMinutes(1);
        this.logger = logger;
        this.state = CircuitState.Closed;
    }
    
    public async Task<T> ExecuteAsync<T>(Func<Task<T>> operation)
    {
        if (state == CircuitState.Open)
        {
            if (DateTime.UtcNow - lastFailureTime < timeout)
            {
                throw new CircuitBreakerOpenException("Circuit breaker is open");
            }
            
            // Transition to half-open state
            lock (lockObject)
            {
                if (state == CircuitState.Open)
                {
                    state = CircuitState.HalfOpen;
                    logger?.LogInformation("Circuit breaker transitioning to half-open state");
                }
            }
        }
        
        try
        {
            var result = await operation();
            
            // Success - reset circuit breaker
            if (state == CircuitState.HalfOpen)
            {
                lock (lockObject)
                {
                    state = CircuitState.Closed;
                    failureCount = 0;
                    logger?.LogInformation("Circuit breaker closed after successful operation");
                }
            }
            
            return result;
        }
        catch (Exception ex)
        {
            RecordFailure(ex);
            throw;
        }
    }
    
    private void RecordFailure(Exception ex)
    {
        lock (lockObject)
        {
            failureCount++;
            lastFailureTime = DateTime.UtcNow;
            
            if (failureCount >= threshold || state == CircuitState.HalfOpen)
            {
                state = CircuitState.Open;
                logger?.LogWarning($"Circuit breaker opened after {failureCount} failures. Last error: {ex.Message}");
            }
        }
    }
    
    public CircuitState State => state;
}

public enum CircuitState
{
    Closed,
    Open,
    HalfOpen
}

public class CircuitBreakerOpenException : Exception
{
    public CircuitBreakerOpenException(string message) : base(message) { }
}
```

### 2. Retry Pattern with Exponential Backoff

```csharp
public class ILGPURetryPolicy
{
    private readonly int maxRetries;
    private readonly TimeSpan baseDelay;
    private readonly double backoffMultiplier;
    private readonly ILogger logger;
    
    public ILGPURetryPolicy(
        int maxRetries = 3,
        TimeSpan? baseDelay = null,
        double backoffMultiplier = 2.0,
        ILogger logger = null)
    {
        this.maxRetries = maxRetries;
        this.baseDelay = baseDelay ?? TimeSpan.FromMilliseconds(100);
        this.backoffMultiplier = backoffMultiplier;
        this.logger = logger;
    }
    
    public async Task<T> ExecuteAsync<T>(Func<Task<T>> operation)
    {
        for (int attempt = 0; attempt <= maxRetries; attempt++)
        {
            try
            {
                return await operation();
            }
            catch (Exception ex) when (ShouldRetry(ex, attempt))
            {
                var delay = CalculateDelay(attempt);
                logger?.LogWarning($"Attempt {attempt + 1} failed: {ex.Message}. Retrying in {delay.TotalMilliseconds}ms");
                
                if (attempt < maxRetries)
                {
                    await Task.Delay(delay);
                }
            }
        }
        
        throw new InvalidOperationException($"Operation failed after {maxRetries + 1} attempts");
    }
    
    private bool ShouldRetry(Exception ex, int attempt)
    {
        if (attempt >= maxRetries) return false;
        
        return ex switch
        {
            OutOfMemoryException => true,
            AcceleratorException => true,
            TimeoutException => true,
            CompilationException => false, // Don't retry compilation errors
            _ => false
        };
    }
    
    private TimeSpan CalculateDelay(int attempt)
    {
        var delay = TimeSpan.FromTicks((long)(baseDelay.Ticks * Math.Pow(backoffMultiplier, attempt)));
        
        // Add jitter to prevent thundering herd
        var jitter = Random.Shared.NextDouble() * 0.1 + 0.95; // 95-105% of calculated delay
        
        return TimeSpan.FromTicks((long)(delay.Ticks * jitter));
    }
}
```

### 3. Fallback Pattern

```csharp
public class FallbackExecutor
{
    private readonly List<IAccelerator> accelerators;
    private readonly ILogger logger;
    private readonly ILGPUCircuitBreaker circuitBreaker;
    
    public FallbackExecutor(IEnumerable<IAccelerator> accelerators, ILogger logger)
    {
        this.accelerators = accelerators.OrderBy(GetAcceleratorPriority).ToList();
        this.logger = logger;
        this.circuitBreaker = new ILGPUCircuitBreaker(logger: logger);
    }
    
    public async Task<T[]> ExecuteWithFallback<T>(
        T[] data,
        Func<Index1D, ArrayView<T>, ArrayView<T>> kernel) where T : unmanaged
    {
        foreach (var accelerator in accelerators)
        {
            try
            {
                return await circuitBreaker.ExecuteAsync(async () =>
                {
                    return await ExecuteOnAccelerator(accelerator, data, kernel);
                });
            }
            catch (CircuitBreakerOpenException)
            {
                logger.LogWarning($"Circuit breaker open for {accelerator.AcceleratorType}, trying next accelerator");
                continue;
            }
            catch (Exception ex)
            {
                logger.LogError($"Execution failed on {accelerator.AcceleratorType}: {ex.Message}");
                
                // For critical errors, don't try other accelerators
                if (ex is CompilationException)
                {
                    throw;
                }
                
                continue;
            }
        }
        
        throw new InvalidOperationException("All accelerators failed to execute the operation");
    }
    
    private async Task<T[]> ExecuteOnAccelerator<T>(
        IAccelerator accelerator,
        T[] data,
        Func<Index1D, ArrayView<T>, ArrayView<T>> kernel) where T : unmanaged
    {
        using var inputBuffer = accelerator.Allocate1D(data);
        using var outputBuffer = accelerator.Allocate1D<T>(data.Length);
        
        var compiledKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<T>, ArrayView<T>>(kernel);
        
        compiledKernel(data.Length, inputBuffer.View, outputBuffer.View);
        accelerator.Synchronize();
        
        return outputBuffer.GetAsArray1D();
    }
    
    private int GetAcceleratorPriority(IAccelerator accelerator)
    {
        return accelerator.AcceleratorType switch
        {
            AcceleratorType.Cuda => 1,
            AcceleratorType.OpenCL => 2,
            AcceleratorType.Velocity => 3,
            AcceleratorType.CPU => 4,
            _ => 5
        };
    }
}
```

## Testing Patterns

### 1. Property-Based Testing

```csharp
public class ILGPUPropertyTests
{
    [Property]
    public bool VectorAdditionIsCommutative(float[] a, float[] b)
    {
        Assume.That(a != null && b != null && a.Length == b.Length);
        Assume.That(a.Length > 0 && a.Length <= 10000);
        
        using var context = Context.CreateDefault();
        using var accelerator = context.CreateCPUAccelerator(0);
        
        var result1 = VectorAdd(accelerator, a, b);
        var result2 = VectorAdd(accelerator, b, a);
        
        return result1.SequenceEqual(result2);
    }
    
    [Property]
    public bool MatrixMultiplicationIsAssociative(
        float[,] a, float[,] b, float[,] c)
    {
        Assume.That(CanMultiply(a, b) && CanMultiply(b, c));
        
        using var context = Context.CreateDefault();
        using var accelerator = context.CreateCPUAccelerator(0);
        
        // (A * B) * C
        var ab = MatrixMultiply(accelerator, a, b);
        var abc1 = MatrixMultiply(accelerator, ab, c);
        
        // A * (B * C)
        var bc = MatrixMultiply(accelerator, b, c);
        var abc2 = MatrixMultiply(accelerator, a, bc);
        
        return MatricesAreEqual(abc1, abc2, tolerance: 1e-5f);
    }
    
    private static float[] VectorAdd(IAccelerator accelerator, float[] a, float[] b)
    {
        using var bufferA = accelerator.Allocate1D(a);
        using var bufferB = accelerator.Allocate1D(b);
        using var result = accelerator.Allocate1D<float>(a.Length);
        
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);
        
        kernel(a.Length, bufferA.View, bufferB.View, result.View);
        accelerator.Synchronize();
        
        return result.GetAsArray1D();
    }
    
    static void VectorAddKernel(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> result)
    {
        if (index < result.Length)
        {
            result[index] = a[index] + b[index];
        }
    }
}
```

### 2. Performance Regression Testing

```csharp
public class PerformanceRegressionTests
{
    private readonly Dictionary<string, PerformanceBaseline> baselines;
    
    public PerformanceRegressionTests()
    {
        baselines = LoadPerformanceBaselines();
    }
    
    [Fact]
    public async Task VectorAddition_PerformanceWithinBounds()
    {
        const int dataSize = 1024 * 1024;
        const int iterations = 100;
        
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);
        
        var data = GenerateTestData(dataSize);
        
        var stopwatch = Stopwatch.StartNew();
        
        for (int i = 0; i < iterations; i++)
        {
            var result = await ExecuteVectorAddition(accelerator, data, data);
        }
        
        stopwatch.Stop();
        
        var avgTime = stopwatch.Elapsed.TotalMilliseconds / iterations;
        var baseline = baselines["VectorAddition"];
        
        Assert.True(avgTime <= baseline.MaxTime * 1.1, // Allow 10% regression
            $"Performance regression detected. Current: {avgTime:F2}ms, Baseline: {baseline.MaxTime:F2}ms");
        
        var bandwidth = CalculateBandwidth(dataSize, avgTime);
        Assert.True(bandwidth >= baseline.MinBandwidth * 0.9, // Allow 10% bandwidth drop
            $"Bandwidth regression detected. Current: {bandwidth:F2} GB/s, Baseline: {baseline.MinBandwidth:F2} GB/s");
    }
    
    [Theory]
    [InlineData(32)]
    [InlineData(128)]
    [InlineData(512)]
    [InlineData(1024)]
    public async Task MatrixMultiplication_ScalesCorrectly(int matrixSize)
    {
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);
        
        var matrixA = GenerateMatrix(matrixSize, matrixSize);
        var matrixB = GenerateMatrix(matrixSize, matrixSize);
        
        var stopwatch = Stopwatch.StartNew();
        var result = await ExecuteMatrixMultiplication(accelerator, matrixA, matrixB);
        stopwatch.Stop();
        
        var operations = 2L * matrixSize * matrixSize * matrixSize; // Multiply-add operations
        var gflops = operations / (stopwatch.Elapsed.TotalSeconds * 1e9);
        
        var expectedMinGflops = GetExpectedPerformance(matrixSize);
        
        Assert.True(gflops >= expectedMinGflops,
            $"Performance below expected for size {matrixSize}. Got {gflops:F2} GFLOPS, expected >= {expectedMinGflops:F2}");
    }
    
    private Dictionary<string, PerformanceBaseline> LoadPerformanceBaselines()
    {
        // Load from configuration or previous test runs
        return new Dictionary<string, PerformanceBaseline>
        {
            ["VectorAddition"] = new PerformanceBaseline
            {
                MaxTime = 2.0, // milliseconds
                MinBandwidth = 100.0 // GB/s
            }
        };
    }
}

public class PerformanceBaseline
{
    public double MaxTime { get; set; }
    public double MinBandwidth { get; set; }
    public DateTime LastUpdated { get; set; }
}
```

This implementation patterns document covers the essential design patterns, memory management strategies, error handling approaches, and testing methodologies for enterprise ILGPU development. Would you like me to continue with the remaining technical reference documents covering security, monitoring, and deployment patterns?
