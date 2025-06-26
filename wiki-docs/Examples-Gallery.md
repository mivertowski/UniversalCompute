# Examples Gallery

Comprehensive collection of UniversalCompute examples for .NET 9.0 with preview language features, from basic concepts to advanced use cases.

## 🎯 Getting Started Examples (.NET 9.0)

### Basic Vector Operations

Simple vector addition demonstrating core .NET 9.0 concepts with preview language features.

```csharp
using UniversalCompute;
using UniversalCompute.Runtime;

class VectorAddExample
{
    static void Main()
    {
        // Create context and accelerator
        using var context = Context.Create().EnableAllAccelerators();
        using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);
        
        // Vector addition kernel
        static void VectorAddKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
        {
            result[index] = a[index] + b[index];
        }
        
        // Allocate memory and execute
        const int size = 1000;
        using var bufferA = accelerator.Allocate1D<float>(size);
        using var bufferB = accelerator.Allocate1D<float>(size);
        using var bufferResult = accelerator.Allocate1D<float>(size);
        
        // Initialize data
        var dataA = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
        var dataB = Enumerable.Range(0, size).Select(i => (float)i * 2).ToArray();
        
        bufferA.CopyFromCPU(dataA);
        bufferB.CopyFromCPU(dataB);
        
        // Execute kernel
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);
        kernel(size, bufferA.View, bufferB.View, bufferResult.View);
        accelerator.Synchronize();
        
        // Get results
        var result = bufferResult.GetAsArray1D();
        Console.WriteLine($"First 5 results: [{string.Join(", ", result.Take(5))}]");
    }
}
```

### Matrix Multiplication

2D matrix operations using UniversalCompute.

```csharp
using UniversalCompute;
using UniversalCompute.Runtime;

class MatrixMultiplyExample
{
    static void Main()
    {
        using var context = Context.Create().EnableAllAccelerators();
        using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);
        
        const int N = 512;
        
        // Matrix multiplication kernel
        static void MatMulKernel(Index2D index, ArrayView2D<float, Stride2D.DenseX> a, ArrayView2D<float, Stride2D.DenseX> b, ArrayView2D<float, Stride2D.DenseX> c)
        {
            var row = index.X;
            var col = index.Y;
            
            if (row >= a.Extent.X || col >= b.Extent.Y)
                return;
                
            float sum = 0;
            for (int k = 0; k < a.Extent.Y; k++)
            {
                sum += a[row, k] * b[k, col];
            }
            c[row, col] = sum;
        }
        
        // Allocate matrices
        using var matrixA = accelerator.Allocate2D<float>(new LongIndex2D(N, N));
        using var matrixB = accelerator.Allocate2D<float>(new LongIndex2D(N, N));
        using var matrixC = accelerator.Allocate2D<float>(new LongIndex2D(N, N));
        
        // Initialize with random data
        var random = new Random(42);
        var dataA = new float[N, N];
        var dataB = new float[N, N];
        
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                dataA[i, j] = random.NextSingle();
                dataB[i, j] = random.NextSingle();
            }
        }
        
        matrixA.CopyFromCPU(dataA);
        matrixB.CopyFromCPU(dataB);
        
        // Execute matrix multiplication
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(MatMulKernel);
        
        var stopwatch = Stopwatch.StartNew();
        kernel(matrixC.Extent, matrixA.View, matrixB.View, matrixC.View);
        accelerator.Synchronize();
        stopwatch.Stop();
        
        Console.WriteLine($"Matrix multiplication ({N}x{N}): {stopwatch.ElapsedMilliseconds} ms");
        
        // Calculate GFLOPS
        var gflops = (2.0 * N * N * N) / (stopwatch.ElapsedMilliseconds / 1000.0) / 1e9;
        Console.WriteLine($"Performance: {gflops:F2} GFLOPS");
    }
}
```

## 🔬 Scientific Computing Examples (.NET 9.0 Optimized)

### FFT Signal Processing

Fast Fourier Transform for signal analysis.

```csharp
using UniversalCompute;
using UniversalCompute.FFT;
using System.Numerics;

class FFTSignalProcessingExample
{
    static void Main()
    {
        using var context = Context.Create().EnableAllAccelerators();
        using var fftManager = new FFTManager(context);
        using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);
        
        // Generate test signal: 50Hz + 120Hz sine waves with noise
        const int sampleRate = 1000; // Hz
        const int duration = 2;       // seconds
        const int N = sampleRate * duration;
        
        var signal = new Complex[N];
        var time = new double[N];
        
        for (int i = 0; i < N; i++)
        {
            time[i] = i / (double)sampleRate;
            var t = time[i];
            
            // Signal: 50Hz + 120Hz + noise
            var amplitude1 = Math.Sin(2 * Math.PI * 50 * t);
            var amplitude2 = 0.5 * Math.Sin(2 * Math.PI * 120 * t);
            var noise = 0.1 * (Random.Shared.NextDouble() - 0.5);
            
            signal[i] = new Complex(amplitude1 + amplitude2 + noise, 0);
        }
        
        // Allocate GPU memory
        using var inputBuffer = accelerator.Allocate1D<Complex>(N);
        using var outputBuffer = accelerator.Allocate1D<Complex>(N);
        
        // Copy signal to GPU and perform FFT
        inputBuffer.CopyFromCPU(signal);
        
        var stopwatch = Stopwatch.StartNew();
        fftManager.FFT1D(inputBuffer.View, outputBuffer.View, forward: true);
        stopwatch.Stop();
        
        // Get frequency domain result
        var fftResult = outputBuffer.GetAsArray1D();
        
        Console.WriteLine($"FFT computation time: {stopwatch.ElapsedMilliseconds} ms");
        
        // Find peak frequencies
        var frequencies = new double[N / 2];
        var magnitudes = new double[N / 2];
        
        for (int i = 0; i < N / 2; i++)
        {
            frequencies[i] = i * sampleRate / (double)N;
            magnitudes[i] = fftResult[i].Magnitude;
        }
        
        // Find top 3 frequency components
        var peaks = frequencies.Zip(magnitudes, (freq, mag) => new { Frequency = freq, Magnitude = mag })
                              .Where(x => x.Frequency > 1) // Ignore DC component
                              .OrderByDescending(x => x.Magnitude)
                              .Take(3)
                              .ToList();
        
        Console.WriteLine("Top frequency components:");
        foreach (var peak in peaks)
        {
            Console.WriteLine($"  {peak.Frequency:F1} Hz: {peak.Magnitude:F2}");
        }
    }
}
```

### Monte Carlo Pi Estimation

Parallel random number generation for statistical computing.

```csharp
using UniversalCompute;
using UniversalCompute.Runtime;

class MonteCarloExample
{
    static void Main()
    {
        using var context = Context.Create().EnableAllAccelerators();
        using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);
        
        const int numSamples = 10_000_000;
        
        // Monte Carlo kernel for Pi estimation
        static void MonteCarloKernel(Index1D index, ArrayView<int> results, int seed)
        {
            // Simple linear congruential generator
            var rng = (uint)(seed + index);
            
            int insideCircle = 0;
            const int samplesPerThread = 1000;
            
            for (int i = 0; i < samplesPerThread; i++)
            {
                // Generate pseudo-random numbers
                rng = rng * 1664525u + 1013904223u;
                var x = (rng / (float)uint.MaxValue) * 2 - 1; // [-1, 1]
                
                rng = rng * 1664525u + 1013904223u;
                var y = (rng / (float)uint.MaxValue) * 2 - 1; // [-1, 1]
                
                if (x * x + y * y <= 1.0f)
                    insideCircle++;
            }
            
            results[index] = insideCircle;
        }
        
        // Calculate number of threads needed
        const int samplesPerThread = 1000;
        var numThreads = numSamples / samplesPerThread;
        
        // Allocate result buffer
        using var resultBuffer = accelerator.Allocate1D<int>(numThreads);
        
        // Execute Monte Carlo simulation
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(MonteCarloKernel);
        
        var stopwatch = Stopwatch.StartNew();
        kernel(numThreads, resultBuffer.View, Environment.TickCount);
        accelerator.Synchronize();
        stopwatch.Stop();
        
        // Collect results and estimate Pi
        var results = resultBuffer.GetAsArray1D();
        var totalInside = results.Sum();
        var piEstimate = 4.0 * totalInside / numSamples;
        
        Console.WriteLine($"Monte Carlo Pi Estimation");
        Console.WriteLine($"Samples: {numSamples:N0}");
        Console.WriteLine($"Threads: {numThreads:N0}");
        Console.WriteLine($"Computation time: {stopwatch.ElapsedMilliseconds} ms");
        Console.WriteLine($"Pi estimate: {piEstimate:F6}");
        Console.WriteLine($"Error: {Math.Abs(piEstimate - Math.PI):F6}");
        Console.WriteLine($"Samples per second: {numSamples / (stopwatch.ElapsedMilliseconds / 1000.0):F0}");
    }
}
```

## 🧠 Machine Learning Examples (.NET 9.0 Enhanced)

### Neural Network Forward Pass

Simple neural network inference using UniversalCompute.

```csharp
using UniversalCompute;
using UniversalCompute.Runtime;
using UniversalCompute.Core;

class NeuralNetworkExample
{
    static void Main()
    {
        using var context = Context.Create().EnableAllAccelerators();
        using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);
        
        // Network dimensions
        const int inputSize = 784;   // 28x28 image
        const int hiddenSize = 128;
        const int outputSize = 10;   // 10 classes
        const int batchSize = 32;
        
        // Dense layer forward pass kernel
        static void DenseForwardKernel(Index2D index, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView<float> bias, ArrayView2D<float, Stride2D.DenseX> output)
        {
            var batch = index.X;
            var neuron = index.Y;
            
            if (batch >= input.Extent.X || neuron >= weights.Extent.Y)
                return;
            
            float sum = bias[neuron];
            for (int i = 0; i < weights.Extent.X; i++)
            {
                sum += input[batch, i] * weights[i, neuron];
            }
            
            // ReLU activation
            output[batch, neuron] = Math.Max(0, sum);
        }
        
        // Allocate network parameters and data
        using var inputData = accelerator.Allocate2D<float>(new LongIndex2D(batchSize, inputSize));
        using var weights1 = accelerator.Allocate2D<float>(new LongIndex2D(inputSize, hiddenSize));
        using var bias1 = accelerator.Allocate1D<float>(hiddenSize);
        using var hidden = accelerator.Allocate2D<float>(new LongIndex2D(batchSize, hiddenSize));
        using var weights2 = accelerator.Allocate2D<float>(new LongIndex2D(hiddenSize, outputSize));
        using var bias2 = accelerator.Allocate1D<float>(outputSize);
        using var output = accelerator.Allocate2D<float>(new LongIndex2D(batchSize, outputSize));
        
        // Initialize with random weights
        var random = new Random(42);
        
        // Input data (random for demo)
        var inputArray = new float[batchSize, inputSize];
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputSize; i++)
            {
                inputArray[b, i] = random.NextSingle();
            }
        }
        
        // Weights and biases
        var weights1Array = new float[inputSize, hiddenSize];
        var bias1Array = new float[hiddenSize];
        var weights2Array = new float[hiddenSize, outputSize];
        var bias2Array = new float[outputSize];
        
        // Xavier initialization
        var scale1 = Math.Sqrt(2.0 / inputSize);
        var scale2 = Math.Sqrt(2.0 / hiddenSize);
        
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                weights1Array[i, j] = (float)(random.NextGaussian() * scale1);
            }
        }
        
        for (int i = 0; i < hiddenSize; i++)
        {
            bias1Array[i] = 0;
            for (int j = 0; j < outputSize; j++)
            {
                weights2Array[i, j] = (float)(random.NextGaussian() * scale2);
            }
        }
        
        for (int i = 0; i < outputSize; i++)
        {
            bias2Array[i] = 0;
        }
        
        // Copy data to GPU
        inputData.CopyFromCPU(inputArray);
        weights1.CopyFromCPU(weights1Array);
        bias1.CopyFromCPU(bias1Array);
        weights2.CopyFromCPU(weights2Array);
        bias2.CopyFromCPU(bias2Array);
        
        // Load kernels
        var denseKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView2D<float, Stride2D.DenseX>>(DenseForwardKernel);
        
        // Forward pass timing
        var stopwatch = Stopwatch.StartNew();
        
        // Layer 1: Input -> Hidden
        denseKernel(hidden.Extent, inputData.View, weights1.View, bias1.View, hidden.View);
        
        // Layer 2: Hidden -> Output
        denseKernel(output.Extent, hidden.View, weights2.View, bias2.View, output.View);
        
        accelerator.Synchronize();
        stopwatch.Stop();
        
        // Get predictions
        var predictions = output.GetAsArray2D();
        
        Console.WriteLine($"Neural Network Forward Pass");
        Console.WriteLine($"Input size: {inputSize}");
        Console.WriteLine($"Hidden size: {hiddenSize}");
        Console.WriteLine($"Output size: {outputSize}");
        Console.WriteLine($"Batch size: {batchSize}");
        Console.WriteLine($"Inference time: {stopwatch.ElapsedMilliseconds:F2} ms");
        Console.WriteLine($"Throughput: {batchSize / (stopwatch.ElapsedMilliseconds / 1000.0):F0} samples/sec");
        
        // Show first prediction
        Console.WriteLine($"First sample predictions: [{string.Join(", ", Enumerable.Range(0, outputSize).Select(i => predictions[0, i].ToString("F3")))}]");
    }
}

// Extension method for Gaussian random numbers
public static class RandomExtensions
{
    public static double NextGaussian(this Random random, double mean = 0, double stdDev = 1)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }
}
```

## 🎮 Graphics and Image Processing (.NET 9.0)

### Image Convolution

2D convolution for image filtering and computer vision.

```csharp
using UniversalCompute;
using UniversalCompute.Runtime;

class ImageConvolutionExample
{
    static void Main()
    {
        using var context = Context.Create().EnableAllAccelerators();
        using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);
        
        // Image dimensions
        const int width = 1920;
        const int height = 1080;
        const int channels = 3; // RGB
        
        // Convolution kernel (3x3 edge detection)
        var kernel = new float[3, 3]
        {
            { -1, -1, -1 },
            { -1,  8, -1 },
            { -1, -1, -1 }
        };
        
        // 2D convolution kernel
        static void ConvolutionKernel(Index2D index, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> filter, ArrayView2D<float, Stride2D.DenseX> output)
        {
            var x = index.X;
            var y = index.Y;
            
            if (x >= output.Extent.X || y >= output.Extent.Y)
                return;
            
            var filterSize = filter.Extent.X; // Assuming square filter
            var halfFilter = filterSize / 2;
            
            float sum = 0;
            
            for (int fx = 0; fx < filterSize; fx++)
            {
                for (int fy = 0; fy < filterSize; fy++)
                {
                    var ix = x + fx - halfFilter;
                    var iy = y + fy - halfFilter;
                    
                    // Handle boundaries with zero-padding
                    if (ix >= 0 && ix < input.Extent.X && iy >= 0 && iy < input.Extent.Y)
                    {
                        sum += input[ix, iy] * filter[fx, fy];
                    }
                }
            }
            
            output[x, y] = sum;
        }
        
        // Allocate image data
        using var inputImage = accelerator.Allocate2D<float>(new LongIndex2D(height, width));
        using var filterBuffer = accelerator.Allocate2D<float>(new LongIndex2D(3, 3));
        using var outputImage = accelerator.Allocate2D<float>(new LongIndex2D(height, width));
        
        // Generate test image (gradient pattern)
        var imageData = new float[height, width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Create a pattern with edges
                imageData[y, x] = (float)(Math.Sin(x * 0.01) * Math.Cos(y * 0.01) + 
                                        Math.Sin(x * 0.005) * Math.Sin(y * 0.005));
            }
        }
        
        // Copy data to GPU
        inputImage.CopyFromCPU(imageData);
        filterBuffer.CopyFromCPU(kernel);
        
        // Execute convolution
        var convKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(ConvolutionKernel);
        
        var stopwatch = Stopwatch.StartNew();
        convKernel(outputImage.Extent, inputImage.View, filterBuffer.View, outputImage.View);
        accelerator.Synchronize();
        stopwatch.Stop();
        
        // Calculate performance metrics
        var totalPixels = width * height;
        var megapixels = totalPixels / 1_000_000.0;
        var pixelsPerSecond = totalPixels / (stopwatch.ElapsedMilliseconds / 1000.0);
        
        Console.WriteLine($"Image Convolution Performance");
        Console.WriteLine($"Image size: {width}x{height} ({megapixels:F1} MP)");
        Console.WriteLine($"Filter size: 3x3");
        Console.WriteLine($"Processing time: {stopwatch.ElapsedMilliseconds} ms");
        Console.WriteLine($"Throughput: {pixelsPerSecond / 1_000_000:F1} MP/s");
        
        // Optionally save result (simplified output)
        var result = outputImage.GetAsArray2D();
        Console.WriteLine($"Output range: [{result.Cast<float>().Min():F3}, {result.Cast<float>().Max():F3}]");
    }
}
```

## 🔬 Advanced Examples (.NET 9.0 Performance)

### Multi-GPU Workload Distribution

Distribute computation across multiple accelerators.

```csharp
using UniversalCompute;
using UniversalCompute.Runtime;

class MultiGPUExample
{
    static async Task Main()
    {
        using var context = Context.Create().EnableAllAccelerators();
        
        // Get all available accelerators
        var accelerators = new List<Accelerator>();
        foreach (var device in context.Devices)
        {
            try
            {
                accelerators.Add(device.CreateAccelerator(context));
            }
            catch
            {
                // Skip unavailable devices
            }
        }
        
        if (accelerators.Count == 0)
        {
            Console.WriteLine("No accelerators available");
            return;
        }
        
        Console.WriteLine($"Using {accelerators.Count} accelerators:");
        for (int i = 0; i < accelerators.Count; i++)
        {
            Console.WriteLine($"  {i}: {accelerators[i].Name}");
        }
        
        try
        {
            // Large workload to distribute
            const int totalSize = 10_000_000;
            var workloadPerDevice = totalSize / accelerators.Count;
            var remainder = totalSize % accelerators.Count;
            
            // Prepare input data
            var inputData = new float[totalSize];
            var random = new Random(42);
            for (int i = 0; i < totalSize; i++)
            {
                inputData[i] = random.NextSingle() * 100;
            }
            
            // Distribute work across accelerators
            var tasks = new List<Task<(int DeviceId, float[] Result, double Time)>>();
            
            for (int deviceId = 0; deviceId < accelerators.Count; deviceId++)
            {
                var startIndex = deviceId * workloadPerDevice;
                var size = workloadPerDevice + (deviceId == accelerators.Count - 1 ? remainder : 0);
                var deviceData = new ArraySegment<float>(inputData, startIndex, size).ToArray();
                
                tasks.Add(ProcessOnDevice(accelerators[deviceId], deviceId, deviceData));
            }
            
            // Wait for all devices to complete
            var results = await Task.WhenAll(tasks);
            
            // Combine results and analyze performance
            var totalResult = new List<float>();
            var totalTime = 0.0;
            
            foreach (var (deviceId, result, time) in results.OrderBy(r => r.DeviceId))
            {
                totalResult.AddRange(result);
                totalTime = Math.Max(totalTime, time); // Parallel execution time
                Console.WriteLine($"Device {deviceId}: {result.Length:N0} elements in {time:F2} ms");
            }
            
            Console.WriteLine($"Total processing time: {totalTime:F2} ms");
            Console.WriteLine($"Total throughput: {totalSize / (totalTime / 1000.0) / 1_000_000:F2} M elements/sec");
            Console.WriteLine($"Combined result length: {totalResult.Count:N0}");
        }
        finally
        {
            // Clean up all accelerators
            foreach (var accelerator in accelerators)
            {
                accelerator.Dispose();
            }
        }
    }
    
    static async Task<(int DeviceId, float[] Result, double Time)> ProcessOnDevice(Accelerator accelerator, int deviceId, float[] data)
    {
        // Complex computation kernel
        static void ComplexKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            var x = input[index];
            
            // Expensive computation
            var result = 0.0f;
            for (int i = 0; i < 100; i++) // Simulate computational intensity
            {
                result += Math.Sin(x + i) * Math.Cos(x * i) + Math.Sqrt(Math.Abs(x));
            }
            
            output[index] = result;
        }
        
        // Allocate memory on this specific device
        using var inputBuffer = accelerator.Allocate1D<float>(data.Length);
        using var outputBuffer = accelerator.Allocate1D<float>(data.Length);
        
        // Copy data to device
        inputBuffer.CopyFromCPU(data);
        
        // Load and execute kernel
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(ComplexKernel);
        
        var stopwatch = Stopwatch.StartNew();
        kernel(data.Length, inputBuffer.View, outputBuffer.View);
        accelerator.Synchronize();
        stopwatch.Stop();
        
        // Get results
        var result = outputBuffer.GetAsArray1D();
        
        return (deviceId, result, stopwatch.ElapsedMilliseconds);
    }
}
```

## 📊 Performance Examples (.NET 9.0 Benchmarks)

### Memory Bandwidth Test

Measure memory bandwidth across different accelerators.

```csharp
using UniversalCompute;
using UniversalCompute.Runtime;

class MemoryBandwidthTest
{
    static void Main()
    {
        using var context = Context.Create().EnableAllAccelerators();
        
        Console.WriteLine("Memory Bandwidth Benchmark");
        Console.WriteLine("=========================");
        
        foreach (var device in context.Devices)
        {
            try
            {
                using var accelerator = device.CreateAccelerator(context);
                TestMemoryBandwidth(accelerator);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ {device.Name}: Failed ({ex.Message})");
            }
        }
    }
    
    static void TestMemoryBandwidth(Accelerator accelerator)
    {
        Console.WriteLine($"\n🔬 Testing: {accelerator.Name}");
        
        // Test different data sizes
        var sizes = new[] { 1_000_000, 10_000_000, 100_000_000 };
        
        foreach (var size in sizes)
        {
            // Skip if not enough memory
            var requiredMemory = size * sizeof(float) * 3; // 3 buffers
            if (requiredMemory > accelerator.MemorySize * 0.8) // Use 80% of available memory
                continue;
                
            TestBandwidthForSize(accelerator, size);
        }
    }
    
    static void TestBandwidthForSize(Accelerator accelerator, int size)
    {
        // Memory copy kernel
        static void MemcpyKernel(Index1D index, ArrayView<float> source, ArrayView<float> destination)
        {
            destination[index] = source[index];
        }
        
        // Streaming kernel (read + write)
        static void StreamKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> c)
        {
            c[index] = a[index] + b[index];
        }
        
        using var bufferA = accelerator.Allocate1D<float>(size);
        using var bufferB = accelerator.Allocate1D<float>(size);
        using var bufferC = accelerator.Allocate1D<float>(size);
        
        // Initialize data
        var data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = i;
        }
        bufferA.CopyFromCPU(data);
        bufferB.CopyFromCPU(data);
        
        var memcpyKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(MemcpyKernel);
        var streamKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(StreamKernel);
        
        // Test memory copy bandwidth
        var stopwatch = Stopwatch.StartNew();
        memcpyKernel(size, bufferA.View, bufferC.View);
        accelerator.Synchronize();
        stopwatch.Stop();
        
        var copyBandwidth = (size * sizeof(float) * 2) / (stopwatch.ElapsedMilliseconds / 1000.0) / (1024 * 1024 * 1024);
        
        // Test streaming bandwidth
        stopwatch.Restart();
        streamKernel(size, bufferA.View, bufferB.View, bufferC.View);
        accelerator.Synchronize();
        stopwatch.Stop();
        
        var streamBandwidth = (size * sizeof(float) * 3) / (stopwatch.ElapsedMilliseconds / 1000.0) / (1024 * 1024 * 1024);
        
        Console.WriteLine($"  Size: {size / 1_000_000:F1}M elements");
        Console.WriteLine($"    Copy bandwidth: {copyBandwidth:F2} GB/s");
        Console.WriteLine($"    Stream bandwidth: {streamBandwidth:F2} GB/s");
    }
}
```

## 🎯 Real-World Applications (.NET 9.0)

### Financial Risk Calculation

Monte Carlo simulation for option pricing.

```csharp
using UniversalCompute;
using UniversalCompute.Runtime;

class FinancialRiskExample
{
    static void Main()
    {
        using var context = Context.Create().EnableAllAccelerators();
        using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);
        
        // Black-Scholes Monte Carlo option pricing
        const int numSimulations = 10_000_000;
        const int numSteps = 252; // Trading days in a year
        
        // Option parameters
        const float S0 = 100.0f;     // Initial stock price
        const float K = 105.0f;      // Strike price
        const float T = 1.0f;        // Time to maturity (1 year)
        const float r = 0.05f;       // Risk-free rate
        const float sigma = 0.2f;    // Volatility
        
        // Monte Carlo kernel for option pricing
        static void MonteCarloOptionKernel(Index1D index, ArrayView<float> results, float S0, float K, float T, float r, float sigma, int numSteps, int seed)
        {
            // Initialize RNG
            var rng = (uint)(seed + index * 12345);
            
            var dt = T / numSteps;
            var drift = (r - 0.5f * sigma * sigma) * dt;
            var diffusion = sigma * Math.Sqrt(dt);
            
            var S = S0;
            
            // Simulate stock price path
            for (int step = 0; step < numSteps; step++)
            {
                // Box-Muller transformation for normal random numbers
                rng = rng * 1664525u + 1013904223u;
                var u1 = rng / (float)uint.MaxValue;
                rng = rng * 1664525u + 1013904223u;
                var u2 = rng / (float)uint.MaxValue;
                
                var z = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
                
                S *= Math.Exp(drift + diffusion * z);
            }
            
            // Calculate payoff (European call option)
            var payoff = Math.Max(S - K, 0.0f);
            results[index] = payoff * Math.Exp(-r * T); // Discounted payoff
        }
        
        using var resultBuffer = accelerator.Allocate1D<float>(numSimulations);
        
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float, float, float, float, float, int, int>(MonteCarloOptionKernel);
        
        Console.WriteLine("Monte Carlo Option Pricing");
        Console.WriteLine($"Initial price: ${S0}");
        Console.WriteLine($"Strike price: ${K}");
        Console.WriteLine($"Time to maturity: {T} years");
        Console.WriteLine($"Risk-free rate: {r:P}");
        Console.WriteLine($"Volatility: {sigma:P}");
        Console.WriteLine($"Simulations: {numSimulations:N0}");
        
        var stopwatch = Stopwatch.StartNew();
        kernel(numSimulations, resultBuffer.View, S0, K, T, r, sigma, numSteps, Environment.TickCount);
        accelerator.Synchronize();
        stopwatch.Stop();
        
        // Calculate option price statistics
        var results = resultBuffer.GetAsArray1D();
        var optionPrice = results.Average();
        var standardError = Math.Sqrt(results.Select(x => (x - optionPrice) * (x - optionPrice)).Average() / numSimulations);
        
        Console.WriteLine($"\nResults:");
        Console.WriteLine($"Option price: ${optionPrice:F4} ± ${standardError:F4}");
        Console.WriteLine($"Computation time: {stopwatch.ElapsedMilliseconds:N0} ms");
        Console.WriteLine($"Simulations per second: {numSimulations / (stopwatch.ElapsedMilliseconds / 1000.0):F0}");
        
        // Analytical Black-Scholes for comparison
        var analyticalPrice = BlackScholesCallOption(S0, K, T, r, sigma);
        var error = Math.Abs(optionPrice - analyticalPrice);
        
        Console.WriteLine($"Analytical price: ${analyticalPrice:F4}");
        Console.WriteLine($"Monte Carlo error: ${error:F4} ({error / analyticalPrice:P2})");
    }
    
    static double BlackScholesCallOption(double S, double K, double T, double r, double sigma)
    {
        var d1 = (Math.Log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
        var d2 = d1 - sigma * Math.Sqrt(T);
        
        var callPrice = S * NormalCDF(d1) - K * Math.Exp(-r * T) * NormalCDF(d2);
        return callPrice;
    }
    
    static double NormalCDF(double x)
    {
        return 0.5 * (1 + Erf(x / Math.Sqrt(2)));
    }
    
    static double Erf(double x)
    {
        // Approximation of error function
        var a1 = 0.254829592;
        var a2 = -0.284496736;
        var a3 = 1.421413741;
        var a4 = -1.453152027;
        var a5 = 1.061405429;
        var p = 0.3275911;
        
        var sign = x >= 0 ? 1 : -1;
        x = Math.Abs(x);
        
        var t = 1.0 / (1.0 + p * x);
        var y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        
        return sign * y;
    }
}
```

---

## 🎓 Learning Path

### Beginner Examples
1. **Vector Addition** - Basic kernel concepts
2. **Matrix Multiplication** - 2D indexing
3. **Image Processing** - Real-world application

### Intermediate Examples
1. **FFT Signal Processing** - Built-in algorithms
2. **Neural Networks** - AI/ML fundamentals
3. **Monte Carlo Simulations** - Statistical computing

### Advanced Examples
1. **Multi-GPU Distribution** - Parallel execution
2. **Performance Optimization** - Memory bandwidth
3. **Financial Applications** - Real-world complexity

---

## 🔗 Related Resources

- **[Quick Start Tutorial](Quick-Start-Tutorial)** - Your first UniversalCompute application
- **[API Reference](API-Reference)** - Complete documentation
- **[Hardware Accelerators](Hardware-Accelerators)** - Specialized hardware support
- **[Performance Tuning](Performance-Tuning)** - Optimization strategies

---

**🚀 Start building amazing applications with these examples!**