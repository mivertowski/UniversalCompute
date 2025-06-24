// ---------------------------------------------------------------------------------------
//                           ILGPU Phase 8 Universal Compute Platform
//                    01-UniversalKernels: Write-Once, Run-Anywhere Programming
// ---------------------------------------------------------------------------------------

using System;
using System.Diagnostics;
using System.Numerics;
using ILGPU;
using ILGPU.CrossPlatform;
using ILGPU.Memory.Unified;
using ILGPU.Runtime;
using ILGPU.Runtime.Scheduling;

namespace Phase8.Universal.UniversalKernels
{
    /// <summary>
    /// Demonstrates the Universal Compute Platform's revolutionary write-once, run-anywhere
    /// programming model with automatic platform optimization and intelligent scheduling.
    /// </summary>
    class Program
    {
        /// <summary>
        /// Universal kernel that automatically optimizes for all hardware platforms.
        /// The same code runs optimally on Intel NPU, NVIDIA Tensor Cores, Apple Neural Engine,
        /// AMD MFMA units, and traditional CPU/GPU architectures.
        /// </summary>
        [UniversalKernel(EnableOptimizations = true, SupportsMixedPrecision = true)]
        [AppleOptimization(UseAMX = true, UseNeuralEngine = true, UseMetalPerformanceShaders = true)]
        [IntelOptimization(UseAMX = true, UseAVX512 = true, UseNPU = true, UseDLBoost = true)]
        [NvidiaOptimization(UseTensorCores = true, UseCuBLAS = true, OptimizePTXGeneration = true)]
        [AMDOptimization(UseMFMA = true, UseROCmBLAS = true, OptimizeForOccupancy = true)]
        static void UniversalMatrixMultiply(
            ArrayView2D<float, Stride2D.DenseX> matrixA,
            ArrayView2D<float, Stride2D.DenseX> matrixB,
            ArrayView2D<float, Stride2D.DenseX> result)
        {
            // Universal indexing works across all platforms - CUDA, OpenCL, Metal, CPU
            var globalPos = UniversalGrid.GlobalIndex.XY;
            var row = globalPos.Y;
            var col = globalPos.X;

            // Bounds checking using universal grid properties
            if (row < result.Height && col < result.Width)
            {
                float sum = 0.0f;
                
                // The compiler automatically optimizes this loop for:
                // - Intel AMX/NPU: Uses matrix extensions and neural processing
                // - NVIDIA: Uses Tensor Cores and cuBLAS optimizations
                // - Apple: Uses AMX and Neural Engine acceleration
                // - AMD: Uses MFMA and ROCm BLAS optimizations
                // - CPU: Uses SIMD vectorization (AVX-512, NEON, etc.)
                for (int k = 0; k < matrixA.Width; k++)
                {
                    sum += matrixA[row, k] * matrixB[k, col];
                }
                
                result[row, col] = sum;
            }
        }

        /// <summary>
        /// Universal AI/ML processing kernel with automatic mixed precision.
        /// Demonstrates advanced features like neural network layer processing
        /// with automatic hardware acceleration.
        /// </summary>
        [UniversalKernel(SupportsMixedPrecision = true, PreferredStrategy = KernelExecutionStrategy.GPU)]
        [AppleOptimization(UseNeuralEngine = true, OptimizeForLatency = true)]
        [IntelOptimization(UseNPU = true, UseDLBoost = true)]
        [NvidiaOptimization(UseTensorCores = true, UseCuDNN = true)]
        static void UniversalNeuralNetworkLayer(
            ArrayView<float> inputs,
            ArrayView2D<float, Stride2D.DenseX> weights,
            ArrayView<float> biases,
            ArrayView<float> outputs)
        {
            var outputIndex = UniversalGrid.GlobalIndex.X;
            
            if (outputIndex < outputs.Length)
            {
                float sum = biases[outputIndex];
                
                // Automatically optimized for:
                // - Apple Neural Engine: Direct neural processing acceleration
                // - Intel NPU: AI-specific instruction acceleration
                // - NVIDIA Tensor Cores: Mixed precision matrix operations
                // - Traditional GPUs: Parallel multiply-accumulate operations
                for (int i = 0; i < inputs.Length; i++)
                {
                    sum += inputs[i] * weights[outputIndex, i];
                }
                
                // Universal activation function with platform-specific optimizations
                outputs[outputIndex] = MathF.Max(0.0f, sum); // ReLU activation
            }
        }

        /// <summary>
        /// Universal vector processing kernel demonstrating SIMD optimization
        /// across all CPU architectures (Intel AVX, ARM NEON, etc.).
        /// </summary>
        [UniversalKernel(PreferredStrategy = KernelExecutionStrategy.CPU)]
        [IntelOptimization(UseAVX512 = true, OptimizeForThroughput = true)]
        [AppleOptimization(UseTiledMemory = true)]
        static void UniversalSIMDProcessing(
            ArrayView<Vector4> vectors,
            ArrayView<Vector4> result,
            float scalar)
        {
            var index = UniversalGrid.GlobalIndex.X;
            
            if (index < vectors.Length)
            {
                // Universal SIMD operations automatically use:
                // - Intel: AVX-512, AVX2, SSE optimizations
                // - Apple: NEON optimizations with AMX acceleration
                // - AMD: Optimized vector processing
                var vec = vectors[index];
                
                // Platform-optimized vector operations
                result[index] = new Vector4(
                    vec.X * scalar + MathF.Sin(vec.X),
                    vec.Y * scalar + MathF.Cos(vec.Y),
                    vec.Z * scalar + MathF.Sqrt(MathF.Abs(vec.Z)),
                    vec.W * scalar + MathF.Log(MathF.Abs(vec.W) + 1.0f)
                );
            }
        }

        static async Task Main(string[] args)
        {
            Console.WriteLine("🌟 ILGPU Phase 8: Universal Compute Platform");
            Console.WriteLine("===========================================");
            Console.WriteLine("Write-Once, Run-Anywhere with Automatic Optimization\n");

            // Initialize Universal Compute Platform
            using var context = Context.CreateDefault();
            using var memoryManager = new UniversalMemoryManager(context);
            
            // Discover and initialize all available accelerators
            var availableAccelerators = await DiscoverAcceleratorsAsync(context);
            Console.WriteLine($"🎯 Discovered {availableAccelerators.Count} accelerators\n");

            // Initialize adaptive scheduler for intelligent workload distribution
            using var scheduler = new AdaptiveScheduler(availableAccelerators, SchedulingPolicy.PerformanceOptimized);

            // Demonstrate universal matrix multiplication
            await DemonstrateUniversalMatrixMultiplication(memoryManager, scheduler);
            
            // Demonstrate universal neural network processing
            await DemonstrateUniversalNeuralProcessing(memoryManager, scheduler);
            
            // Demonstrate universal SIMD processing
            await DemonstrateUniversalSIMDProcessing(memoryManager, scheduler);
            
            // Show performance analytics
            ShowPerformanceAnalytics(scheduler, memoryManager);

            Console.WriteLine("\n🎉 Universal Compute Platform demonstration completed!");
            Console.WriteLine("✨ Same code, optimal performance on every platform!");
        }

        /// <summary>
        /// Discovers and initializes all available accelerators with graceful fallbacks.
        /// </summary>
        static async Task<Dictionary<ComputeDevice, IAccelerator>> DiscoverAcceleratorsAsync(Context context)
        {
            var accelerators = new Dictionary<ComputeDevice, IAccelerator>();

            // Try NVIDIA CUDA (Tensor Cores, cuBLAS, cuDNN)
            if (context.GetCudaDevices().Any())
            {
                var cudaDevice = context.CreateCudaAccelerator(0);
                accelerators[ComputeDevice.CUDA] = cudaDevice;
                Console.WriteLine($"✅ NVIDIA CUDA: {cudaDevice.Name} ({cudaDevice.MemorySize / (1024*1024)} MB)");
            }

            // Try OpenCL (AMD GPUs, Intel GPUs, other accelerators)
            if (context.GetOpenCLDevices().Any())
            {
                var openclDevice = context.CreateOpenCLAccelerator(0);
                accelerators[ComputeDevice.OpenCL] = openclDevice;
                Console.WriteLine($"✅ OpenCL: {openclDevice.Name} ({openclDevice.MemorySize / (1024*1024)} MB)");
            }

            // CPU with SIMD optimizations (always available)
            var cpuDevice = context.CreateCPUAccelerator(0);
            accelerators[ComputeDevice.CPU] = cpuDevice;
            Console.WriteLine($"✅ CPU SIMD: {cpuDevice.Name} ({Environment.ProcessorCount} cores)");

            // Note: Intel NPU, Apple Neural Engine detection would be added here
            // These require platform-specific detection and initialization

            return accelerators;
        }

        /// <summary>
        /// Demonstrates universal matrix multiplication with automatic optimization.
        /// </summary>
        static async Task DemonstrateUniversalMatrixMultiplication(
            UniversalMemoryManager memoryManager, 
            AdaptiveScheduler scheduler)
        {
            Console.WriteLine("🔢 Universal Matrix Multiplication Demonstration");
            Console.WriteLine("-----------------------------------------------");

            const int matrixSize = 256;
            
            // Allocate universal memory with intelligent placement
            using var matrixA = memoryManager.AllocateUniversal2D<float>(
                matrixSize, matrixSize, MemoryPlacement.Auto, MemoryAccessPattern.Sequential);
            using var matrixB = memoryManager.AllocateUniversal2D<float>(
                matrixSize, matrixSize, MemoryPlacement.Auto, MemoryAccessPattern.Sequential);
            using var result = memoryManager.AllocateUniversal2D<float>(
                matrixSize, matrixSize, MemoryPlacement.Auto, MemoryAccessPattern.Sequential);

            // Initialize matrices
            var random = new Random(42);
            await InitializeMatrixAsync(matrixA, random);
            await InitializeMatrixAsync(matrixB, random);

            // Create compute graph for matrix multiplication
            var graph = new ComputeGraph();
            var matmulNode = new ComputeNode(new MatMulOp(matrixSize, matrixSize, matrixSize)) 
            { 
                Id = "UniversalMatMul" 
            };
            graph.AddNode(matmulNode);

            // Execute with adaptive scheduling
            var stopwatch = Stopwatch.StartNew();
            var executionPlan = await scheduler.CreateExecutionPlanAsync(graph);
            
            // The universal kernel automatically optimizes for the selected device
            var selectedDevice = executionPlan.Assignments[matmulNode];
            var accelerator = scheduler.GetAcceleratorForDevice(selectedDevice);
            
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>>(UniversalMatrixMultiply);
                
            kernel(matrixA.GetView2D(), matrixB.GetView2D(), result.GetView2D());
            accelerator.Synchronize();
            
            stopwatch.Stop();

            // Verify results
            var resultData = await result.CopyToAsync(new float[matrixSize, matrixSize]);
            bool isValid = await ValidateMatrixMultiplication(matrixA, matrixB, resultData);

            Console.WriteLine($"✅ Matrix {matrixSize}x{matrixSize} multiplication completed");
            Console.WriteLine($"   Device: {selectedDevice}");
            Console.WriteLine($"   Execution time: {stopwatch.ElapsedMilliseconds} ms");
            Console.WriteLine($"   Performance: {(2L * matrixSize * matrixSize * matrixSize) / (stopwatch.ElapsedMilliseconds + 1)} MFLOPS");
            Console.WriteLine($"   Validation: {(isValid ? "✅ PASSED" : "❌ FAILED")}\n");
        }

        /// <summary>
        /// Demonstrates universal neural network layer processing.
        /// </summary>
        static async Task DemonstrateUniversalNeuralProcessing(
            UniversalMemoryManager memoryManager,
            AdaptiveScheduler scheduler)
        {
            Console.WriteLine("🧠 Universal Neural Network Processing Demonstration");
            Console.WriteLine("--------------------------------------------------");

            const int inputSize = 1024;
            const int outputSize = 512;

            // Allocate memory with neural processing optimization
            using var inputs = memoryManager.AllocateUniversal<float>(
                inputSize, MemoryPlacement.Auto, MemoryAccessPattern.Sequential);
            using var weights = memoryManager.AllocateUniversal2D<float>(
                outputSize, inputSize, MemoryPlacement.Auto, MemoryAccessPattern.Random);
            using var biases = memoryManager.AllocateUniversal<float>(
                outputSize, MemoryPlacement.Auto, MemoryAccessPattern.Sequential);
            using var outputs = memoryManager.AllocateUniversal<float>(
                outputSize, MemoryPlacement.Auto, MemoryAccessPattern.Sequential);

            // Initialize neural network parameters
            await InitializeNeuralNetworkAsync(inputs, weights, biases);

            // Execute neural processing with device selection
            var graph = new ComputeGraph();
            var neuralNode = new ComputeNode(new VectorOp(outputSize)) { Id = "NeuralLayer" };
            graph.AddNode(neuralNode);

            var executionPlan = await scheduler.CreateExecutionPlanAsync(graph);
            var selectedDevice = executionPlan.Assignments[neuralNode];
            var accelerator = scheduler.GetAcceleratorForDevice(selectedDevice);

            var stopwatch = Stopwatch.StartNew();
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                ArrayView<float>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView<float>,
                ArrayView<float>>(UniversalNeuralNetworkLayer);
                
            kernel(inputs.GetView1D(), weights.GetView2D(), biases.GetView1D(), outputs.GetView1D());
            accelerator.Synchronize();
            stopwatch.Stop();

            Console.WriteLine($"✅ Neural layer ({inputSize}→{outputSize}) processing completed");
            Console.WriteLine($"   Device: {selectedDevice}");
            Console.WriteLine($"   Execution time: {stopwatch.ElapsedMilliseconds} ms");
            Console.WriteLine($"   Throughput: {(long)inputSize * outputSize / (stopwatch.ElapsedMilliseconds + 1)} ops/ms\n");
        }

        /// <summary>
        /// Demonstrates universal SIMD vector processing.
        /// </summary>
        static async Task DemonstrateUniversalSIMDProcessing(
            UniversalMemoryManager memoryManager,
            AdaptiveScheduler scheduler)
        {
            Console.WriteLine("⚡ Universal SIMD Vector Processing Demonstration");
            Console.WriteLine("------------------------------------------------");

            const int vectorCount = 16384;
            const float scalar = 3.14159f;

            // Allocate vector data
            using var vectors = memoryManager.AllocateUniversal<Vector4>(
                vectorCount, MemoryPlacement.Auto, MemoryAccessPattern.Sequential);
            using var results = memoryManager.AllocateUniversal<Vector4>(
                vectorCount, MemoryPlacement.Auto, MemoryAccessPattern.Sequential);

            // Initialize vector data
            var vectorData = new Vector4[vectorCount];
            var random = new Random(42);
            for (int i = 0; i < vectorCount; i++)
            {
                vectorData[i] = new Vector4(
                    (float)random.NextDouble(),
                    (float)random.NextDouble(),
                    (float)random.NextDouble(),
                    (float)random.NextDouble());
            }
            await vectors.CopyFromAsync(vectorData);

            // Execute SIMD processing
            var graph = new ComputeGraph();
            var simdNode = new ComputeNode(new VectorOp(vectorCount)) { Id = "SIMDProcessing" };
            graph.AddNode(simdNode);

            var executionPlan = await scheduler.CreateExecutionPlanAsync(graph);
            var selectedDevice = executionPlan.Assignments[simdNode];
            var accelerator = scheduler.GetAcceleratorForDevice(selectedDevice);

            var stopwatch = Stopwatch.StartNew();
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                ArrayView<Vector4>,
                ArrayView<Vector4>,
                float>(UniversalSIMDProcessing);
                
            kernel(vectors.GetView1D(), results.GetView1D(), scalar);
            accelerator.Synchronize();
            stopwatch.Stop();

            Console.WriteLine($"✅ SIMD vector processing ({vectorCount} Vector4s) completed");
            Console.WriteLine($"   Device: {selectedDevice}");
            Console.WriteLine($"   Execution time: {stopwatch.ElapsedMilliseconds} ms");
            Console.WriteLine($"   Throughput: {vectorCount * 4 / (stopwatch.ElapsedMilliseconds + 1)} elements/ms\n");
        }

        /// <summary>
        /// Shows comprehensive performance analytics across all systems.
        /// </summary>
        static void ShowPerformanceAnalytics(AdaptiveScheduler scheduler, UniversalMemoryManager memoryManager)
        {
            Console.WriteLine("📊 Performance Analytics");
            Console.WriteLine("------------------------");

            var schedStats = scheduler.GetPerformanceStatistics();
            Console.WriteLine($"Scheduler Performance:");
            Console.WriteLine($"  Total executions: {schedStats.TotalExecutions}");
            Console.WriteLine($"  Average execution time: {schedStats.AverageExecutionTimeMs:F2} ms");
            Console.WriteLine($"  Device utilization:");
            foreach (var (device, utilization) in schedStats.DeviceUtilization)
            {
                Console.WriteLine($"    {device}: {utilization:P1}");
            }

            var memStats = memoryManager.GetGlobalMemoryStatistics();
            Console.WriteLine($"\nMemory Management:");
            Console.WriteLine($"  Total allocated: {memStats.TotalAllocatedBytes / (1024 * 1024)} MB");
            Console.WriteLine($"  Peak allocation: {memStats.PeakAllocatedBytes / (1024 * 1024)} MB");
            Console.WriteLine($"  Active allocations: {memStats.ActiveAllocations}");

            var recommendations = memoryManager.GetMemoryRecommendations();
            if (recommendations.Any())
            {
                Console.WriteLine($"  Recommendations: {string.Join(", ", recommendations)}");
            }
        }

        // Helper methods for initialization and validation
        static async Task InitializeMatrixAsync(IUniversalBuffer2D<float> matrix, Random random)
        {
            var data = new float[matrix.Height, matrix.Width];
            for (int i = 0; i < matrix.Height; i++)
            {
                for (int j = 0; j < matrix.Width; j++)
                {
                    data[i, j] = (float)random.NextDouble();
                }
            }
            await matrix.CopyFromAsync(data);
        }

        static async Task<bool> ValidateMatrixMultiplication(
            IUniversalBuffer2D<float> matrixA,
            IUniversalBuffer2D<float> matrixB,
            float[,] result)
        {
            // Simple validation - check a few random elements
            var dataA = await matrixA.CopyToAsync(new float[matrixA.Height, matrixA.Width]);
            var dataB = await matrixB.CopyToAsync(new float[matrixB.Height, matrixB.Width]);
            
            var random = new Random(42);
            for (int check = 0; check < 10; check++)
            {
                int row = random.Next(result.GetLength(0));
                int col = random.Next(result.GetLength(1));
                
                float expected = 0;
                for (int k = 0; k < dataA.GetLength(1); k++)
                {
                    expected += dataA[row, k] * dataB[k, col];
                }
                
                if (MathF.Abs(result[row, col] - expected) > 0.001f)
                    return false;
            }
            return true;
        }

        static async Task InitializeNeuralNetworkAsync(
            IUniversalBuffer<float> inputs,
            IUniversalBuffer2D<float> weights,
            IUniversalBuffer<float> biases)
        {
            var random = new Random(42);
            
            var inputData = new float[inputs.Length];
            for (int i = 0; i < inputData.Length; i++)
            {
                inputData[i] = (float)random.NextDouble();
            }
            await inputs.CopyFromAsync(inputData);

            var weightData = new float[weights.Height, weights.Width];
            for (int i = 0; i < weights.Height; i++)
            {
                for (int j = 0; j < weights.Width; j++)
                {
                    weightData[i, j] = (float)(random.NextDouble() * 2 - 1); // [-1, 1]
                }
            }
            await weights.CopyFromAsync(weightData);

            var biasData = new float[biases.Length];
            for (int i = 0; i < biasData.Length; i++)
            {
                biasData[i] = (float)(random.NextDouble() * 0.1); // Small bias values
            }
            await biases.CopyFromAsync(biasData);
        }
    }
}