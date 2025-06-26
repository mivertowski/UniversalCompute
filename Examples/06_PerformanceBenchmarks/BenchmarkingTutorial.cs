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

// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Examples.Common;

namespace ILGPU.Examples.PerformanceBenchmarks
{
    /// <summary>
    /// Comprehensive tutorial on benchmarking ILGPU applications and hardware accelerators.
    /// This example demonstrates best practices for measuring and optimizing performance.
    /// </summary>
    class BenchmarkingTutorial
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("📊 ILGPU Performance Benchmarking Tutorial");
            Console.WriteLine("==========================================\n");

            try
            {
                // Detect available hardware
                var hardware = HardwareDetection.DetectAvailableHardware();
                HardwareDetection.PrintHardwareReport(hardware);

                using var context = Context.CreateDefault();
                
                // Run benchmarking tutorials
                await RunBasicBenchmarking(context);
                await RunMemoryBenchmarks(context);
                await RunKernelOptimization(context);
                await RunConcurrencyBenchmarks(context);
                await RunRealWorldScenarios(context);
                
                // Provide optimization guidelines
                ProvideOptimizationGuidelines();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                Console.WriteLine($"Details: {ex}");
            }

            Console.WriteLine("\n✅ Tutorial completed. Press any key to exit...");
            Console.ReadKey();
        }

        /// <summary>
        /// Demonstrates basic benchmarking principles and measurement techniques.
        /// </summary>
        static async Task RunBasicBenchmarking(Context context)
        {
            Console.WriteLine("📏 Basic Benchmarking Principles");
            Console.WriteLine("===============================\n");

            using var accelerator = context.CreateCPUAccelerator();
            
            Console.WriteLine("🔍 Key Benchmarking Concepts:");
            Console.WriteLine("• Warm-up iterations to stabilize performance");
            Console.WriteLine("• Multiple measurements for statistical validity");
            Console.WriteLine("• Proper synchronization before timing");
            Console.WriteLine("• Separate compilation from execution timing");
            Console.WriteLine("• Consider memory allocation overhead");
            Console.WriteLine();

            // Demonstrate proper benchmarking of a simple kernel
            await DemonstrateProperBenchmarking(accelerator);
            await DemonstrateCommonMistakes(accelerator);
        }

        /// <summary>
        /// Benchmarks memory operations and data transfer patterns.
        /// </summary>
        static async Task RunMemoryBenchmarks(Context context)
        {
            Console.WriteLine("💾 Memory Performance Benchmarks");
            Console.WriteLine("===============================\n");

            using var accelerator = context.CreateCPUAccelerator();

            var dataSizes = new[] { 1024, 4096, 16384, 65536, 262144, 1048576 };
            
            Console.WriteLine($"{"Data Size",-12} {"H2D (GB/s)",-12} {"D2H (GB/s)",-12} {"D2D (GB/s)",-12} {"Allocation",-12}");
            Console.WriteLine($"{new string('-', 65)}");

            foreach (var size in dataSizes)
            {
                var results = await BenchmarkMemoryOperations(accelerator, size);
                
                Console.WriteLine($"{FormatDataSize(size),-12} " +
                                $"{results.HostToDevice,-12:F2} " +
                                $"{results.DeviceToHost,-12:F2} " +
                                $"{results.DeviceToDevice,-12:F2} " +
                                $"{results.AllocationTime,-12:F2} ms");
            }

            Console.WriteLine();
            Console.WriteLine("📝 Memory Optimization Tips:");
            Console.WriteLine("• Minimize host-device transfers");
            Console.WriteLine("• Use pinned memory for frequent transfers");
            Console.WriteLine("• Batch small transfers together");
            Console.WriteLine("• Reuse memory buffers when possible");
            Console.WriteLine("• Consider unified memory on supported platforms");
            Console.WriteLine();
        }

        /// <summary>
        /// Demonstrates kernel optimization techniques and their performance impact.
        /// </summary>
        static async Task RunKernelOptimization(Context context)
        {
            Console.WriteLine("⚡ Kernel Optimization Benchmarks");
            Console.WriteLine("=================================\n");

            using var accelerator = context.CreateCPUAccelerator();

            var optimizations = new[]
            {
                new { Name = "Naive Implementation", Kernel = (Action<Index1D, ArrayView<float>, ArrayView<float>>)NaiveKernel },
                new { Name = "Vectorized Operations", Kernel = (Action<Index1D, ArrayView<float>, ArrayView<float>>)VectorizedKernel },
                new { Name = "Loop Unrolling", Kernel = (Action<Index1D, ArrayView<float>, ArrayView<float>>)UnrolledKernel },
                new { Name = "Memory Coalescing", Kernel = (Action<Index1D, ArrayView<float>, ArrayView<float>>)CoalescedKernel }
            };

            const int dataSize = 1048576; // 1M elements
            var inputData = CreateTestData(dataSize);

            Console.WriteLine($"{"Optimization",-20} {"Time (ms)",-12} {"GFLOPS",-12} {"Speedup",-10} {"Efficiency",-12}");
            Console.WriteLine($"{new string('-', 70)}");

            double baselineTime = 0;

            foreach (var optimization in optimizations)
            {
                var time = await BenchmarkKernel(accelerator, optimization.Kernel, inputData);
                var gflops = CalculateGFLOPS(dataSize, time);
                var speedup = baselineTime > 0 ? baselineTime / time : 1.0;
                var efficiency = CalculateEfficiency(accelerator, gflops);

                if (baselineTime == 0) baselineTime = time;

                Console.WriteLine($"{optimization.Name,-20} {time,-12:F2} {gflops,-12:F1} {speedup,-10:F1}x {efficiency,-12:F1}%");
            }

            Console.WriteLine();
            Console.WriteLine("🎯 Kernel Optimization Techniques:");
            Console.WriteLine("• Use appropriate data types (avoid unnecessary precision)");
            Console.WriteLine("• Minimize divergent branches");
            Console.WriteLine("• Optimize memory access patterns");
            Console.WriteLine("• Use shared memory effectively");
            Console.WriteLine("• Consider loop unrolling for small loops");
            Console.WriteLine("• Leverage hardware-specific intrinsics");
            Console.WriteLine();
        }

        /// <summary>
        /// Benchmarks concurrent execution and multi-stream scenarios.
        /// </summary>
        static async Task RunConcurrencyBenchmarks(Context context)
        {
            Console.WriteLine("🔄 Concurrency and Multi-Stream Benchmarks");
            Console.WriteLine("==========================================\n");

            using var accelerator = context.CreateCPUAccelerator();

            var scenarios = new[]
            {
                new { Name = "Single Stream", Streams = 1 },
                new { Name = "Dual Streams", Streams = 2 },
                new { Name = "Quad Streams", Streams = 4 },
                new { Name = "Optimal Streams", Streams = Math.Min(8, Environment.ProcessorCount) }
            };

            const int workloadSize = 262144; // 256K elements per stream

            Console.WriteLine($"{"Scenario",-16} {"Streams",-8} {"Total Time",-12} {"Throughput",-12} {"Efficiency",-12}");
            Console.WriteLine($"{new string('-', 65)}");

            foreach (var scenario in scenarios)
            {
                var results = await BenchmarkConcurrentExecution(accelerator, scenario.Streams, workloadSize);
                
                Console.WriteLine($"{scenario.Name,-16} {scenario.Streams,-8} " +
                                $"{results.TotalTime,-12:F2} ms " +
                                $"{results.Throughput,-12:F1} GFLOPS " +
                                $"{results.Efficiency,-12:F1}%");
            }

            Console.WriteLine();
            Console.WriteLine("⚡ Concurrency Best Practices:");
            Console.WriteLine("• Use multiple streams for overlapping computation and I/O");
            Console.WriteLine("• Balance workload across available compute units");
            Console.WriteLine("• Avoid over-subscription of resources");
            Console.WriteLine("• Consider NUMA topology on multi-socket systems");
            Console.WriteLine("• Use async operations for better CPU utilization");
            Console.WriteLine();
        }

        /// <summary>
        /// Benchmarks real-world application scenarios.
        /// </summary>
        static async Task RunRealWorldScenarios(Context context)
        {
            Console.WriteLine("🌍 Real-World Scenario Benchmarks");
            Console.WriteLine("=================================\n");

            using var accelerator = context.CreateCPUAccelerator();

            var scenarios = new[]
            {
                new WorkloadScenario { Name = "Image Processing Pipeline", Operations = 5, DataSize = 1920 * 1080 },
                new WorkloadScenario { Name = "Scientific Simulation", Operations = 10, DataSize = 1024 * 1024 },
                new WorkloadScenario { Name = "Financial Monte Carlo", Operations = 1000, DataSize = 10000 },
                new WorkloadScenario { Name = "Machine Learning Inference", Operations = 3, DataSize = 224 * 224 * 3 },
                new WorkloadScenario { Name = "Signal Processing FFT", Operations = 1, DataSize = 65536 }
            };

            Console.WriteLine($"{"Scenario",-25} {"Latency",-12} {"Throughput",-15} {"Memory BW",-12}");
            Console.WriteLine($"{new string('-', 70)}");

            foreach (var scenario in scenarios)
            {
                var results = await BenchmarkRealWorldScenario(accelerator, scenario);
                
                Console.WriteLine($"{scenario.Name,-25} " +
                                $"{results.Latency,-12:F2} ms " +
                                $"{results.Throughput,-15:F1} ops/sec " +
                                $"{results.MemoryBandwidth,-12:F1} GB/s");
            }

            Console.WriteLine();
            Console.WriteLine("🎯 Application-Level Optimization:");
            Console.WriteLine("• Profile the entire pipeline, not just compute kernels");
            Console.WriteLine("• Identify and eliminate bottlenecks systematically");
            Console.WriteLine("• Consider end-to-end latency vs. throughput trade-offs");
            Console.WriteLine("• Optimize for your specific use case and hardware");
            Console.WriteLine("• Measure power efficiency for mobile/edge deployments");
            Console.WriteLine();
        }

        /// <summary>
        /// Provides comprehensive optimization guidelines.
        /// </summary>
        static void ProvideOptimizationGuidelines()
        {
            Console.WriteLine("💡 Performance Optimization Guidelines");
            Console.WriteLine("====================================\n");

            Console.WriteLine("🔍 Performance Analysis Methodology:");
            Console.WriteLine("1. Establish baseline measurements");
            Console.WriteLine("2. Identify theoretical performance limits");
            Console.WriteLine("3. Profile to find bottlenecks");
            Console.WriteLine("4. Apply targeted optimizations");
            Console.WriteLine("5. Measure and validate improvements");
            Console.WriteLine("6. Repeat until satisfactory performance");
            Console.WriteLine();

            Console.WriteLine("⚡ Optimization Priority Order:");
            Console.WriteLine("1. Algorithm selection (highest impact)");
            Console.WriteLine("2. Memory access optimization");
            Console.WriteLine("3. Compute optimization");
            Console.WriteLine("4. Parallelization strategy");
            Console.WriteLine("5. Hardware-specific tuning");
            Console.WriteLine("6. Low-level micro-optimizations");
            Console.WriteLine();

            Console.WriteLine("🎯 Hardware-Specific Guidelines:");
            Console.WriteLine();
            Console.WriteLine("Apple Neural Engine:");
            Console.WriteLine("• Optimize for low-latency inference");
            Console.WriteLine("• Use Core ML model formats when possible");
            Console.WriteLine("• Minimize precision to FP16 or INT8");
            Console.WriteLine("• Batch size typically 1 for best latency");
            Console.WriteLine();

            Console.WriteLine("Intel NPU:");
            Console.WriteLine("• Use ONNX Runtime with OpenVINO backend");
            Console.WriteLine("• Apply model quantization (INT8/BF16)");
            Console.WriteLine("• Optimize batch sizes for throughput");
            Console.WriteLine("• Leverage dynamic shape support");
            Console.WriteLine();

            Console.WriteLine("Intel AMX:");
            Console.WriteLine("• Design algorithms around tile operations");
            Console.WriteLine("• Use optimal tile sizes (16x64 bytes)");
            Console.WriteLine("• Prefer BF16 for ML workloads");
            Console.WriteLine("• Cache blocking for large matrices");
            Console.WriteLine();

            Console.WriteLine("CUDA GPUs:");
            Console.WriteLine("• Maximize occupancy within reasonable bounds");
            Console.WriteLine("• Use appropriate memory types (shared, constant)");
            Console.WriteLine("• Optimize for memory coalescing");
            Console.WriteLine("• Consider tensor core utilization");
            Console.WriteLine();

            Console.WriteLine("📊 Benchmarking Best Practices:");
            Console.WriteLine("• Use representative workloads");
            Console.WriteLine("• Measure statistical significance");
            Console.WriteLine("• Account for thermal throttling");
            Console.WriteLine("• Test across different input sizes");
            Console.WriteLine("• Validate correctness alongside performance");
            Console.WriteLine("• Document test conditions and environment");
            Console.WriteLine();

            Console.WriteLine("🚀 Production Deployment Tips:");
            Console.WriteLine("• Monitor performance in production");
            Console.WriteLine("• Plan for graceful degradation");
            Console.WriteLine("• Implement adaptive batch sizing");
            Console.WriteLine("• Use performance budgets and SLAs");
            Console.WriteLine("• Consider power and thermal constraints");
            Console.WriteLine("• Prepare fallback execution paths");
        }

        #region Benchmarking Implementation Methods

        /// <summary>
        /// Demonstrates proper benchmarking technique.
        /// </summary>
        static async Task DemonstrateProperBenchmarking(Accelerator accelerator)
        {
            Console.WriteLine("✅ Proper Benchmarking Example:");
            Console.WriteLine("------------------------------");

            const int dataSize = 1048576;
            const int warmupIterations = 5;
            const int measurementIterations = 20;

            // Prepare data
            var input = accelerator.Allocate1D<float>(dataSize);
            var output = accelerator.Allocate1D<float>(dataSize);
            
            // Compile kernel (exclude from timing)
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SimpleKernel);

            // Warm-up phase
            Console.WriteLine($"   Performing {warmupIterations} warm-up iterations...");
            for (int i = 0; i < warmupIterations; i++)
            {
                kernel(dataSize, input.View, output.View);
                accelerator.Synchronize();
            }

            // Measurement phase
            Console.WriteLine($"   Measuring performance over {measurementIterations} iterations...");
            var times = new List<double>();
            
            for (int i = 0; i < measurementIterations; i++)
            {
                accelerator.Synchronize(); // Ensure clean start
                
                var stopwatch = Stopwatch.StartNew();
                kernel(dataSize, input.View, output.View);
                accelerator.Synchronize(); // Wait for completion
                stopwatch.Stop();
                
                times.Add(stopwatch.ElapsedMilliseconds);
            }

            // Statistical analysis
            var avgTime = times.Average();
            var minTime = times.Min();
            var maxTime = times.Max();
            var stdDev = Math.Sqrt(times.Average(t => Math.Pow(t - avgTime, 2)));
            var gflops = CalculateGFLOPS(dataSize, avgTime);

            Console.WriteLine($"   Results: Avg={avgTime:F2}ms, Min={minTime:F2}ms, Max={maxTime:F2}ms");
            Console.WriteLine($"   Std Dev: {stdDev:F2}ms ({stdDev/avgTime*100:F1}%)");
            Console.WriteLine($"   Performance: {gflops:F1} GFLOPS");
            Console.WriteLine();

            input.Dispose();
            output.Dispose();
        }

        /// <summary>
        /// Shows common benchmarking mistakes to avoid.
        /// </summary>
        static async Task DemonstrateCommonMistakes(Accelerator accelerator)
        {
            Console.WriteLine("❌ Common Benchmarking Mistakes:");
            Console.WriteLine("-------------------------------");

            const int dataSize = 1048576;

            Console.WriteLine("   Mistake 1: Including compilation time");
            var stopwatch = Stopwatch.StartNew();
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SimpleKernel);
            var input = accelerator.Allocate1D<float>(dataSize);
            var output = accelerator.Allocate1D<float>(dataSize);
            kernel(dataSize, input.View, output.View);
            accelerator.Synchronize();
            stopwatch.Stop();
            Console.WriteLine($"   └─ Wrong timing (with compilation): {stopwatch.ElapsedMilliseconds:F2}ms");

            Console.WriteLine("   Mistake 2: Not synchronizing before measurement");
            stopwatch.Restart();
            kernel(dataSize, input.View, output.View);
            // Missing: accelerator.Synchronize();
            stopwatch.Stop();
            Console.WriteLine($"   └─ Wrong timing (no sync): {stopwatch.ElapsedMilliseconds:F2}ms");

            Console.WriteLine("   Mistake 3: Single measurement (no statistical validity)");
            accelerator.Synchronize();
            stopwatch.Restart();
            kernel(dataSize, input.View, output.View);
            accelerator.Synchronize();
            stopwatch.Stop();
            Console.WriteLine($"   └─ Unreliable single measurement: {stopwatch.ElapsedMilliseconds:F2}ms");

            Console.WriteLine("   ✅ Always use proper methodology for reliable results!");
            Console.WriteLine();

            input.Dispose();
            output.Dispose();
        }

        /// <summary>
        /// Benchmarks memory operations.
        /// </summary>
        static async Task<MemoryBenchmarkResults> BenchmarkMemoryOperations(Accelerator accelerator, int elementCount)
        {
            const int iterations = 10;
            var data = new float[elementCount];
            var results = new MemoryBenchmarkResults();

            // Host to Device
            using var buffer = accelerator.Allocate1D<float>(elementCount);
            var times = new List<double>();
            
            for (int i = 0; i < iterations; i++)
            {
                accelerator.Synchronize();
                var stopwatch = Stopwatch.StartNew();
                buffer.CopyFromCPU(data);
                accelerator.Synchronize();
                stopwatch.Stop();
                times.Add(stopwatch.ElapsedMilliseconds);
            }
            
            var avgTime = times.Average();
            var bytes = elementCount * sizeof(float);
            results.HostToDevice = bytes / (avgTime * 1e6); // GB/s

            // Device to Host
            times.Clear();
            for (int i = 0; i < iterations; i++)
            {
                accelerator.Synchronize();
                var stopwatch = Stopwatch.StartNew();
                buffer.GetAsArray1D();
                accelerator.Synchronize();
                stopwatch.Stop();
                times.Add(stopwatch.ElapsedMilliseconds);
            }
            
            avgTime = times.Average();
            results.DeviceToHost = bytes / (avgTime * 1e6); // GB/s

            // Device to Device (simulate with kernel)
            using var buffer2 = accelerator.Allocate1D<float>(elementCount);
            var copyKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(CopyKernel);
            
            times.Clear();
            for (int i = 0; i < iterations; i++)
            {
                accelerator.Synchronize();
                var stopwatch = Stopwatch.StartNew();
                copyKernel(elementCount, buffer.View, buffer2.View);
                accelerator.Synchronize();
                stopwatch.Stop();
                times.Add(stopwatch.ElapsedMilliseconds);
            }
            
            avgTime = times.Average();
            results.DeviceToDevice = bytes / (avgTime * 1e6); // GB/s

            // Allocation time
            times.Clear();
            for (int i = 0; i < iterations; i++)
            {
                var stopwatch = Stopwatch.StartNew();
                using var tempBuffer = accelerator.Allocate1D<float>(elementCount);
                stopwatch.Stop();
                times.Add(stopwatch.ElapsedMilliseconds);
            }
            
            results.AllocationTime = times.Average();

            return results;
        }

        /// <summary>
        /// Benchmarks a specific kernel implementation.
        /// </summary>
        static async Task<double> BenchmarkKernel(Accelerator accelerator, 
            Action<Index1D, ArrayView<float>, ArrayView<float>> kernelAction, float[] inputData)
        {
            const int iterations = 10;
            const int warmupIterations = 3;

            using var input = accelerator.Allocate1D<float>(inputData.Length);
            using var output = accelerator.Allocate1D<float>(inputData.Length);
            
            input.CopyFromCPU(inputData);
            
            var kernel = accelerator.LoadAutoGroupedStreamKernel(kernelAction);

            // Warm-up
            for (int i = 0; i < warmupIterations; i++)
            {
                kernel(inputData.Length, input.View, output.View);
                accelerator.Synchronize();
            }

            // Measurement
            var times = new List<double>();
            for (int i = 0; i < iterations; i++)
            {
                accelerator.Synchronize();
                var stopwatch = Stopwatch.StartNew();
                kernel(inputData.Length, input.View, output.View);
                accelerator.Synchronize();
                stopwatch.Stop();
                times.Add(stopwatch.ElapsedMilliseconds);
            }

            return times.Average();
        }

        /// <summary>
        /// Benchmarks concurrent execution scenarios.
        /// </summary>
        static async Task<ConcurrencyBenchmarkResults> BenchmarkConcurrentExecution(
            Accelerator accelerator, int streamCount, int workloadSize)
        {
            var tasks = new List<Task>();
            var stopwatch = Stopwatch.StartNew();

            // Create concurrent workloads
            for (int i = 0; i < streamCount; i++)
            {
                tasks.Add(Task.Run(() => ExecuteWorkload(accelerator, workloadSize)));
            }

            await Task.WhenAll(tasks);
            stopwatch.Stop();

            var totalOperations = streamCount * workloadSize * 2; // 2 ops per element
            var gflops = totalOperations / (stopwatch.ElapsedMilliseconds * 1e6);
            var efficiency = gflops / (accelerator.NumMultiprocessors * 2.0) * 100; // Rough estimate

            return new ConcurrencyBenchmarkResults
            {
                TotalTime = stopwatch.ElapsedMilliseconds,
                Throughput = gflops,
                Efficiency = efficiency
            };
        }

        /// <summary>
        /// Executes a workload for concurrency testing.
        /// </summary>
        static void ExecuteWorkload(Accelerator accelerator, int size)
        {
            using var input = accelerator.Allocate1D<float>(size);
            using var output = accelerator.Allocate1D<float>(size);
            
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SimpleKernel);
            kernel(size, input.View, output.View);
            accelerator.Synchronize();
        }

        /// <summary>
        /// Benchmarks real-world application scenarios.
        /// </summary>
        static async Task<RealWorldBenchmarkResults> BenchmarkRealWorldScenario(
            Accelerator accelerator, WorkloadScenario scenario)
        {
            const int iterations = 5;
            var times = new List<double>();

            for (int i = 0; i < iterations; i++)
            {
                var stopwatch = Stopwatch.StartNew();
                
                // Simulate the workload
                for (int op = 0; op < scenario.Operations; op++)
                {
                    using var buffer1 = accelerator.Allocate1D<float>(scenario.DataSize);
                    using var buffer2 = accelerator.Allocate1D<float>(scenario.DataSize);
                    
                    var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SimpleKernel);
                    kernel(scenario.DataSize, buffer1.View, buffer2.View);
                    accelerator.Synchronize();
                }
                
                stopwatch.Stop();
                times.Add(stopwatch.ElapsedMilliseconds);
            }

            var avgTime = times.Average();
            var throughput = 1000.0 / avgTime;
            var totalBytes = scenario.DataSize * scenario.Operations * sizeof(float) * 2; // Read + Write
            var memoryBandwidth = totalBytes / (avgTime * 1e6);

            return new RealWorldBenchmarkResults
            {
                Latency = avgTime,
                Throughput = throughput,
                MemoryBandwidth = memoryBandwidth
            };
        }

        #endregion

        #region Helper Methods and Kernels

        static void SimpleKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            output[index] = input[index] * 2.0f + 1.0f;
        }

        static void NaiveKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            output[index] = input[index] * input[index] + input[index] + 1.0f;
        }

        static void VectorizedKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            var value = input[index];
            output[index] = value * value + value + 1.0f;
        }

        static void UnrolledKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            if (index < input.Length - 3)
            {
                var v1 = input[index];
                var v2 = input[index + 1];
                var v3 = input[index + 2];
                var v4 = input[index + 3];
                
                output[index] = v1 * v1 + v1 + 1.0f;
                output[index + 1] = v2 * v2 + v2 + 1.0f;
                output[index + 2] = v3 * v3 + v3 + 1.0f;
                output[index + 3] = v4 * v4 + v4 + 1.0f;
            }
        }

        static void CoalescedKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            // Simulate coalesced memory access pattern
            var value = input[index];
            output[index] = value * value + value + 1.0f;
        }

        static void CopyKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            output[index] = input[index];
        }

        static float[] CreateTestData(int size)
        {
            var data = new float[size];
            var random = new Random(42);
            for (int i = 0; i < size; i++)
            {
                data[i] = (float)random.NextDouble();
            }
            return data;
        }

        static double CalculateGFLOPS(int elementCount, double timeMs)
        {
            var operations = elementCount * 2L; // 2 operations per element
            return operations / (timeMs * 1e6);
        }

        static double CalculateEfficiency(Accelerator accelerator, double achievedGFLOPS)
        {
            var theoreticalGFLOPS = accelerator.NumMultiprocessors * 2.0; // Rough estimate
            return (achievedGFLOPS / theoreticalGFLOPS) * 100;
        }

        static string FormatDataSize(int elements)
        {
            var bytes = elements * sizeof(float);
            if (bytes < 1024) return $"{bytes} B";
            if (bytes < 1024 * 1024) return $"{bytes / 1024} KB";
            return $"{bytes / (1024 * 1024)} MB";
        }

        #endregion

        #region Data Structures

        public class MemoryBenchmarkResults
        {
            public double HostToDevice { get; set; }
            public double DeviceToHost { get; set; }
            public double DeviceToDevice { get; set; }
            public double AllocationTime { get; set; }
        }

        public class ConcurrencyBenchmarkResults
        {
            public double TotalTime { get; set; }
            public double Throughput { get; set; }
            public double Efficiency { get; set; }
        }

        public class RealWorldBenchmarkResults
        {
            public double Latency { get; set; }
            public double Throughput { get; set; }
            public double MemoryBandwidth { get; set; }
        }

        public class WorkloadScenario
        {
            public string Name { get; set; } = "";
            public int Operations { get; set; }
            public int DataSize { get; set; }
        }

        #endregion
    }
}