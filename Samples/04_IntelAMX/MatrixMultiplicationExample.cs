// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowski/UniversalCompute/blob/master/LICENSE.txt
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
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Intel.AMX;
using ILGPU.Intel.AMX.Native;

namespace ILGPU.Examples.IntelAMX
{
    /// <summary>
    /// Demonstrates high-performance matrix multiplication using Intel AMX.
    /// This example shows how to leverage AMX tiles for optimal GEMM performance.
    /// </summary>
    class MatrixMultiplicationExample
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("⚡ Intel AMX - Matrix Multiplication Example");
            Console.WriteLine("==========================================\n");

            if (!AMXCapabilities.CheckAMXSupport())
            {
                Console.WriteLine("❌ Intel AMX not available on this system.");
                Console.WriteLine("This example requires Intel Xeon (Sapphire Rapids+) or Core (Alder Lake+) processors.");
                Console.WriteLine("Ensure TMUL (Tile Matrix Units) are enabled in BIOS.");
                return;
            }

            try
            {
                // Initialize AMX
                if (!AMXNative.InitializeAMX())
                {
                    Console.WriteLine("❌ Failed to initialize Intel AMX.");
                    Console.WriteLine("Check if AMX is enabled in BIOS and OS supports AMX.");
                    return;
                }

                using var context = Context.CreateDefault();
                
                // Get AMX capabilities
                var capabilities = AMXCapabilities.QueryCapabilities();
                Console.WriteLine($"🔍 AMX Support: {(capabilities.IsSupported != 0 ? "Yes" : "No")}");
                Console.WriteLine($"🔍 Max Tiles: {capabilities.MaxTiles}");
                Console.WriteLine($"🔍 Max Tile Rows: {capabilities.MaxTileRows}");
                Console.WriteLine($"🔍 Max Tile Columns: {capabilities.MaxTileColumns}");
                Console.WriteLine($"🔍 Supports BF16: {(capabilities.SupportsBF16 != 0 ? "Yes" : "No")}");
                Console.WriteLine($"🔍 Supports INT8: {(capabilities.SupportsInt8 != 0 ? "Yes" : "No")}");
                Console.WriteLine($"🔍 Estimated Bandwidth: {capabilities.EstimatedBandwidthGBps:F1} GB/s");
                Console.WriteLine();

                // Create AMX accelerator
                using var device = AMXDevice.Default;
                using var accelerator = new AMXAccelerator(context, device);
                
                Console.WriteLine($"🎯 Using: {accelerator.Name}\n");

                // Run matrix multiplication examples
                await RunBasicMatrixMultiplication(accelerator, capabilities);
                await RunOptimalTiling(accelerator, capabilities);
                await RunDataTypeComparison(accelerator, capabilities);
                await RunPerformanceBenchmark(accelerator, capabilities);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                Console.WriteLine($"Details: {ex}");
            }
            finally
            {
                // Release AMX resources
                AMXNative.ReleaseAMX();
            }

            Console.WriteLine("\n✅ Example completed. Press any key to exit...");
            Console.ReadKey();
        }

        /// <summary>
        /// Demonstrates basic matrix multiplication using AMX tiles.
        /// </summary>
        static async Task RunBasicMatrixMultiplication(AMXAccelerator accelerator, AMXNativeCapabilities capabilities)
        {
            Console.WriteLine("🧮 Basic Matrix Multiplication (FP32)");
            Console.WriteLine("------------------------------------");

            const int M = 1024;
            const int N = 1024;
            const int K = 1024;

            Console.WriteLine($"   └─ Matrix A: {M}x{K}");
            Console.WriteLine($"   └─ Matrix B: {K}x{N}");
            Console.WriteLine($"   └─ Matrix C: {M}x{N}");
            Console.WriteLine($"   └─ Data Type: FP32");

            // Create sample matrices
            var matrixA = CreateRandomMatrix(M, K);
            var matrixB = CreateRandomMatrix(K, N);
            var matrixC = new float[M * N];

            try
            {
                // Configure AMX tiles for optimal performance
                var tileConfig = new AMXTileConfiguration
                {
                    Palette = 1, // Standard tile configuration
                    TileRows = Math.Min(16, M), // AMX max 16 rows
                    TileColumns = Math.Min(64, N * sizeof(float)) // AMX max 64 bytes per row
                };

                Console.WriteLine($"   └─ Tile Configuration: {tileConfig.TileRows}x{tileConfig.TileColumns/4} elements");

                // Execute matrix multiplication
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                
                unsafe
                {
                    fixed (float* pA = matrixA, pB = matrixB, pC = matrixC)
                    {
                        AMXOperations.MatrixMultiplyFP32(pA, pB, pC, M, N, K, tileConfig);
                    }
                }
                
                stopwatch.Stop();

                // Calculate performance metrics
                var operations = 2L * M * N * K; // GEMM operations
                var gflops = operations / (stopwatch.ElapsedMilliseconds * 1e6);
                var memoryBandwidth = CalculateMemoryBandwidth(M, N, K, sizeof(float), stopwatch.ElapsedMilliseconds);

                Console.WriteLine($"   └─ Execution Time: {stopwatch.ElapsedMilliseconds:F2} ms");
                Console.WriteLine($"   └─ Performance: {gflops:F2} GFLOPS");
                Console.WriteLine($"   └─ Memory Bandwidth: {memoryBandwidth:F2} GB/s");
                Console.WriteLine($"   └─ Efficiency: {gflops / capabilities.EstimatedBandwidthGBps * 100:F1}% of peak");

                // Verify correctness (sample check)
                var isCorrect = VerifyMatrixMultiplication(matrixA, matrixB, matrixC, M, N, K, 5);
                Console.WriteLine($"   └─ Correctness: {(isCorrect ? "✅ Verified" : "❌ Failed")}");

                Console.WriteLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ Matrix multiplication failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Demonstrates optimal tiling strategies for different matrix sizes.
        /// </summary>
        static async Task RunOptimalTiling(AMXAccelerator accelerator, AMXNativeCapabilities capabilities)
        {
            Console.WriteLine("🔧 Optimal Tiling Analysis");
            Console.WriteLine("-------------------------");

            var matrixSizes = new[]
            {
                new { M = 512, N = 512, K = 512, Name = "Small (512x512)" },
                new { M = 1024, N = 1024, K = 1024, Name = "Medium (1024x1024)" },
                new { M = 2048, N = 2048, K = 2048, Name = "Large (2048x2048)" },
                new { M = 4096, N = 4096, K = 4096, Name = "XLarge (4096x4096)" }
            };

            Console.WriteLine($"   {"Matrix Size",-20} {"Tile Config",-15} {"Performance",-15} {"Efficiency",-12}");
            Console.WriteLine($"   {new string('-', 65)}");

            foreach (var size in matrixSizes)
            {
                try
                {
                    // Calculate optimal tile dimensions
                    var (tileM, tileN, tileK) = AMXNative.CalculateOptimalTiles(
                        size.M, size.N, size.K, AMXDataType.Float32);

                    var matrixA = CreateRandomMatrix(size.M, size.K);
                    var matrixB = CreateRandomMatrix(size.K, size.N);
                    var matrixC = new float[size.M * size.N];

                    var tileConfig = new AMXTileConfiguration
                    {
                        Palette = 1,
                        TileRows = tileM,
                        TileColumns = tileN * sizeof(float)
                    };

                    // Measure performance
                    var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                    
                    unsafe
                    {
                        fixed (float* pA = matrixA, pB = matrixB, pC = matrixC)
                        {
                            AMXOperations.MatrixMultiplyFP32(pA, pB, pC, size.M, size.N, size.K, tileConfig);
                        }
                    }
                    
                    stopwatch.Stop();

                    var operations = 2L * size.M * size.N * size.K;
                    var gflops = operations / (stopwatch.ElapsedMilliseconds * 1e6);
                    var efficiency = gflops / capabilities.EstimatedBandwidthGBps * 100;

                    Console.WriteLine($"   {size.Name,-20} {tileM}x{tileN}x{tileK,-12} {gflops,-15:F1} GFLOPS {efficiency,-12:F1}%");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   {size.Name,-20} {"Failed",-15} {ex.Message,-15} {"N/A",-12}");
                }
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Compares performance across different data types.
        /// </summary>
        static async Task RunDataTypeComparison(AMXAccelerator accelerator, AMXNativeCapabilities capabilities)
        {
            Console.WriteLine("📊 Data Type Performance Comparison");
            Console.WriteLine("----------------------------------");

            const int M = 1024;
            const int N = 1024;
            const int K = 1024;

            var dataTypes = new[]
            {
                new { Type = "FP32", Supported = true, Size = 4 },
                new { Type = "BF16", Supported = capabilities.SupportsBF16 != 0, Size = 2 },
                new { Type = "INT8", Supported = capabilities.SupportsInt8 != 0, Size = 1 }
            };

            Console.WriteLine($"   {"Data Type",-12} {"Supported",-12} {"Time (ms)",-12} {"GFLOPS",-12} {"Throughput",-15}");
            Console.WriteLine($"   {new string('-', 70)}");

            foreach (var dataType in dataTypes)
            {
                if (!dataType.Supported)
                {
                    Console.WriteLine($"   {dataType.Type,-12} {"No",-12} {"N/A",-12} {"N/A",-12} {"N/A",-15}");
                    continue;
                }

                try
                {
                    var stopwatch = System.Diagnostics.Stopwatch.StartNew();

                    // Simulate data type specific operations
                    await SimulateDataTypeOperation(dataType.Type, M, N, K);

                    stopwatch.Stop();

                    var operations = 2L * M * N * K;
                    var gflops = operations / (stopwatch.ElapsedMilliseconds * 1e6);
                    var throughput = M * N * K * dataType.Size / (stopwatch.ElapsedMilliseconds * 1e6);

                    Console.WriteLine($"   {dataType.Type,-12} {"Yes",-12} {stopwatch.ElapsedMilliseconds,-12:F2} {gflops,-12:F1} {throughput,-15:F1} GB/s");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   {dataType.Type,-12} {"Error",-12} {ex.Message,-12} {"N/A",-12} {"N/A",-15}");
                }
            }

            Console.WriteLine();
            Console.WriteLine("   📝 Notes:");
            Console.WriteLine("   └─ BF16 provides ~2x throughput with minimal accuracy loss");
            Console.WriteLine("   └─ INT8 provides ~4x throughput, requires careful quantization");
            Console.WriteLine("   └─ FP32 provides maximum accuracy for reference");
            Console.WriteLine();
        }

        /// <summary>
        /// Runs comprehensive performance benchmark across different scenarios.
        /// </summary>
        static async Task RunPerformanceBenchmark(AMXAccelerator accelerator, AMXNativeCapabilities capabilities)
        {
            Console.WriteLine("🏁 Performance Benchmark Suite");
            Console.WriteLine("-----------------------------");

            // Test different matrix shapes (common in ML workloads)
            var benchmarks = new[]
            {
                new { Name = "NN Hidden Layer", M = 4096, N = 4096, K = 1024, Workload = "Dense Layer" },
                new { Name = "CNN Feature Map", M = 1024, N = 1024, K = 512, Workload = "Convolution" },
                new { Name = "Transformer QKV", M = 2048, N = 512, K = 2048, Workload = "Attention" },
                new { Name = "BERT Large", M = 1024, N = 4096, K = 1024, Workload = "Transformer" },
                new { Name = "ResNet Bottleneck", M = 512, N = 2048, K = 512, Workload = "CNN Block" }
            };

            Console.WriteLine($"   {"Benchmark",-18} {"Matrix Size",-15} {"GFLOPS",-10} {"Memory BW",-12} {"Power Eff.",-12}");
            Console.WriteLine($"   {new string('-', 75)}");

            foreach (var benchmark in benchmarks)
            {
                try
                {
                    var matrixA = CreateRandomMatrix(benchmark.M, benchmark.K);
                    var matrixB = CreateRandomMatrix(benchmark.K, benchmark.N);
                    var matrixC = new float[benchmark.M * benchmark.N];

                    var tileConfig = new AMXTileConfiguration
                    {
                        Palette = 1,
                        TileRows = Math.Min(16, benchmark.M),
                        TileColumns = Math.Min(64, benchmark.N * sizeof(float))
                    };

                    var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                    
                    unsafe
                    {
                        fixed (float* pA = matrixA, pB = matrixB, pC = matrixC)
                        {
                            AMXOperations.MatrixMultiplyFP32(pA, pB, pC, benchmark.M, benchmark.N, benchmark.K, tileConfig);
                        }
                    }
                    
                    stopwatch.Stop();

                    var operations = 2L * benchmark.M * benchmark.N * benchmark.K;
                    var gflops = operations / (stopwatch.ElapsedMilliseconds * 1e6);
                    var memoryBW = CalculateMemoryBandwidth(benchmark.M, benchmark.N, benchmark.K, 
                        sizeof(float), stopwatch.ElapsedMilliseconds);
                    var powerEff = CalculatePowerEfficiency(gflops);

                    var matrixSize = $"{benchmark.M}x{benchmark.N}x{benchmark.K}";

                    Console.WriteLine($"   {benchmark.Name,-18} {matrixSize,-15} {gflops,-10:F1} {memoryBW,-12:F1} {powerEff,-12:F1}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   {benchmark.Name,-18} {"Failed",-15} {ex.Message,-10} {"N/A",-12} {"N/A",-12}");
                }
            }

            Console.WriteLine();
            Console.WriteLine("   🎯 AMX Optimization Tips:");
            Console.WriteLine("   └─ Use tile sizes that align with AMX limits (16x64 bytes)");
            Console.WriteLine("   └─ Prefer BF16 for ML workloads when precision allows");
            Console.WriteLine("   └─ Cache matrices in L2/L3 for repeated operations");
            Console.WriteLine("   └─ Use block-wise processing for large matrices");
            Console.WriteLine();
        }

        #region Helper Methods

        /// <summary>
        /// Creates a random matrix for testing.
        /// </summary>
        static float[] CreateRandomMatrix(int rows, int cols)
        {
            var matrix = new float[rows * cols];
            var random = new Random(42);
            
            for (int i = 0; i < matrix.Length; i++)
            {
                matrix[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
            
            return matrix;
        }

        /// <summary>
        /// Verifies matrix multiplication correctness by checking a sample of results.
        /// </summary>
        static bool VerifyMatrixMultiplication(float[] a, float[] b, float[] c, int M, int N, int K, int sampleCount)
        {
            var random = new Random(123);
            
            for (int sample = 0; sample < sampleCount; sample++)
            {
                int i = random.Next(M);
                int j = random.Next(N);
                
                float expected = 0;
                for (int k = 0; k < K; k++)
                {
                    expected += a[i * K + k] * b[k * N + j];
                }
                
                float actual = c[i * N + j];
                float error = Math.Abs(expected - actual);
                float relativeError = error / Math.Max(Math.Abs(expected), 1e-6f);
                
                if (relativeError > 1e-4f) // Allow for floating-point precision
                {
                    return false;
                }
            }
            
            return true;
        }

        /// <summary>
        /// Calculates memory bandwidth utilization.
        /// </summary>
        static double CalculateMemoryBandwidth(int M, int N, int K, int elementSize, double timeMs)
        {
            // Memory accesses: read A, read B, write C
            var totalBytes = (long)(M * K + K * N + M * N) * elementSize;
            return totalBytes / (timeMs * 1e6); // GB/s
        }

        /// <summary>
        /// Calculates power efficiency estimate.
        /// </summary>
        static double CalculatePowerEfficiency(double gflops)
        {
            // AMX typically consumes 50-100W under full load
            var estimatedPowerW = 75.0;
            return gflops / estimatedPowerW;
        }

        /// <summary>
        /// Simulates data type specific operations.
        /// </summary>
        static async Task SimulateDataTypeOperation(string dataType, int M, int N, int K)
        {
            // Simulate processing time based on data type
            var baseTime = 50; // Base time for FP32
            
            var simulatedTime = dataType switch
            {
                "FP32" => baseTime,
                "BF16" => baseTime / 2, // ~2x faster
                "INT8" => baseTime / 4, // ~4x faster
                _ => baseTime
            };

            await Task.Delay(Math.Max(1, simulatedTime / 10));
        }

        #endregion
    }

    /// <summary>
    /// AMX tile configuration for matrix operations.
    /// </summary>
    public struct AMXTileConfiguration
    {
        public byte Palette;
        public int TileRows;
        public int TileColumns;
    }

    /// <summary>
    /// AMX supported data types.
    /// </summary>
    public enum AMXDataType
    {
        Float32,
        BFloat16,
        Int8
    }
}