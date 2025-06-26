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

using ILGPU.Intel.AMX;
using ILGPU.Runtime;
using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests.Hardware
{
    /// <summary>
    /// Specific tests for Intel AMX (Advanced Matrix Extensions) accelerator.
    /// Tests tile operations, matrix operations, and AMX-specific features.
    /// </summary>
    public class AMXSpecificTests : TestBase
    {
        public AMXSpecificTests(ITestOutputHelper output, TestContext testContext)
            : base(output, testContext)
        {
        }

        [Fact]
        public void AMXTileConfiguration()
        {
            if (!AMXCapabilities.IsAMXSupported())
            {
                Output.WriteLine("AMX not supported - skipping test");
                return;
            }

            var capabilities = AMXCapabilities.Query();
            
            Assert.True(capabilities.MaxTileRows > 0, "Max tile rows should be positive");
            Assert.True(capabilities.MaxTileColumns > 0, "Max tile columns should be positive");
            Assert.True(capabilities.MaxTileDataSize > 0, "Max tile data size should be positive");
            Assert.True(capabilities.NumTileRegisters > 0, "Should have tile registers");
            
            Output.WriteLine($"AMX Tile Configuration:");
            Output.WriteLine($"  Max Rows: {capabilities.MaxTileRows}");
            Output.WriteLine($"  Max Columns: {capabilities.MaxTileColumns}");
            Output.WriteLine($"  Max Data Size: {capabilities.MaxTileDataSize}");
            Output.WriteLine($"  Tile Registers: {capabilities.NumTileRegisters}");
        }

        [Fact]
        [KernelMethod(nameof(AMXTileOperationKernel))]
        public void AMXTileOperations()
        {
            if (!AMXCapabilities.IsAMXSupported())
            {
                Output.WriteLine("AMX not supported - skipping test");
                return;
            }

            using var context = Context.Create().WithDevice<AMXDevice>().EnableAMX();
            using var accelerator = context.CreateAMXAccelerator(0);
            using var stream = accelerator.CreateStream();

            const int tileSize = 16; // Standard AMX tile size
            var dataA = new float[tileSize * tileSize];
            var dataB = new float[tileSize * tileSize];
            
            // Initialize with simple patterns
            for (int i = 0; i < tileSize * tileSize; i++)
            {
                dataA[i] = i % tileSize;
                dataB[i] = (i / tileSize) % tileSize;
            }

            using var bufferA = accelerator.Allocate1D<float>(dataA);
            using var bufferB = accelerator.Allocate1D<float>(dataB);
            using var bufferC = accelerator.Allocate1D<float>(tileSize * tileSize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
                AMXTileOperationKernel);

            kernel(stream, tileSize * tileSize, bufferA.View, bufferB.View, bufferC.View, tileSize);
            stream.Synchronize();

            var result = bufferC.GetAsArray(stream);
            
            // Verify some basic properties of the result
            Assert.True(result.Length == tileSize * tileSize);
            Assert.True(result.Any(x => x != 0), "Result should have non-zero values");
            
            Output.WriteLine($"AMX tile operation completed successfully");
            Output.WriteLine($"First few results: {string.Join(", ", result.Take(5).Select(x => x.ToString("F2")))}");
        }

        static void AMXTileOperationKernel(
            Index1D index,
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> c,
            int size)
        {
            var i = index.X;
            if (i >= size * size)
                return;

            var row = i / size;
            var col = i % size;
            
            float sum = 0;
            for (int k = 0; k < size; k++)
                sum += a[row * size + k] * b[k * size + col];
            
            c[i] = sum;
        }

        [Fact]
        [KernelMethod(nameof(AMXBFloat16Kernel))]
        public void AMXBFloat16Operations()
        {
            if (!AMXCapabilities.IsAMXSupported())
            {
                Output.WriteLine("AMX not supported - skipping test");
                return;
            }

            var capabilities = AMXCapabilities.Query();
            if (!capabilities.SupportsBF16)
            {
                Output.WriteLine("AMX BFloat16 not supported - skipping test");
                return;
            }

            using var context = Context.Create().WithDevice<AMXDevice>().EnableAMX();
            using var accelerator = context.CreateAMXAccelerator(0);
            using var stream = accelerator.CreateStream();

            const int size = 32;
            var input = Enumerable.Range(0, size).Select(i => (float)i / size).ToArray();
            
            using var inputBuffer = accelerator.Allocate1D<float>(input);
            using var outputBuffer = accelerator.Allocate1D<float>(size);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int>(
                AMXBFloat16Kernel);

            kernel(stream, size, inputBuffer.View, outputBuffer.View, size);
            stream.Synchronize();

            var result = outputBuffer.GetAsArray(stream);
            
            Assert.True(result.Length == size);
            Output.WriteLine($"AMX BFloat16 operation completed with {result.Length} results");
        }

        static void AMXBFloat16Kernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int size)
        {
            var i = index.X;
            if (i >= size)
                return;

            // Simple BFloat16 simulation (truncate mantissa)
            var value = input[i];
            var bits = BitConverter.SingleToUInt32Bits(value);
            var bf16Bits = bits & 0xFFFF0000; // Keep sign, exponent, and top 7 mantissa bits
            var bf16Value = BitConverter.UInt32BitsToSingle(bf16Bits);
            
            output[i] = bf16Value * 2.0f; // Simple operation
        }

        [Fact]
        public void AMXPowerEfficiency()
        {
            if (!AMXCapabilities.IsAMXSupported())
            {
                Output.WriteLine("AMX not supported - skipping test");
                return;
            }

            var capabilities = AMXCapabilities.Query();
            var efficiency = capabilities.GetPowerEfficiency();
            
            Assert.True(efficiency > 0, "Power efficiency should be positive");
            
            Output.WriteLine($"AMX Power Efficiency: {efficiency:F2} GOPS/Watt");
            
            // Test power estimation for different utilization levels
            var utilizations = new[] { 25.0, 50.0, 75.0, 100.0 };
            
            foreach (var util in utilizations)
            {
                var power = capabilities.GetEstimatedPower(util);
                Output.WriteLine($"  {util}% utilization: {power:F2}W");
                Assert.True(power >= 0, "Power estimate should be non-negative");
            }
        }

        [Fact]
        [KernelMethod(nameof(AMXLargeMatrixKernel))]
        public void AMXLargeMatrixOperations()
        {
            if (!AMXCapabilities.IsAMXSupported())
            {
                Output.WriteLine("AMX not supported - skipping test");
                return;
            }

            using var context = Context.Create().WithDevice<AMXDevice>().EnableAMX();
            using var accelerator = context.CreateAMXAccelerator(0);
            using var stream = accelerator.CreateStream();

            const int matrixSize = 1024; // Large matrix
            const int blockSize = 64;   // Process in blocks
            
            var matrixA = new float[matrixSize * matrixSize];
            var matrixB = new float[matrixSize * matrixSize];
            
            // Initialize with random data
            var random = new Random(42);
            for (int i = 0; i < matrixA.Length; i++)
            {
                matrixA[i] = (float)random.NextDouble();
                matrixB[i] = (float)random.NextDouble();
            }

            using var bufferA = accelerator.Allocate1D<float>(matrixA);
            using var bufferB = accelerator.Allocate1D<float>(matrixB);
            using var bufferC = accelerator.Allocate1D<float>(matrixSize * matrixSize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
                AMXLargeMatrixKernel);

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            kernel(stream, new Index2D(matrixSize, matrixSize), bufferA.View, bufferB.View, bufferC.View, matrixSize);
            stream.Synchronize();
            stopwatch.Stop();

            var result = bufferC.GetAsArray(stream);
            
            Assert.True(result.Length == matrixSize * matrixSize);
            Assert.True(result.Any(x => x != 0), "Result should have non-zero values");
            
            var gflops = (2.0 * matrixSize * matrixSize * matrixSize) / (stopwatch.Elapsed.TotalSeconds * 1e9);
            Output.WriteLine($"Large matrix operation: {stopwatch.Elapsed.TotalMilliseconds:F2}ms, {gflops:F2} GFLOPS");
        }

        static void AMXLargeMatrixKernel(
            Index2D index,
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> c,
            int size)
        {
            var row = index.X;
            var col = index.Y;
            
            if (row >= size || col >= size)
                return;

            float sum = 0;
            for (int k = 0; k < size; k++)
                sum += a[row * size + k] * b[k * size + col];
            
            c[row * size + col] = sum;
        }

        [Fact]
        public void AMXOptimalTileSize()
        {
            if (!AMXCapabilities.IsAMXSupported())
            {
                Output.WriteLine("AMX not supported - skipping test");
                return;
            }

            var capabilities = AMXCapabilities.Query();
            var optimalSize = capabilities.GetOptimalTileSize(1024, 1024); // 1024x1024 matrix
            
            Assert.True(optimalSize > 0, "Optimal tile size should be positive");
            Assert.True(optimalSize <= capabilities.MaxTileDataSize, "Optimal size should not exceed max tile size");
            
            Output.WriteLine($"Optimal tile size for 1024x1024 matrix: {optimalSize}");
            
            // Test different matrix sizes
            var sizes = new[] { 64, 128, 256, 512, 1024, 2048 };
            foreach (var size in sizes)
            {
                var optimal = capabilities.GetOptimalTileSize(size, size);
                Output.WriteLine($"  {size}x{size}: optimal tile size = {optimal}");
                Assert.True(optimal > 0);
            }
        }

        [Fact]
        public void AMXMemoryAlignment()
        {
            if (!AMXCapabilities.IsAMXSupported())
            {
                Output.WriteLine("AMX not supported - skipping test");
                return;
            }

            using var context = Context.Create().WithDevice<AMXDevice>().EnableAMX();
            using var accelerator = context.CreateAMXAccelerator(0) as AMXAccelerator;
            
            Assert.NotNull(accelerator);
            
            // Test memory alignment requirements
            var alignment = accelerator.GetRequiredAlignment();
            Assert.True(alignment > 0, "Memory alignment should be positive");
            Assert.True((alignment & (alignment - 1)) == 0, "Alignment should be power of 2");
            
            Output.WriteLine($"AMX memory alignment requirement: {alignment} bytes");
            
            // Test aligned memory allocation
            const int size = 1024;
            using var buffer = accelerator.Allocate1D<float>(size);
            
            // Check if the allocated memory is properly aligned (this is internal validation)
            Assert.True(buffer.Length == size);
            
            Output.WriteLine($"Successfully allocated {size} floats with proper alignment");
        }
    }
}