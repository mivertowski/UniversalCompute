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
// Change License: Apache License, Version 2.0using ILGPU.Apple.NeuralEngine;
using ILGPU.Intel.AMX;
using ILGPU.Intel.NPU;
using ILGPU.Runtime;
using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests.Hardware
{
    /// <summary>
    /// Comprehensive test suite for hardware accelerator implementations.
    /// Tests AMX, NPU, and Apple Neural Engine accelerators.
    /// </summary>
    public class HardwareAcceleratorTests : TestBase
    {
        public HardwareAcceleratorTests(ITestOutputHelper output, TestContext testContext)
            : base(output, testContext)
        {
        }

        #region AMX Tests

        [Fact]
        [KernelMethod(nameof(AMXMatrixMultiplyKernel))]
        public void AMXMatrixMultiply()
        {
            if (!AMXCapabilities.IsAMXSupported())
            {
                Output.WriteLine("AMX not supported on this system - skipping test");
                return;
            }

            using var context = Context.Create()
                .WithDevice<AMXDevice>()
                .EnableAMX();

            if (!context.GetAMXDevices().Any())
            {
                Output.WriteLine("No AMX devices available - skipping test");
                return;
            }

            using var accelerator = context.CreateAMXAccelerator(0);
            using var stream = accelerator.CreateStream();

            const int size = 64;
            var matrixA = Enumerable.Range(0, size * size).Select(i => (float)i).ToArray();
            var matrixB = Enumerable.Range(0, size * size).Select(i => (float)(i * 2)).ToArray();
            var expected = new float[size * size];

            // CPU reference calculation
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                    for (int k = 0; k < size; k++)
                        expected[i * size + j] += matrixA[i * size + k] * matrixB[k * size + j];

            using var bufferA = accelerator.Allocate1D<float>(matrixA);
            using var bufferB = accelerator.Allocate1D<float>(matrixB);
            using var bufferC = accelerator.Allocate1D<float>(size * size);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
                AMXMatrixMultiplyKernel);

            kernel(stream, new Index2D(size, size), bufferA.View, bufferB.View, bufferC.View, size);
            stream.Synchronize();

            var result = bufferC.GetAsArray(stream);
            for (int i = 0; i < Math.Min(10, result.Length); i++)
            {
                Assert.True(Math.Abs(result[i] - expected[i]) < 1e-3f, 
                    $"Mismatch at index {i}: expected {expected[i]}, got {result[i]}");
            }
        }

        static void AMXMatrixMultiplyKernel(
            Index2D index,
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> c,
            int size)
        {
            var x = index.X;
            var y = index.Y;
            
            if (x >= size || y >= size)
                return;

            float sum = 0.0f;
            for (int k = 0; k < size; k++)
                sum += a[x * size + k] * b[k * size + y];
            
            c[x * size + y] = sum;
        }

        [Fact]
        public void AMXCapabilitiesDetection()
        {
            var isSupported = AMXCapabilities.IsAMXSupported();
            var capabilities = AMXCapabilities.Query();

            Output.WriteLine($"AMX Supported: {isSupported}");
            Output.WriteLine($"AMX Capabilities: {capabilities}");

            if (isSupported)
            {
                Assert.True(capabilities.MaxTileRows > 0);
                Assert.True(capabilities.MaxTileColumns > 0);
                Assert.True(capabilities.MaxTileDataSize > 0);
            }
        }

        [Fact]
        public void AMXDeviceEnumeration()
        {
            using var context = Context.Create();
            var devices = context.GetAMXDevices();
            
            Output.WriteLine($"Found {devices.Count()} AMX devices");
            
            foreach (var device in devices)
            {
                Output.WriteLine($"Device: {device.Name}");
                Output.WriteLine($"Max Group Size: {device.MaxGroupSize}");
                Output.WriteLine($"Warp Size: {device.WarpSize}");
                Assert.True(device.MaxGroupSize.Size > 0);
            }
        }

        #endregion

        #region NPU Tests

        [Fact]
        [KernelMethod(nameof(NPUInferenceKernel))]
        public void NPUInference()
        {
            if (!NPUCapabilities.DetectNPU())
            {
                Output.WriteLine("Intel NPU not supported on this system - skipping test");
                return;
            }

            using var context = Context.Create()
                .WithDevice<IntelNPUDevice>()
                .EnableNPU();

            if (!context.GetNPUDevices().Any())
            {
                Output.WriteLine("No NPU devices available - skipping test");
                return;
            }

            using var accelerator = context.CreateNPUAccelerator(0);
            using var stream = accelerator.CreateStream();

            const int inputSize = 224 * 224 * 3; // Typical CNN input
            const int outputSize = 1000; // ImageNet classes

            var input = Enumerable.Range(0, inputSize).Select(i => (float)i / inputSize).ToArray();
            var weights = new float[outputSize * inputSize];
            Random.Shared.NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(weights.AsSpan()));

            using var inputBuffer = accelerator.Allocate1D<float>(input);
            using var weightsBuffer = accelerator.Allocate1D<float>(weights);
            using var outputBuffer = accelerator.Allocate1D<float>(outputSize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
                NPUInferenceKernel);

            kernel(stream, outputSize, inputBuffer.View, weightsBuffer.View, outputBuffer.View, inputSize, outputSize);
            stream.Synchronize();

            var result = outputBuffer.GetAsArray(stream);
            Assert.True(result.Length == outputSize);
            Assert.True(result.Any(x => x != 0)); // Should have some non-zero outputs
        }

        static void NPUInferenceKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> weights,
            ArrayView<float> output,
            int inputSize,
            int outputSize)
        {
            var i = index.X;
            if (i >= outputSize)
                return;

            float sum = 0.0f;
            for (int j = 0; j < inputSize; j++)
                sum += input[j] * weights[i * inputSize + j];
            
            output[i] = Math.Max(0.0f, sum); // ReLU activation
        }

        [Fact]
        public void NPUCapabilitiesDetection()
        {
            var isSupported = NPUCapabilities.DetectNPU();
            var capabilities = NPUCapabilities.Query();

            Output.WriteLine($"NPU Supported: {isSupported}");
            Output.WriteLine($"NPU Capabilities: {capabilities}");

            if (isSupported)
            {
                Assert.True(capabilities.MaxTOPS > 0);
                Assert.True(capabilities.ComputeUnits > 0);
                Assert.True(capabilities.MemoryBandwidth > 0);
            }
        }

        [Fact]
        public void NPUDeviceEnumeration()
        {
            using var context = Context.Create();
            var devices = context.GetNPUDevices();
            
            Output.WriteLine($"Found {devices.Count()} NPU devices");
            
            foreach (var device in devices)
            {
                Output.WriteLine($"Device: {device.Name}");
                Output.WriteLine($"Max Group Size: {device.MaxGroupSize}");
                Output.WriteLine($"Capabilities: {device.Capabilities}");
                Assert.True(device.MaxGroupSize.Size > 0);
            }
        }

        #endregion

        #region Apple Neural Engine Tests

        [Fact]
        [KernelMethod(nameof(ANENeuralNetworkKernel))]
        public void ANENeuralNetwork()
        {
            if (!ANECapabilities.DetectNeuralEngine())
            {
                Output.WriteLine("Apple Neural Engine not supported on this system - skipping test");
                return;
            }

            using var context = Context.Create()
                .WithDevice<AppleNeuralEngineDevice>()
                .EnableANE();

            if (!context.GetANEDevices().Any())
            {
                Output.WriteLine("No ANE devices available - skipping test");
                return;
            }

            using var accelerator = context.CreateANEAccelerator(0);
            using var stream = accelerator.CreateStream();

            const int batchSize = 1;
            const int channels = 3;
            const int height = 224;
            const int width = 224;
            const int filterCount = 64;

            var input = new float[batchSize * channels * height * width];
            var filters = new float[filterCount * channels * 3 * 3]; // 3x3 conv filters
            
            Random.Shared.NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(input.AsSpan()));
            Random.Shared.NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(filters.AsSpan()));

            var outputSize = batchSize * filterCount * height * width;

            using var inputBuffer = accelerator.Allocate1D<float>(input);
            using var filtersBuffer = accelerator.Allocate1D<float>(filters);
            using var outputBuffer = accelerator.Allocate1D<float>(outputSize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
                ANENeuralNetworkKernel);

            kernel(stream, new Index2D(height, width), inputBuffer.View, filtersBuffer.View, outputBuffer.View, 
                   channels, filterCount, height);
            stream.Synchronize();

            var result = outputBuffer.GetAsArray(stream);
            Assert.True(result.Length == outputSize);
            Assert.True(result.Any(x => x != 0)); // Should have some non-zero outputs after convolution
        }

        static void ANENeuralNetworkKernel(
            Index2D index,
            ArrayView<float> input,
            ArrayView<float> filters,
            ArrayView<float> output,
            int channels,
            int filterCount,
            int height)
        {
            var y = index.X;
            var x = index.Y;
            var width = height; // Assume square images

            if (y >= height || x >= width)
                return;

            for (int f = 0; f < filterCount; f++)
            {
                float sum = 0.0f;
                
                // Simple 3x3 convolution
                for (int c = 0; c < channels; c++)
                {
                    for (int fy = -1; fy <= 1; fy++)
                    {
                        for (int fx = -1; fx <= 1; fx++)
                        {
                            int iy = y + fy;
                            int ix = x + fx;
                            
                            if (iy >= 0 && iy < height && ix >= 0 && ix < width)
                            {
                                var inputIdx = c * height * width + iy * width + ix;
                                var filterIdx = f * channels * 9 + c * 9 + (fy + 1) * 3 + (fx + 1);
                                sum += input[inputIdx] * filters[filterIdx];
                            }
                        }
                    }
                }
                
                var outputIdx = f * height * width + y * width + x;
                output[outputIdx] = Math.Max(0.0f, sum); // ReLU activation
            }
        }

        [Fact]
        public void ANECapabilitiesDetection()
        {
            var isSupported = ANECapabilities.DetectNeuralEngine();
            var capabilities = ANECapabilities.Query();

            Output.WriteLine($"ANE Supported: {isSupported}");
            Output.WriteLine($"ANE Capabilities: {capabilities}");

            if (isSupported)
            {
                Assert.True(capabilities.MaxTOPS > 0);
                Assert.True(capabilities.Generation != ANEGeneration.None);
                Assert.True(capabilities.MaxBatchSize > 0);
            }
        }

        [Fact]
        public void ANEDeviceEnumeration()
        {
            using var context = Context.Create();
            var devices = context.GetANEDevices();
            
            Output.WriteLine($"Found {devices.Count()} ANE devices");
            
            foreach (var device in devices)
            {
                Output.WriteLine($"Device: {device.Name}");
                Output.WriteLine($"Max Group Size: {device.MaxGroupSize}");
                Output.WriteLine($"ANE Capabilities: {device.ANECapabilities}");
                Assert.True(device.MaxGroupSize.Size > 0);
            }
        }

        #endregion

        #region Cross-Platform Tests

        [Fact]
        public void HardwareAcceleratorInteroperability()
        {
            using var context = Context.Create();
            
            var amxDevices = context.GetAMXDevices();
            var npuDevices = context.GetNPUDevices();
            var aneDevices = context.GetANEDevices();
            
            Output.WriteLine($"Total hardware accelerators: AMX={amxDevices.Count()}, NPU={npuDevices.Count()}, ANE={aneDevices.Count()}");
            
            // Test that devices don't interfere with each other
            if (amxDevices.Any() && npuDevices.Any())
            {
                using var amxAccelerator = context.CreateAMXAccelerator(0);
                using var npuAccelerator = context.CreateNPUAccelerator(0);
                
                Assert.NotNull(amxAccelerator);
                Assert.NotNull(npuAccelerator);
                Assert.NotEqual(amxAccelerator.AcceleratorType, npuAccelerator.AcceleratorType);
            }
        }

        [Fact]
        public void MemoryBandwidthBenchmark()
        {
            using var context = Context.Create();
            
            // Test memory bandwidth for available accelerators
            var allDevices = context.GetDevices().Where(d => 
                d is AMXDevice || d is IntelNPUDevice || d is AppleNeuralEngineDevice);
            
            foreach (var device in allDevices)
            {
                using var accelerator = context.CreateAccelerator(device);
                using var stream = accelerator.CreateStream();
                
                const int dataSize = 1024 * 1024; // 1MB
                var data = new float[dataSize];
                Random.Shared.NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(data.AsSpan()));
                
                using var buffer = accelerator.Allocate1D<float>(data);
                
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                var result = buffer.GetAsArray(stream);
                stream.Synchronize();
                stopwatch.Stop();
                
                var bandwidthMBps = (dataSize * sizeof(float)) / (stopwatch.Elapsed.TotalSeconds * 1024 * 1024);
                
                Output.WriteLine($"{device.Name}: {bandwidthMBps:F2} MB/s memory bandwidth");
                Assert.True(result.Length == dataSize);
            }
        }

        [Fact]
        public void ProfilingMarkerAccuracy()
        {
            using var context = Context.Create();
            var devices = context.GetDevices().Where(d => 
                d is AMXDevice || d is IntelNPUDevice || d is AppleNeuralEngineDevice);
            
            foreach (var device in devices)
            {
                using var accelerator = context.CreateAccelerator(device);
                using var stream = accelerator.CreateStream();
                
                var marker1 = stream.AddProfilingMarker();
                System.Threading.Thread.Sleep(10); // Small delay
                var marker2 = stream.AddProfilingMarker();
                
                stream.Synchronize();
                
                var elapsed = marker2.MeasureFrom(marker1);
                Output.WriteLine($"{device.Name}: Profiling marker accuracy: {elapsed.TotalMilliseconds:F2}ms");
                
                Assert.True(elapsed.TotalMilliseconds >= 5); // Should be at least 5ms
                Assert.True(elapsed.TotalMilliseconds <= 50); // Should be less than 50ms
            }
        }

        #endregion

        #region Performance Tests

        [Fact]
        public void MatrixMultiplicationPerformance()
        {
            const int[] sizes = { 64, 128, 256, 512 };
            
            using var context = Context.Create();
            var devices = context.GetDevices().Where(d => 
                d is AMXDevice || d is IntelNPUDevice || d is AppleNeuralEngineDevice);
            
            foreach (var device in devices)
            {
                using var accelerator = context.CreateAccelerator(device);
                using var stream = accelerator.CreateStream();
                
                Output.WriteLine($"\nPerformance test for {device.Name}:");
                
                foreach (var size in sizes)
                {
                    var matrixA = new float[size * size];
                    var matrixB = new float[size * size];
                    Random.Shared.NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(matrixA.AsSpan()));
                    Random.Shared.NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(matrixB.AsSpan()));
                    
                    using var bufferA = accelerator.Allocate1D<float>(matrixA);
                    using var bufferB = accelerator.Allocate1D<float>(matrixB);
                    using var bufferC = accelerator.Allocate1D<float>(size * size);
                    
                    var kernel = accelerator.LoadAutoGroupedStreamKernel<
                        Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
                        AMXMatrixMultiplyKernel);
                    
                    var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                    kernel(stream, new Index2D(size, size), bufferA.View, bufferB.View, bufferC.View, size);
                    stream.Synchronize();
                    stopwatch.Stop();
                    
                    var gflops = (2.0 * size * size * size) / (stopwatch.Elapsed.TotalSeconds * 1e9);
                    Output.WriteLine($"  Size {size}x{size}: {stopwatch.Elapsed.TotalMilliseconds:F2}ms, {gflops:F2} GFLOPS");
                }
            }
        }

        #endregion
    }
}