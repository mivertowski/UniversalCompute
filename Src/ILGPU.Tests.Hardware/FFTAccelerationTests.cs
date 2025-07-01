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

using ILGPU.FFT;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.HardwareDetection;
using System;
using System.Linq;
using System.Numerics;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests.Hardware
{
    /// <summary>
    /// Tests for hardware-accelerated FFT operations.
    /// </summary>
    public class FFTAccelerationTests : IDisposable
    {
        private readonly ITestOutputHelper output;
        private readonly Context context;

        public FFTAccelerationTests(ITestOutputHelper output)
        {
            this.output = output;
            context = Context.CreateDefault();
            HardwareManager.Initialize();
        }

        [Fact]
        public void FFTAcceleratorSelectionWorks()
        {
            // Act
            using var accelerator = HardwareManager.GetBestAccelerator(WorkloadType.FFTOperations, context);

            // Assert
            Assert.NotNull(accelerator);
            output.WriteLine($"Best FFT accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");

            // Verify it's a GPU if available
            var capabilities = HardwareManager.Capabilities;
            if (capabilities.CUDA.IsSupported && capabilities.CUDA.SupportsCUFFT)
            {
                output.WriteLine("CUDA with cuFFT is available");
            }
            else if (capabilities.ROCm.IsSupported && capabilities.ROCm.SupportsROCFFT)
            {
                output.WriteLine("ROCm with rocFFT is available");
            }
            else
            {
                output.WriteLine("Using CPU fallback for FFT");
            }
        }

        [SkippableFact]
        public void CudaFFT1DTest()
        {
            Skip.IfNot(HardwareManager.Capabilities.CUDA.IsSupported, "CUDA not supported");

            // Arrange
            const int length = 1024;
            var device = CudaDevice.GetBestDevice();
            Assert.NotNull(device);

            using var cudaAccel = context.CreateCudaAccelerator(device);
            var fftAccel = new CudaFFTAccelerator(cudaAccel);

            // Create test signal (sine wave)
            var input = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                input[i] = new Complex(Math.Sin(2 * Math.PI * 8 * i / length), 0);
            }
            var output = new Complex[length];

            // Act
            output.WriteLine($"Running cuFFT on {device.Name}");
            fftAccel.Execute1D(input, output, true);

            // Assert - Check for peak at frequency 8
            var magnitudes = output.Select(c => c.Magnitude).ToArray();
            var maxIndex = Array.IndexOf(magnitudes, magnitudes.Max());
            
            // The peak should be at index 8 (for positive frequencies)
            Assert.True(maxIndex == 8 || maxIndex == length - 8, 
                $"Expected peak at frequency 8, found at {maxIndex}");
            
            output.WriteLine($"cuFFT 1D test passed - peak found at index {maxIndex}");
        }

        [Fact]
        public void FFT1DWithBestAcceleratorTest()
        {
            // Arrange
            const int length = 512;
            using var accelerator = HardwareManager.GetBestAccelerator(WorkloadType.FFTOperations, context);
            
            IFFTAccelerator fftAccel = accelerator switch
            {
                CudaAccelerator cuda => new CudaFFTAccelerator(cuda),
                _ => new CPUFFTAccelerator(accelerator)
            };

            // Create test signal - combination of frequencies
            var input = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                input[i] = new Complex(
                    Math.Sin(2 * Math.PI * 10 * i / length) +  // 10 Hz
                    0.5 * Math.Sin(2 * Math.PI * 20 * i / length), // 20 Hz
                    0);
            }
            var output = new Complex[length];

            // Act
            output.WriteLine($"Running FFT on {accelerator.Name}");
            fftAccel.Execute1D(input, output, true);

            // Assert - Find peaks
            var magnitudes = output.Take(length / 2).Select(c => c.Magnitude).ToArray();
            var threshold = magnitudes.Average() * 3; // Peaks should be 3x average
            
            var peaks = magnitudes
                .Select((mag, idx) => new { Magnitude = mag, Index = idx })
                .Where(x => x.Magnitude > threshold)
                .OrderByDescending(x => x.Magnitude)
                .Take(2)
                .ToList();

            Assert.Equal(2, peaks.Count);
            output.WriteLine($"Found {peaks.Count} peaks at frequencies: {string.Join(", ", peaks.Select(p => p.Index))}");
        }

        [Fact]
        public void FFT2DTest()
        {
            // Arrange
            const int width = 64;
            const int height = 64;
            using var accelerator = HardwareManager.GetBestAccelerator(WorkloadType.FFTOperations, context);
            
            IFFTAccelerator fftAccel = accelerator switch
            {
                CudaAccelerator cuda => new CudaFFTAccelerator(cuda),
                _ => new CPUFFTAccelerator(accelerator)
            };

            // Create 2D test pattern - diagonal stripes
            var input = new Complex[height, width];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    input[y, x] = new Complex(Math.Sin(2 * Math.PI * (x + y) / 16), 0);
                }
            }
            var output = new Complex[height, width];

            // Act
            output.WriteLine($"Running 2D FFT on {accelerator.Name}");
            fftAccel.Execute2D(input, output, true);

            // Assert - DC component should be near zero for sine pattern
            var dcComponent = output[0, 0].Magnitude;
            Assert.True(dcComponent < 1.0, $"DC component too large: {dcComponent}");
            
            // Find the peak (should be away from origin)
            double maxMag = 0;
            int maxX = 0, maxY = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var mag = output[y, x].Magnitude;
                    if (mag > maxMag)
                    {
                        maxMag = mag;
                        maxX = x;
                        maxY = y;
                    }
                }
            }

            output.WriteLine($"2D FFT peak found at ({maxX}, {maxY}) with magnitude {maxMag:F2}");
            Assert.True(maxX != 0 || maxY != 0, "Peak should not be at DC");
        }

        [Fact]
        public void InverseFFTTest()
        {
            // Arrange
            const int length = 256;
            using var accelerator = HardwareManager.GetBestAccelerator(WorkloadType.FFTOperations, context);
            
            IFFTAccelerator fftAccel = accelerator switch
            {
                CudaAccelerator cuda => new CudaFFTAccelerator(cuda),
                _ => new CPUFFTAccelerator(accelerator)
            };

            // Create test signal
            var original = new Complex[length];
            var random = new Random(42);
            for (int i = 0; i < length; i++)
            {
                original[i] = new Complex(random.NextDouble() - 0.5, random.NextDouble() - 0.5);
            }

            var forward = new Complex[length];
            var backward = new Complex[length];

            // Act - Forward then inverse FFT
            fftAccel.Execute1D(original, forward, true);
            fftAccel.Execute1D(forward, backward, false);

            // Normalize (FFT then IFFT scales by N)
            for (int i = 0; i < length; i++)
            {
                backward[i] /= length;
            }

            // Assert - Should recover original signal
            for (int i = 0; i < length; i++)
            {
                var diff = (original[i] - backward[i]).Magnitude;
                Assert.True(diff < 1e-6, $"Element {i} differs by {diff}");
            }

            output.WriteLine($"FFT->IFFT round trip successful on {accelerator.Name}");
        }

        [Fact]
        public void BatchedFFTTest()
        {
            // Arrange
            const int length = 128;
            const int batchSize = 4;
            using var accelerator = HardwareManager.GetBestAccelerator(WorkloadType.FFTOperations, context);
            
            IFFTAccelerator fftAccel = accelerator switch
            {
                CudaAccelerator cuda => new CudaFFTAccelerator(cuda),
                _ => new CPUFFTAccelerator(accelerator)
            };

            // Create batch of signals
            var inputs = new Complex[batchSize][];
            var outputs = new Complex[batchSize][];
            
            for (int b = 0; b < batchSize; b++)
            {
                inputs[b] = new Complex[length];
                outputs[b] = new Complex[length];
                
                // Different frequency for each batch
                for (int i = 0; i < length; i++)
                {
                    inputs[b][i] = new Complex(Math.Sin(2 * Math.PI * (b + 1) * 2 * i / length), 0);
                }
            }

            // Act - Process batch
            output.WriteLine($"Running batched FFT ({batchSize} signals) on {accelerator.Name}");
            for (int b = 0; b < batchSize; b++)
            {
                fftAccel.Execute1D(inputs[b], outputs[b], true);
            }

            // Assert - Each should have peak at different frequency
            for (int b = 0; b < batchSize; b++)
            {
                var magnitudes = outputs[b].Select(c => c.Magnitude).ToArray();
                var maxIndex = Array.IndexOf(magnitudes, magnitudes.Max());
                
                // Expected frequency is (b+1)*2
                var expectedFreq = (b + 1) * 2;
                Assert.True(
                    Math.Abs(maxIndex - expectedFreq) <= 1 || 
                    Math.Abs(maxIndex - (length - expectedFreq)) <= 1,
                    $"Batch {b}: Expected peak near {expectedFreq}, found at {maxIndex}");
            }

            output.WriteLine("Batched FFT test passed");
        }

        [Fact]
        public void PerformanceComparisonTest()
        {
            // Compare FFT performance across available accelerators
            const int length = 4096;
            const int iterations = 100;
            
            var input = new Complex[length];
            var random = new Random(42);
            for (int i = 0; i < length; i++)
            {
                input[i] = new Complex(random.NextDouble(), random.NextDouble());
            }

            var results = new System.Collections.Generic.Dictionary<string, double>();

            // Test CPU
            using (var cpuAccel = context.CreateCPUAccelerator())
            {
                var fftAccel = new CPUFFTAccelerator(cpuAccel);
                var output = new Complex[length];
                
                // Warmup
                fftAccel.Execute1D(input, output, true);
                
                // Benchmark
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < iterations; i++)
                {
                    fftAccel.Execute1D(input, output, true);
                }
                sw.Stop();
                
                results["CPU"] = sw.Elapsed.TotalMilliseconds / iterations;
            }

            // Test GPU if available
            if (HardwareManager.Capabilities.CUDA.IsSupported)
            {
                try
                {
                    var device = CudaDevice.GetBestDevice();
                    if (device != null)
                    {
                        using var cudaAccel = context.CreateCudaAccelerator(device);
                        var fftAccel = new CudaFFTAccelerator(cudaAccel);
                        var output = new Complex[length];
                        
                        // Warmup
                        fftAccel.Execute1D(input, output, true);
                        
                        // Benchmark
                        var sw = System.Diagnostics.Stopwatch.StartNew();
                        for (int i = 0; i < iterations; i++)
                        {
                            fftAccel.Execute1D(input, output, true);
                        }
                        sw.Stop();
                        
                        results[$"CUDA ({device.Name})"] = sw.Elapsed.TotalMilliseconds / iterations;
                    }
                }
                catch (Exception ex)
                {
                    output.WriteLine($"CUDA FFT benchmark failed: {ex.Message}");
                }
            }

            // Report results
            output.WriteLine($"\nFFT Performance Results ({length} points, {iterations} iterations):");
            foreach (var (name, time) in results.OrderBy(kvp => kvp.Value))
            {
                output.WriteLine($"  {name}: {time:F3} ms/FFT");
            }

            if (results.Count > 1)
            {
                var fastest = results.OrderBy(kvp => kvp.Value).First();
                var slowest = results.OrderByDescending(kvp => kvp.Value).First();
                var speedup = slowest.Value / fastest.Value;
                output.WriteLine($"\nSpeedup: {fastest.Key} is {speedup:F1}x faster than {slowest.Key}");
            }
        }

        public void Dispose()
        {
            context?.Dispose();
        }
    }
}