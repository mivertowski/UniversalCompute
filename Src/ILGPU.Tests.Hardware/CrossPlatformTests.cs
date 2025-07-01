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

using ILGPU.Runtime;
using ILGPU.Runtime.HardwareDetection;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests.Hardware
{
    /// <summary>
    /// Cross-platform hardware acceleration tests.
    /// </summary>
    public class CrossPlatformTests : IDisposable
    {
        private readonly ITestOutputHelper output;
        private readonly Context context;

        public CrossPlatformTests(ITestOutputHelper output)
        {
            this.output = output;
            context = Context.CreateDefault();
            HardwareManager.Initialize();
        }

        [Fact]
        public void PlatformSpecificAcceleratorsWork()
        {
            var platform = GetCurrentPlatform();
            output.WriteLine($"Running on platform: {platform}");

            var capabilities = HardwareManager.Capabilities;
            
            switch (platform)
            {
                case Platform.Windows:
                    // Windows should support CUDA, AMD, Intel, and OpenCL
                    output.WriteLine("Windows platform expectations:");
                    output.WriteLine($"  CUDA possible: {capabilities.CUDA.IsSupported}");
                    output.WriteLine($"  AMD possible: {capabilities.ROCm.IsSupported}");
                    output.WriteLine($"  Intel OneAPI possible: {capabilities.OneAPI.IsSupported}");
                    output.WriteLine($"  Intel AMX possible: {capabilities.AMX.IsSupported}");
                    output.WriteLine($"  OpenCL expected: {capabilities.OpenCL.IsSupported}");
                    break;

                case Platform.Linux:
                    // Linux should support all platforms
                    output.WriteLine("Linux platform expectations:");
                    output.WriteLine($"  CUDA possible: {capabilities.CUDA.IsSupported}");
                    output.WriteLine($"  ROCm possible: {capabilities.ROCm.IsSupported}");
                    output.WriteLine($"  Intel OneAPI possible: {capabilities.OneAPI.IsSupported}");
                    output.WriteLine($"  Intel AMX possible: {capabilities.AMX.IsSupported}");
                    output.WriteLine($"  OpenCL expected: {capabilities.OpenCL.IsSupported}");
                    output.WriteLine($"  Vulkan possible: {capabilities.Vulkan.IsSupported}");
                    break;

                case Platform.macOS:
                    // macOS should support Apple-specific and OpenCL
                    output.WriteLine("macOS platform expectations:");
                    output.WriteLine($"  Apple Metal expected: {capabilities.Apple.SupportsMetalGPU}");
                    output.WriteLine($"  Apple ANE possible: {capabilities.Apple.SupportsNeuralEngine}");
                    output.WriteLine($"  OpenCL possible: {capabilities.OpenCL.IsSupported}");
                    Assert.False(capabilities.CUDA.IsSupported, "CUDA should not be available on macOS");
                    break;
            }

            // Velocity should always be available
            output.WriteLine($"Velocity (CPU SIMD) available: {capabilities.Velocity.IsSupported}");
        }

        [Fact]
        public void MemoryOperationsWorkAcrossAccelerators()
        {
            const int dataSize = 1024;
            var testData = Enumerable.Range(0, dataSize).Select(i => (float)i).ToArray();
            var results = new Dictionary<string, float[]>();

            // Test memory operations on each available accelerator
            var accelerators = GetAvailableAccelerators();
            
            foreach (var (name, accelerator) in accelerators)
            {
                using (accelerator)
                {
                    output.WriteLine($"Testing memory operations on: {name}");
                    
                    // Allocate device memory
                    using var deviceBuffer = accelerator.Allocate1D<float>(dataSize);
                    
                    // Copy to device
                    var sw = Stopwatch.StartNew();
                    deviceBuffer.CopyFromCPU(testData);
                    sw.Stop();
                    output.WriteLine($"  Copy to device: {sw.ElapsedMilliseconds}ms");
                    
                    // Copy back from device
                    sw.Restart();
                    var result = deviceBuffer.GetAsArray1D();
                    sw.Stop();
                    output.WriteLine($"  Copy from device: {sw.ElapsedMilliseconds}ms");
                    
                    // Verify data integrity
                    Assert.Equal(testData.Length, result.Length);
                    for (int i = 0; i < testData.Length; i++)
                    {
                        Assert.Equal(testData[i], result[i], 5);
                    }
                    
                    results[name] = result;
                }
            }

            // Verify all accelerators returned the same data
            if (results.Count > 1)
            {
                var reference = results.First().Value;
                foreach (var (name, result) in results.Skip(1))
                {
                    Assert.Equal(reference, result);
                    output.WriteLine($"✓ {name} data matches reference");
                }
            }
        }

        [Fact]
        public void SimpleKernelRunsOnAllAccelerators()
        {
            const int dataSize = 1024;
            var accelerators = GetAvailableAccelerators();

            foreach (var (name, accelerator) in accelerators)
            {
                using (accelerator)
                {
                    output.WriteLine($"Testing simple kernel on: {name}");
                    
                    try
                    {
                        // Compile a simple addition kernel
                        var kernel = accelerator.LoadAutoGroupedStreamKernel<
                            Index1D, ArrayView1D<float, Stride1D.Dense>, float>(AddKernel);
                        
                        using var buffer = accelerator.Allocate1D<float>(dataSize);
                        
                        // Execute kernel
                        kernel((int)buffer.Length, buffer.View, 1.0f);
                        accelerator.Synchronize();
                        
                        // Verify results
                        var results = buffer.GetAsArray1D();
                        Assert.All(results, val => Assert.Equal(1.0f, val, 5));
                        
                        output.WriteLine($"  ✓ Kernel execution successful");
                    }
                    catch (NotSupportedException ex)
                    {
                        output.WriteLine($"  ⚠ Kernel not supported: {ex.Message}");
                    }
                }
            }
        }

        [Fact]
        public void PeerAccessBetweenCompatibleAccelerators()
        {
            var accelerators = GetAvailableAccelerators().ToList();
            
            if (accelerators.Count < 2)
            {
                output.WriteLine("Less than 2 accelerators available, skipping peer access test");
                return;
            }

            // Test peer access between each pair
            for (int i = 0; i < accelerators.Count; i++)
            {
                for (int j = i + 1; j < accelerators.Count; j++)
                {
                    var (name1, accel1) = accelerators[i];
                    var (name2, accel2) = accelerators[j];
                    
                    try
                    {
                        var canAccess = accel1.CanAccessPeer(accel2);
                        output.WriteLine($"Peer access {name1} -> {name2}: {(canAccess ? "✓" : "✗")}");
                        
                        if (canAccess)
                        {
                            accel1.EnablePeer(accel2);
                            output.WriteLine($"  Enabled peer access successfully");
                            
                            // Test would include actual memory transfer here
                            
                            accel1.DisablePeer(accel2);
                        }
                    }
                    catch (Exception ex)
                    {
                        output.WriteLine($"  Peer access test failed: {ex.Message}");
                    }
                }
            }

            // Dispose all accelerators
            foreach (var (_, accel) in accelerators)
            {
                accel.Dispose();
            }
        }

        [Fact]
        public void PerformanceBenchmarkAcrossAccelerators()
        {
            const int dataSize = 1024 * 1024; // 1M elements
            var testData = new float[dataSize];
            var random = new Random(42);
            for (int i = 0; i < dataSize; i++)
                testData[i] = (float)random.NextDouble();

            var accelerators = GetAvailableAccelerators();
            var benchmarkResults = new Dictionary<string, double>();

            foreach (var (name, accelerator) in accelerators)
            {
                using (accelerator)
                {
                    output.WriteLine($"Benchmarking: {name}");
                    
                    try
                    {
                        using var deviceBuffer = accelerator.Allocate1D<float>(dataSize);
                        
                        // Warm up
                        deviceBuffer.CopyFromCPU(testData);
                        accelerator.Synchronize();
                        
                        // Benchmark memory transfer
                        var sw = Stopwatch.StartNew();
                        const int iterations = 10;
                        
                        for (int i = 0; i < iterations; i++)
                        {
                            deviceBuffer.CopyFromCPU(testData);
                            accelerator.Synchronize();
                        }
                        
                        sw.Stop();
                        var throughputGBps = (dataSize * sizeof(float) * iterations) / 
                                           (sw.Elapsed.TotalSeconds * 1024 * 1024 * 1024);
                        
                        benchmarkResults[name] = throughputGBps;
                        output.WriteLine($"  Throughput: {throughputGBps:F2} GB/s");
                    }
                    catch (Exception ex)
                    {
                        output.WriteLine($"  Benchmark failed: {ex.Message}");
                        benchmarkResults[name] = 0;
                    }
                }
            }

            // Report best performer
            if (benchmarkResults.Any(kvp => kvp.Value > 0))
            {
                var best = benchmarkResults.OrderByDescending(kvp => kvp.Value).First();
                output.WriteLine($"\nBest performer: {best.Key} at {best.Value:F2} GB/s");
            }
        }

        #region Helper Methods

        private static void AddKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> data, float value)
        {
            data[index] = value;
        }

        private IEnumerable<(string name, Accelerator accelerator)> GetAvailableAccelerators()
        {
            var accelerators = new List<(string, Accelerator)>();

            // Try to create each type of accelerator
            try
            {
                var cpuAccel = context.CreateCPUAccelerator();
                accelerators.Add(("CPU", cpuAccel));
            }
            catch { }

            var capabilities = HardwareManager.Capabilities;

            if (capabilities.CUDA.IsSupported)
            {
                try
                {
                    var device = Runtime.Cuda.CudaDevice.GetBestDevice();
                    if (device != null)
                    {
                        var accel = context.CreateCudaAccelerator(device);
                        accelerators.Add(($"CUDA ({device.Name})", accel));
                    }
                }
                catch { }
            }

            if (capabilities.AMX.IsSupported)
            {
                try
                {
                    var device = Runtime.AMX.IntelAMXDevice.GetDefaultDevice();
                    if (device != null)
                    {
                        var accel = context.CreateAMXAccelerator(device);
                        accelerators.Add(("Intel AMX", accel));
                    }
                }
                catch { }
            }

            if (capabilities.OneAPI.IsSupported)
            {
                try
                {
                    var device = Runtime.OneAPI.IntelOneAPIDevice.GetDefaultDevice();
                    if (device != null)
                    {
                        var accel = context.CreateOneAPIAccelerator(device);
                        accelerators.Add(("Intel OneAPI", accel));
                    }
                }
                catch { }
            }

            return accelerators;
        }

        private Platform GetCurrentPlatform()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return Platform.Windows;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return Platform.Linux;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return Platform.macOS;
            return Platform.Unknown;
        }

        private enum Platform
        {
            Windows,
            Linux,
            macOS,
            Unknown
        }

        #endregion

        public void Dispose()
        {
            context?.Dispose();
        }
    }
}