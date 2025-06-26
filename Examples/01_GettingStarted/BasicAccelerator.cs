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
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Apple.NeuralEngine;
using ILGPU.Intel.NPU;
using ILGPU.Intel.AMX;

namespace ILGPU.Examples.GettingStarted
{
    /// <summary>
    /// Basic hardware accelerator detection and setup example.
    /// This example demonstrates how to detect and initialize different hardware accelerators.
    /// </summary>
    class BasicAccelerator
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("🚀 ILGPU Hardware Accelerator Detection");
            Console.WriteLine("=====================================\n");

            try
            {
                // Create ILGPU context
                using var context = Context.CreateDefault();
                
                // Detect and display available hardware accelerators
                await DetectHardwareAccelerators(context);
                
                // Demonstrate basic accelerator usage
                await DemonstrateBasicUsage(context);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                Console.WriteLine($"Details: {ex}");
            }

            Console.WriteLine("\n✅ Example completed. Press any key to exit...");
            Console.ReadKey();
        }

        /// <summary>
        /// Detects and displays information about available hardware accelerators.
        /// </summary>
        static async Task DetectHardwareAccelerators(Context context)
        {
            Console.WriteLine("🔍 Detecting Hardware Accelerators...\n");

            // Check for Apple Neural Engine
            await CheckAppleNeuralEngine();
            
            // Check for Intel NPU
            await CheckIntelNPU();
            
            // Check for Intel AMX
            await CheckIntelAMX();
            
            // Check for standard accelerators
            CheckStandardAccelerators(context);
        }

        /// <summary>
        /// Checks for Apple Neural Engine availability.
        /// </summary>
        static async Task CheckAppleNeuralEngine()
        {
            try
            {
                bool isAvailable = ANECapabilities.DetectNeuralEngine();
                Console.WriteLine($"🧠 Apple Neural Engine: {(isAvailable ? "✅ Available" : "❌ Not Available")}");
                
                if (isAvailable)
                {
                    var capabilities = ANECapabilities.Query();
                    Console.WriteLine($"   └─ Generation: {capabilities.Generation}");
                    Console.WriteLine($"   └─ Max TOPS: {capabilities.MaxTOPS:F1}");
                    Console.WriteLine($"   └─ Supports FP16: {capabilities.SupportsFloat16}");
                    Console.WriteLine($"   └─ Supports INT8: {capabilities.SupportsInt8}");
                    Console.WriteLine($"   └─ Power Efficiency: {capabilities.GetPowerEfficiency():F1} TOPS/W");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"🧠 Apple Neural Engine: ❌ Error checking availability - {ex.Message}");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Checks for Intel NPU availability.
        /// </summary>
        static async Task CheckIntelNPU()
        {
            try
            {
                bool isAvailable = NPUCapabilities.DetectNPU();
                Console.WriteLine($"🔧 Intel NPU: {(isAvailable ? "✅ Available" : "❌ Not Available")}");
                
                if (isAvailable)
                {
                    var capabilities = NPUCapabilities.Query();
                    Console.WriteLine($"   └─ Generation: {capabilities.Generation}");
                    Console.WriteLine($"   └─ Max TOPS: {capabilities.MaxTOPS:F1}");
                    Console.WriteLine($"   └─ Compute Units: {capabilities.ComputeUnits}");
                    Console.WriteLine($"   └─ Memory Bandwidth: {capabilities.MemoryBandwidth:F1} GB/s");
                    Console.WriteLine($"   └─ Supports BF16: {capabilities.SupportsBF16}");
                    Console.WriteLine($"   └─ Supports INT8: {capabilities.SupportsInt8}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"🔧 Intel NPU: ❌ Error checking availability - {ex.Message}");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Checks for Intel AMX availability.
        /// </summary>
        static async Task CheckIntelAMX()
        {
            try
            {
                bool isAvailable = AMXCapabilities.CheckAMXSupport();
                Console.WriteLine($"⚡ Intel AMX: {(isAvailable ? "✅ Available" : "❌ Not Available")}");
                
                if (isAvailable)
                {
                    var capabilities = AMXCapabilities.QueryCapabilities();
                    Console.WriteLine($"   └─ Max Tiles: {capabilities.MaxTiles}");
                    Console.WriteLine($"   └─ Max Tile Rows: {capabilities.MaxTileRows}");
                    Console.WriteLine($"   └─ Max Tile Columns: {capabilities.MaxTileColumns}");
                    Console.WriteLine($"   └─ Supports BF16: {(capabilities.SupportsBF16 != 0 ? "Yes" : "No")}");
                    Console.WriteLine($"   └─ Supports INT8: {(capabilities.SupportsInt8 != 0 ? "Yes" : "No")}");
                    Console.WriteLine($"   └─ Estimated Bandwidth: {capabilities.EstimatedBandwidthGBps:F1} GB/s");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚡ Intel AMX: ❌ Error checking availability - {ex.Message}");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Checks for standard ILGPU accelerators (CPU, CUDA, OpenCL).
        /// </summary>
        static void CheckStandardAccelerators(Context context)
        {
            Console.WriteLine("💻 Standard Accelerators:");
            
            // CPU accelerator is always available
            Console.WriteLine("   └─ CPU: ✅ Available");
            
            // Check for CUDA
            try
            {
                foreach (var device in context.GetCudaDevices())
                {
                    Console.WriteLine($"   └─ CUDA Device {device.DeviceId}: ✅ {device.Name}");
                    Console.WriteLine($"      └─ Compute Capability: {device.ComputeCapability}");
                    Console.WriteLine($"      └─ Memory: {device.MemorySize / (1024*1024*1024):F1} GB");
                }
            }
            catch
            {
                Console.WriteLine("   └─ CUDA: ❌ Not Available");
            }

            // Check for OpenCL
            try
            {
                foreach (var device in context.GetCLDevices())
                {
                    Console.WriteLine($"   └─ OpenCL Device {device.DeviceId}: ✅ {device.Name}");
                    Console.WriteLine($"      └─ Memory: {device.MemorySize / (1024*1024*1024):F1} GB");
                }
            }
            catch
            {
                Console.WriteLine("   └─ OpenCL: ❌ Not Available");
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Demonstrates basic accelerator usage with simple kernels.
        /// </summary>
        static async Task DemonstrateBasicUsage(Context context)
        {
            Console.WriteLine("🎯 Basic Accelerator Usage Demonstration\n");

            // Try to use the best available accelerator
            Accelerator accelerator = null;
            string acceleratorType = "";

            try
            {
                // Priority order: ANE -> NPU -> AMX -> CUDA -> OpenCL -> CPU
                if (ANECapabilities.DetectNeuralEngine())
                {
                    accelerator = new AppleNeuralEngineAccelerator(context, AppleNeuralEngineDevice.Default);
                    acceleratorType = "Apple Neural Engine";
                }
                else if (NPUCapabilities.DetectNPU())
                {
                    accelerator = new IntelNPUAccelerator(context, IntelNPUDevice.Default);
                    acceleratorType = "Intel NPU";
                }
                else if (AMXCapabilities.CheckAMXSupport())
                {
                    accelerator = new AMXAccelerator(context, AMXDevice.Default);
                    acceleratorType = "Intel AMX";
                }
                else if (context.GetCudaDevices().Length > 0)
                {
                    accelerator = context.CreateCudaAccelerator(0);
                    acceleratorType = "CUDA GPU";
                }
                else if (context.GetCLDevices().Length > 0)
                {
                    accelerator = context.CreateCLAccelerator(0);
                    acceleratorType = "OpenCL";
                }
                else
                {
                    accelerator = context.CreateCPUAccelerator();
                    acceleratorType = "CPU";
                }

                Console.WriteLine($"🎯 Using: {acceleratorType}");
                Console.WriteLine($"   └─ Device: {accelerator.Name}");
                Console.WriteLine($"   └─ Memory: {accelerator.MemorySize / (1024*1024*1024):F1} GB");
                Console.WriteLine($"   └─ Max Group Size: {accelerator.MaxGroupSize}");
                Console.WriteLine($"   └─ Warp Size: {accelerator.WarpSize}");

                // Run a simple vector addition kernel
                await RunVectorAddition(accelerator);
            }
            finally
            {
                accelerator?.Dispose();
            }
        }

        /// <summary>
        /// Runs a simple vector addition kernel to demonstrate basic accelerator usage.
        /// </summary>
        static async Task RunVectorAddition(Accelerator accelerator)
        {
            Console.WriteLine("\n🧮 Running Vector Addition Kernel...");

            const int dataSize = 1024;
            
            // Create and compile the kernel
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                VectorAddKernel);

            // Allocate memory buffers
            using var bufferA = accelerator.Allocate1D<float>(dataSize);
            using var bufferB = accelerator.Allocate1D<float>(dataSize);
            using var bufferC = accelerator.Allocate1D<float>(dataSize);

            // Initialize input data
            var inputA = new float[dataSize];
            var inputB = new float[dataSize];
            for (int i = 0; i < dataSize; i++)
            {
                inputA[i] = i * 2.0f;
                inputB[i] = i * 3.0f;
            }

            // Copy data to accelerator
            bufferA.CopyFromCPU(inputA);
            bufferB.CopyFromCPU(inputB);

            // Execute kernel
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            kernel(dataSize, bufferA.View, bufferB.View, bufferC.View);
            accelerator.Synchronize();
            stopwatch.Stop();

            // Copy result back
            var result = bufferC.GetAsArray1D();

            // Verify results
            bool isCorrect = true;
            for (int i = 0; i < Math.Min(10, dataSize); i++)
            {
                float expected = inputA[i] + inputB[i];
                if (Math.Abs(result[i] - expected) > 1e-6f)
                {
                    isCorrect = false;
                    break;
                }
            }

            Console.WriteLine($"   └─ Data Size: {dataSize} elements");
            Console.WriteLine($"   └─ Execution Time: {stopwatch.ElapsedMilliseconds} ms");
            Console.WriteLine($"   └─ Performance: {(dataSize * 2.0 / stopwatch.ElapsedMilliseconds / 1000):F2} GFLOPS");
            Console.WriteLine($"   └─ Result: {(isCorrect ? "✅ Correct" : "❌ Incorrect")}");
            
            // Show sample results
            Console.WriteLine($"   └─ Sample: A[0]={inputA[0]:F1} + B[0]={inputB[0]:F1} = C[0]={result[0]:F1}");
        }

        /// <summary>
        /// Simple vector addition kernel.
        /// </summary>
        static void VectorAddKernel(
            Index1D index,
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> c)
        {
            c[index] = a[index] + b[index];
        }
    }
}