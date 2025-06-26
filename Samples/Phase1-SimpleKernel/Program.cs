// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                           Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: Program.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------
//                                ILGPU Phase 1 Foundation
//                     01-SimpleKernel: Your First GPU Kernel
// ---------------------------------------------------------------------------------------

using System;
using ILGPU;
using ILGPU.Runtime;

namespace Phase1.Basic.SimpleKernel
{
    /// <summary>
    /// Demonstrates the fundamental concepts of GPU programming with ILGPU:
    /// - Creating ILGPU context and accelerator
    /// - Writing a simple GPU kernel
    /// - Memory allocation and data transfer
    /// - Kernel compilation and execution
    /// </summary>
    class Program
    {
        /// <summary>
        /// Simple GPU kernel that adds a constant value to each array element.
        /// This kernel runs in parallel - each thread processes one array element.
        /// </summary>
        /// <param name="index">Current thread's global index</param>
        /// <param name="dataView">View of the GPU memory buffer</param>
        /// <param name="constant">Value to add to each element</param>
        static void AddConstantKernel(Index1D index, ArrayView<int> dataView, int constant)
        {
            // Each thread processes exactly one array element
            // The 'index' parameter tells us which element this thread should handle
            dataView[index] = dataView[index] + constant;
        }

        /// <summary>
        /// Advanced kernel demonstrating bounds checking and conditional operations.
        /// Shows defensive programming practices for GPU kernels.
        /// </summary>
        static void SafeMultiplyKernel(Index1D index, ArrayView<float> input, ArrayView<float> output, float multiplier)
        {
            // Always check bounds in GPU kernels to prevent memory access violations
            if (index < input.Length && index < output.Length)
            {
                // Perform computation with additional validation
                float value = input[index];
                
                // Example of conditional GPU computation
                if (value > 0.0f)
                {
                    output[index] = value * multiplier;
                }
                else
                {
                    output[index] = 0.0f; // Handle negative values
                }
            }
        }

        static void Main(string[] args)
        {
            Console.WriteLine("🚀 ILGPU Phase 1: Simple Kernel Demonstration");
            Console.WriteLine("===============================================");
            
            // Step 1: Initialize ILGPU Context
            // The context manages the ILGPU runtime and provides access to accelerators
            using var context = Context.CreateDefault();
            Console.WriteLine("✅ ILGPU context created successfully");

            // Step 2: Create Accelerator
            // Try to create the best available accelerator (GPU preferred, CPU fallback)
            using var accelerator = GetBestAccelerator(context);
            Console.WriteLine($"✅ Using accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");
            Console.WriteLine($"   Memory Size: {accelerator.MemorySize / (1024 * 1024)} MB");
            Console.WriteLine($"   Warp Size: {accelerator.WarpSize}");

            // Demonstrate basic kernel execution
            DemonstrateBasicKernel(accelerator);
            
            // Demonstrate advanced kernel features
            DemonstrateAdvancedKernel(accelerator);
            
            // Show memory management best practices
            DemonstrateMemoryManagement(accelerator);

            Console.WriteLine("\n🎉 Phase 1 Simple Kernel demonstration completed successfully!");
            Console.WriteLine("Next: Explore 02-MemoryManagement for advanced memory patterns");
        }

        /// <summary>
        /// Selects the best available accelerator with graceful fallback.
        /// </summary>
        static Accelerator GetBestAccelerator(Context context)
        {
            // Try CUDA first (NVIDIA GPUs)
            if (context.GetCudaDevices().Any())
            {
                Console.WriteLine("🎯 CUDA GPU detected - using GPU acceleration");
                return context.CreateCudaAccelerator(0);
            }
            
            // Try OpenCL (AMD GPUs, Intel GPUs, other)
            if (context.GetOpenCLDevices().Any())
            {
                Console.WriteLine("🎯 OpenCL device detected - using GPU acceleration");
                return context.CreateOpenCLAccelerator(0);
            }
            
            // Fallback to CPU (always available)
            Console.WriteLine("🎯 Using CPU accelerator (good for debugging)");
            return context.CreateCPUAccelerator(0);
        }

        /// <summary>
        /// Demonstrates basic kernel compilation and execution.
        /// </summary>
        static void DemonstrateBasicKernel(Accelerator accelerator)
        {
            Console.WriteLine("\n📋 Basic Kernel Demonstration");
            Console.WriteLine("-----------------------------");

            // Define problem size
            const int dataSize = 1024;
            const int constantValue = 42;

            // Step 1: Compile the kernel
            // LoadAutoGroupedStreamKernel automatically determines optimal thread group size
            var addKernel = accelerator.LoadAutoGroupedStreamKernel<ArrayView<int>, int>(AddConstantKernel);
            Console.WriteLine("✅ Kernel compiled successfully");

            // Step 2: Allocate GPU memory
            using var gpuBuffer = accelerator.Allocate1D<int>(dataSize);
            Console.WriteLine($"✅ Allocated {dataSize * sizeof(int)} bytes on GPU");

            // Step 3: Initialize data on CPU
            var cpuData = new int[dataSize];
            for (int i = 0; i < dataSize; i++)
            {
                cpuData[i] = i; // Fill with 0, 1, 2, 3, ...
            }

            // Step 4: Copy data to GPU
            gpuBuffer.CopyFromCPU(cpuData);
            Console.WriteLine("✅ Data copied to GPU");

            // Step 5: Execute kernel
            addKernel(gpuBuffer.View, constantValue);
            Console.WriteLine($"✅ Kernel executed: added {constantValue} to each element");

            // Step 6: Copy results back to CPU
            var result = gpuBuffer.GetAsArray1D();
            Console.WriteLine("✅ Results copied back to CPU");

            // Step 7: Verify results
            bool allCorrect = true;
            for (int i = 0; i < Math.Min(10, dataSize); i++)
            {
                int expected = i + constantValue;
                if (result[i] != expected)
                {
                    allCorrect = false;
                    Console.WriteLine($"❌ Error at index {i}: expected {expected}, got {result[i]}");
                }
            }

            if (allCorrect)
            {
                Console.WriteLine($"✅ Verification passed! First 10 results: [{string.Join(", ", result.Take(10))}]");
            }
        }

        /// <summary>
        /// Demonstrates advanced kernel features and best practices.
        /// </summary>
        static void DemonstrateAdvancedKernel(Accelerator accelerator)
        {
            Console.WriteLine("\n🔬 Advanced Kernel Demonstration");
            Console.WriteLine("--------------------------------");

            const int dataSize = 2048;
            const float multiplier = 3.14f;

            // Compile the advanced kernel
            var multiplyKernel = accelerator.LoadAutoGroupedStreamKernel<ArrayView<float>, ArrayView<float>, float>(SafeMultiplyKernel);

            // Allocate separate input and output buffers
            using var inputBuffer = accelerator.Allocate1D<float>(dataSize);
            using var outputBuffer = accelerator.Allocate1D<float>(dataSize);

            // Initialize with varied data including negative values
            var inputData = new float[dataSize];
            var random = new Random(42); // Fixed seed for reproducible results
            for (int i = 0; i < dataSize; i++)
            {
                inputData[i] = (float)(random.NextDouble() * 2.0 - 1.0); // Range [-1, 1]
            }

            // Copy to GPU and execute
            inputBuffer.CopyFromCPU(inputData);
            multiplyKernel(inputBuffer.View, outputBuffer.View, multiplier);
            
            // Synchronize to ensure completion
            accelerator.Synchronize();

            // Analyze results
            var outputData = outputBuffer.GetAsArray1D();
            int positiveCount = outputData.Count(x => x > 0);
            int zeroCount = outputData.Count(x => x == 0);
            
            Console.WriteLine($"✅ Processed {dataSize} elements");
            Console.WriteLine($"   Positive results: {positiveCount}");
            Console.WriteLine($"   Zero results (from negative inputs): {zeroCount}");
            Console.WriteLine($"   Sample results: [{string.Join(", ", outputData.Take(5).Select(x => x.ToString("F2")))}]");
        }

        /// <summary>
        /// Demonstrates proper memory management and resource cleanup.
        /// </summary>
        static void DemonstrateMemoryManagement(Accelerator accelerator)
        {
            Console.WriteLine("\n💾 Memory Management Demonstration");
            Console.WriteLine("----------------------------------");

            // Show memory usage before allocation
            var memoryBefore = accelerator.MemoryInfo;
            Console.WriteLine($"Memory before allocation: {memoryBefore.AvailableMemory / (1024 * 1024)} MB available");

            // Allocate various buffer sizes
            var buffers = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
            
            try
            {
                for (int i = 1; i <= 5; i++)
                {
                    int size = i * 1024 * 256; // 256K, 512K, 768K, 1M, 1.25M elements
                    var buffer = accelerator.Allocate1D<float>(size);
                    buffers.Add(buffer);
                    
                    var memoryAfter = accelerator.MemoryInfo;
                    Console.WriteLine($"   Allocated buffer {i}: {size * sizeof(float) / (1024 * 1024)} MB - " +
                                    $"{memoryAfter.AvailableMemory / (1024 * 1024)} MB remaining");
                }

                Console.WriteLine($"✅ Successfully allocated {buffers.Count} buffers");
            }
            finally
            {
                // Always dispose of GPU memory buffers
                foreach (var buffer in buffers)
                {
                    buffer.Dispose();
                }
                Console.WriteLine("✅ All buffers properly disposed");
            }

            // Verify memory cleanup
            var memoryFinal = accelerator.MemoryInfo;
            Console.WriteLine($"Memory after cleanup: {memoryFinal.AvailableMemory / (1024 * 1024)} MB available");
        }
    }
}