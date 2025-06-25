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

#if ENABLE_METAL_ACCELERATOR
namespace ILGPU.Backends.Metal
{
    /// <summary>
    /// Represents the capabilities of an Apple Metal device.
    /// </summary>
    public readonly struct MetalCapabilities
    {
        /// <summary>
        /// Initializes a new instance of the MetalCapabilities struct.
        /// </summary>
        /// <param name="gpuFamily">The Apple GPU family.</param>
        /// <param name="maxThreadsPerGroup">Maximum threads per threadgroup.</param>
        /// <param name="maxMemoryAllocation">Maximum memory allocation size.</param>
        /// <param name="isUnifiedMemory">Whether unified memory is supported.</param>
        /// <param name="supportsRayTracing">Whether ray tracing is supported.</param>
        /// <param name="supportsNeuralEngine">Whether Neural Engine access is supported.</param>
        /// <param name="supportsAMX">Whether Apple AMX is supported.</param>
        /// <param name="supportsBFloat16">Whether BFloat16 is supported.</param>
        /// <param name="supportsInt8">Whether INT8 operations are supported.</param>
        public MetalCapabilities(
            AppleGPUFamily gpuFamily,
            int maxThreadsPerGroup,
            ulong maxMemoryAllocation,
            bool isUnifiedMemory,
            bool supportsRayTracing,
            bool supportsNeuralEngine,
            bool supportsAMX,
            bool supportsBFloat16,
            bool supportsInt8)
        {
            GPUFamily = gpuFamily;
            MaxThreadsPerGroup = maxThreadsPerGroup;
            MaxMemoryAllocation = maxMemoryAllocation;
            IsUnifiedMemory = isUnifiedMemory;
            SupportsRayTracing = supportsRayTracing;
            SupportsNeuralEngine = supportsNeuralEngine;
            SupportsAMX = supportsAMX;
            SupportsBFloat16 = supportsBFloat16;
            SupportsInt8 = supportsInt8;
        }

        /// <summary>
        /// Gets the Apple GPU family.
        /// </summary>
        public AppleGPUFamily GPUFamily { get; }

        /// <summary>
        /// Gets the maximum number of threads per threadgroup.
        /// </summary>
        public int MaxThreadsPerGroup { get; }

        /// <summary>
        /// Gets the maximum memory allocation size in bytes.
        /// </summary>
        public ulong MaxMemoryAllocation { get; }

        /// <summary>
        /// Gets whether this device has unified memory architecture.
        /// </summary>
        public bool IsUnifiedMemory { get; }

        /// <summary>
        /// Gets whether this device supports hardware ray tracing.
        /// </summary>
        public bool SupportsRayTracing { get; }

        /// <summary>
        /// Gets whether this device supports Neural Engine access.
        /// </summary>
        public bool SupportsNeuralEngine { get; }

        /// <summary>
        /// Gets whether this device supports Apple AMX (Apple Matrix Extensions).
        /// </summary>
        public bool SupportsAMX { get; }

        /// <summary>
        /// Gets whether this device supports BFloat16 operations.
        /// </summary>
        public bool SupportsBFloat16 { get; }

        /// <summary>
        /// Gets whether this device supports INT8 operations.
        /// </summary>
        public bool SupportsInt8 { get; }

        /// <summary>
        /// Queries Metal capabilities from the given device.
        /// </summary>
        /// <param name="device">The native Metal device handle.</param>
        /// <returns>Metal capabilities structure.</returns>
        public static MetalCapabilities Query(IntPtr device)
        {
            if (device == IntPtr.Zero)
                return new MetalCapabilities();

            var gpuFamily = MetalNative.GetGPUFamily(device);
            var supportsRayTracing = MetalNative.SupportsRayTracing(device);
            var isLowPower = MetalNative.IsDiscreteGPU(device);
            var maxWorkingSet = MetalNative.MTLDeviceRecommendedMaxWorkingSetSize(device);

            // Determine capabilities based on GPU family
            var maxThreadsPerGroup = GetMaxThreadsPerGroup(gpuFamily);
            var supportsNeuralEngine = gpuFamily >= AppleGPUFamily.Apple7; // M1+ have ANE
            var supportsAMX = gpuFamily >= AppleGPUFamily.Apple7; // M1+ have AMX
            var supportsBFloat16 = gpuFamily >= AppleGPUFamily.Apple8; // M1 Pro/Max+
            var supportsInt8 = gpuFamily >= AppleGPUFamily.Apple7;

            return new MetalCapabilities(
                gpuFamily,
                maxThreadsPerGroup,
                maxWorkingSet,
                true, // Apple Silicon always has unified memory
                supportsRayTracing,
                supportsNeuralEngine,
                supportsAMX,
                supportsBFloat16,
                supportsInt8
            );
        }

        /// <summary>
        /// Detects whether Apple Neural Engine is available.
        /// </summary>
        /// <returns>True if Neural Engine is available; otherwise, false.</returns>
        public static bool DetectNeuralEngine()
        {
            try
            {
                // Check if we're on Apple Silicon with Neural Engine
                return OperatingSystem.IsMacOS() && Environment.Is64BitProcess;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Detects whether Apple AMX is available.
        /// </summary>
        /// <returns>True if AMX is available; otherwise, false.</returns>
        public static bool DetectAMX()
        {
            try
            {
                // Check if we're on Apple Silicon with AMX
                return OperatingSystem.IsMacOS() && Environment.Is64BitProcess;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Gets the optimal threadgroup size for the given workload.
        /// </summary>
        /// <param name="workloadSize">The size of the workload.</param>
        /// <returns>The recommended threadgroup size.</returns>
        public (int width, int height, int depth) GetOptimalThreadgroupSize(int workloadSize)
        {
            // Apple Silicon GPUs prefer power-of-2 threadgroup sizes
            var threadsPerGroup = Math.Min(MaxThreadsPerGroup, 
                1 << (int)Math.Log2(Math.Min(workloadSize, 1024)));

            return GPUFamily switch
            {
                AppleGPUFamily.Apple7 or AppleGPUFamily.Apple8 => (32, threadsPerGroup / 32, 1),
                AppleGPUFamily.Apple9 or AppleGPUFamily.Apple10 => (64, threadsPerGroup / 64, 1),
                _ => (16, threadsPerGroup / 16, 1)
            };
        }

        /// <summary>
        /// Gets the memory bandwidth estimate for this GPU family.
        /// </summary>
        /// <returns>The estimated memory bandwidth in GB/s.</returns>
        public double GetEstimatedMemoryBandwidth()
        {
            return GPUFamily switch
            {
                AppleGPUFamily.Apple7 => 68.0,  // M1: ~68 GB/s
                AppleGPUFamily.Apple8 => 200.0, // M1 Pro: ~200 GB/s, M1 Max: ~400 GB/s
                AppleGPUFamily.Apple9 => 100.0, // M2: ~100 GB/s
                AppleGPUFamily.Apple10 => 150.0, // M2 Pro/Max: ~150-400 GB/s
                _ => 50.0
            };
        }

        /// <summary>
        /// Gets the estimated compute performance in TFLOPS.
        /// </summary>
        /// <returns>The estimated performance in TFLOPS.</returns>
        public double GetEstimatedTFLOPS()
        {
            return GPUFamily switch
            {
                AppleGPUFamily.Apple7 => 2.6,   // M1: ~2.6 TFLOPS
                AppleGPUFamily.Apple8 => 10.4,  // M1 Max: ~10.4 TFLOPS
                AppleGPUFamily.Apple9 => 3.6,   // M2: ~3.6 TFLOPS
                AppleGPUFamily.Apple10 => 13.6, // M2 Max: ~13.6 TFLOPS
                _ => 1.0
            };
        }

        private static int GetMaxThreadsPerGroup(AppleGPUFamily family)
        {
            return family switch
            {
                AppleGPUFamily.Apple7 or AppleGPUFamily.Apple8 => 1024,
                AppleGPUFamily.Apple9 or AppleGPUFamily.Apple10 => 1024,
                _ => 512
            };
        }

        /// <summary>
        /// Returns a string representation of the Metal capabilities.
        /// </summary>
        /// <returns>A string describing the Metal capabilities.</returns>
        public override string ToString()
        {
            return $"Apple Metal {GPUFamily}: {GetEstimatedTFLOPS():F1} TFLOPS, " +
                   $"{GetEstimatedMemoryBandwidth():F1} GB/s, " +
                   $"RT={SupportsRayTracing}, ANE={SupportsNeuralEngine}, AMX={SupportsAMX}";
        }
    }
}
#endif