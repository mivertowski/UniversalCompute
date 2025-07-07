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
using System;
using System.Threading;
using System.Threading.Tasks;

namespace ILGPU.Backends.Vulkan
{
    /// <summary>
    /// Vulkan Compute accelerator for cross-platform GPU computing.
    /// </summary>
    /// <remarks>
    /// This accelerator provides access to Vulkan's compute shaders, enabling
    /// high-performance GPU compute across multiple platforms and vendors.
    /// 
    /// Supported platforms:
    /// - Windows 10+ with Vulkan 1.1+
    /// - Linux with Vulkan 1.1+
    /// - macOS with MoltenVK (Vulkan over Metal)
    /// - Android with Vulkan 1.1+
    /// 
    /// Supported vendors:
    /// - NVIDIA (GeForce, Quadro, Tesla)
    /// - AMD (Radeon, Instinct)
    /// - Intel (Arc, Iris)
    /// - ARM Mali
    /// - Qualcomm Adreno
    /// </remarks>
    public sealed class VulkanAccelerator : Accelerator
    {
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the VulkanAccelerator class.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="device">The ILGPU device.</param>
        /// <param name="vulkanInstance">The Vulkan instance.</param>
        /// <param name="vulkanDevice">The Vulkan device.</param>
        public VulkanAccelerator(Context context, Device device, VulkanInstance vulkanInstance, VulkanDevice vulkanDevice)
            : base(context, device)
        {
            VulkanInstance = vulkanInstance ?? throw new ArgumentNullException(nameof(vulkanInstance));
            VulkanDevice = vulkanDevice ?? throw new ArgumentNullException(nameof(vulkanDevice));

            // Query device capabilities
            Capabilities = VulkanCapabilities.Query(VulkanDevice.Handle);

            // Get compute queue
            ComputeQueue = VulkanDevice.GetQueue(VulkanQueueType.Compute, 0);

            // Initialize accelerator properties
            InitializeAcceleratorProperties();
        }

        /// <summary>
        /// Gets the Vulkan device capabilities.
        /// </summary>
        public new VulkanCapabilities Capabilities { get; }

        /// <summary>
        /// Gets the Vulkan device.
        /// </summary>
        public VulkanDevice VulkanDevice { get; }

        /// <summary>
        /// Gets the Vulkan instance.
        /// </summary>
        public VulkanInstance VulkanInstance { get; }

        /// <summary>
        /// Gets the compute queue.
        /// </summary>
        public VulkanQueue ComputeQueue { get; }

        #region Vulkan Compute Operations

        /// <summary>
        /// Executes a compute shader dispatch.
        /// </summary>
        /// <param name="pipeline">Compute pipeline to execute.</param>
        /// <param name="descriptorSet">Descriptor set containing buffers and resources.</param>
        /// <param name="groupCountX">Number of workgroups in X dimension.</param>
        /// <param name="groupCountY">Number of workgroups in Y dimension.</param>
        /// <param name="groupCountZ">Number of workgroups in Z dimension.</param>
        public void DispatchCompute(
            VulkanComputePipeline pipeline,
            VulkanDescriptorSet descriptorSet,
            uint groupCountX,
            uint groupCountY = 1,
            uint groupCountZ = 1)
        {
            ThrowIfDisposed();

            using var commandBuffer = VulkanDevice.CreateCommandBuffer();
            commandBuffer.Begin();
            commandBuffer.BindComputePipeline(pipeline);
            commandBuffer.BindDescriptorSets(descriptorSet);
            commandBuffer.Dispatch(groupCountX, groupCountY, groupCountZ);
            commandBuffer.End();

            ComputeQueue.Submit(commandBuffer);
            ComputeQueue.WaitIdle();
        }

        /// <summary>
        /// Executes a compute shader dispatch asynchronously.
        /// </summary>
        public async Task DispatchComputeAsync(
            VulkanComputePipeline pipeline,
            VulkanDescriptorSet descriptorSet,
            uint groupCountX,
            uint groupCountY = 1,
            uint groupCountZ = 1,
            CancellationToken cancellationToken = default) => await Task.Run(() => DispatchCompute(pipeline, descriptorSet, groupCountX, groupCountY, groupCountZ), cancellationToken).ConfigureAwait(false);

        /// <summary>
        /// Creates a compute pipeline from SPIR-V bytecode.
        /// </summary>
        /// <param name="spirvBytecode">SPIR-V shader bytecode.</param>
        /// <param name="entryPoint">Shader entry point function name.</param>
        /// <param name="pushConstantSize">Size of push constants in bytes.</param>
        /// <returns>Vulkan compute pipeline.</returns>
        public VulkanComputePipeline CreateComputePipeline(
            byte[] spirvBytecode,
            string entryPoint = "main",
            uint pushConstantSize = 0)
        {
            ThrowIfDisposed();

            var shaderModule = VulkanDevice.CreateShaderModule(spirvBytecode);
            var pipelineLayout = VulkanDevice.CreateComputePipelineLayout(pushConstantSize);
            
            return VulkanDevice.CreateComputePipeline(shaderModule, pipelineLayout, entryPoint);
        }

        /// <summary>
        /// Creates a buffer for use in compute shaders.
        /// </summary>
        /// <param name="size">Buffer size in bytes.</param>
        /// <param name="usage">Buffer usage flags.</param>
        /// <param name="memoryType">Memory type for the buffer.</param>
        /// <returns>Vulkan buffer.</returns>
        public VulkanBuffer CreateBuffer(
            ulong size,
            VulkanBufferUsage usage,
            VulkanMemoryType memoryType = VulkanMemoryType.DeviceLocal)
        {
            ThrowIfDisposed();
            return VulkanDevice.CreateBuffer(size, usage, memoryType);
        }

        /// <summary>
        /// Creates a descriptor set for binding resources to shaders.
        /// </summary>
        /// <param name="layout">Descriptor set layout.</param>
        /// <returns>Vulkan descriptor set.</returns>
        public VulkanDescriptorSet CreateDescriptorSet(VulkanDescriptorSetLayout layout)
        {
            ThrowIfDisposed();
            return VulkanDevice.CreateDescriptorSet(layout);
        }

        #endregion

        #region Accelerator Implementation

        protected override AcceleratorStream CreateStreamInternal() => new VulkanStream(this);

        protected override void SynchronizeInternal() => ComputeQueue.WaitIdle();

        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize) => new VulkanBuffer(this, length, elementSize);

        protected override Kernel LoadKernelInternal(CompiledKernel kernel) => new VulkanKernel(this, kernel);

        protected override Kernel LoadAutoGroupedKernelInternal(
            CompiledKernel kernel,
            out KernelInfo? kernelInfo)
        {
            kernelInfo = new KernelInfo(
                (int)Capabilities.MaxWorkgroupSize,
                Capabilities.MaxSharedMemorySize);
            return LoadKernelInternal(kernel);
        }

        protected override Kernel LoadImplicitlyGroupedKernelInternal(
            CompiledKernel kernel,
            int customGroupSize,
            out KernelInfo? kernelInfo)
        {
            kernelInfo = new KernelInfo(
                Math.Min(customGroupSize, (int)Capabilities.MaxWorkgroupSize),
                Capabilities.MaxSharedMemorySize);
            return LoadKernelInternal(kernel);
        }

        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(
            Kernel kernel,
            int groupSize,
            int dynamicSharedMemorySizeInBytes) =>
            // Vulkan uses subgroups - estimate based on device properties
            (int)(Capabilities.MaxWorkgroupSize / Math.Max(groupSize, 1));

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            Func<int, int> computeSharedMemorySize,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            return Math.Min(maxGroupSize, (int)Capabilities.MaxWorkgroupSize);
        }

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            int dynamicSharedMemorySizeInBytes,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            
            // Consider shared memory limitations
            var maxGroupsForSharedMemory = dynamicSharedMemorySizeInBytes > 0
                ? Capabilities.MaxSharedMemorySize / dynamicSharedMemorySizeInBytes
                : (int)Capabilities.MaxWorkgroupSize;

            return Math.Min(Math.Min(maxGroupSize, maxGroupsForSharedMemory), 
                           (int)Capabilities.MaxWorkgroupSize);
        }

        protected override bool CanAccessPeerInternal(Accelerator otherAccelerator) =>
            // Vulkan supports device-to-device memory transfer on some implementations
            otherAccelerator is VulkanAccelerator otherVulkan &&
                   Capabilities.SupportsDeviceCoherentMemory &&
                   otherVulkan.Capabilities.SupportsDeviceCoherentMemory;

        protected override void EnablePeerAccessInternal(Accelerator otherAccelerator)
        {
            // Vulkan peer access is enabled through memory allocation flags
        }

        protected override void DisablePeerAccessInternal(Accelerator otherAccelerator)
        {
            // Vulkan peer access is managed through memory allocation
        }

        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(IntPtr pinned, long numElements) =>
            // Vulkan doesn't support page locking
            null!;

        public override TExtension CreateExtension<TExtension, TExtensionProvider>(TExtensionProvider provider) => typeof(TExtension) == typeof(VulkanRayTracingExtension)
                ? (TExtension)(object)new VulkanRayTracingExtension(this)
                :
            throw new NotSupportedException($"Extension {typeof(TExtension)} not supported by Vulkan accelerator");

        protected override void OnBind()
        {
            // Vulkan binding is handled by the driver
        }

        protected override void OnUnbind()
        {
            // Vulkan unbinding is handled by the driver
        }

        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    ComputeQueue?.Dispose();
                    VulkanDevice?.Dispose();
                    VulkanInstance?.Dispose();
                }
                _disposed = true;
            }
        }

        #endregion

        #region Private Methods

        private static void InitializeAcceleratorProperties()
        {
            // Properties are now handled through the Device base class
            // No direct assignment needed as they are read-only properties
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(VulkanAccelerator));
        }

        #endregion

        #region Static Methods

        /// <summary>
        /// Checks if Vulkan is available on this system.
        /// </summary>
        /// <returns>True if Vulkan is available; otherwise, false.</returns>
        public static bool IsAvailable() =>
            // TODO: Implement VulkanNative.IsVulkanSupported()
            false; // Vulkan not implemented yet

        /// <summary>
        /// Creates a Vulkan accelerator if available.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="deviceIndex">The Vulkan device index (0 for default).</param>
        /// <returns>Vulkan accelerator or null if not available.</returns>
        public static VulkanAccelerator? CreateIfAvailable(Context context, int deviceIndex = 0)
        {
            if (!IsAvailable()) return null;

#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var vulkanInstance = VulkanInstance.Create();
                var physicalDevices = vulkanInstance.EnumeratePhysicalDevices();
                
                if (deviceIndex >= physicalDevices.Length) return null;

                var physicalDevice = physicalDevices[deviceIndex];
                var vulkanDevice = physicalDevice.CreateLogicalDevice();

                // TODO: Implement proper Vulkan device creation
                // For now, return null as Vulkan integration is not complete
                return null;
            }
            catch
            {
                return null;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Enumerates all available Vulkan devices.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>Array of Vulkan accelerators.</returns>
        public static VulkanAccelerator[] EnumerateDevices(Context context)
        {
            if (!IsAvailable()) return [];

#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var vulkanInstance = VulkanInstance.Create();
                var physicalDevices = vulkanInstance.EnumeratePhysicalDevices();
                var accelerators = new VulkanAccelerator[physicalDevices.Length];

                for (int i = 0; i < physicalDevices.Length; i++)
                {
                    // TODO: Implement proper Vulkan device enumeration
                    // Skip for now as Device cannot be instantiated directly
                }

                return accelerators;
            }
            catch
            {
                return [];
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        #endregion
    }

    /// <summary>
    /// Vulkan ray tracing extension for advanced graphics features.
    /// </summary>
    public sealed class VulkanRayTracingExtension
    {
        private readonly VulkanAccelerator _accelerator;

        internal VulkanRayTracingExtension(VulkanAccelerator accelerator)
        {
            _accelerator = accelerator;
        }

        /// <summary>
        /// Checks if ray tracing is supported.
        /// </summary>
        public bool IsRayTracingSupported => _accelerator.Capabilities.SupportsRayTracing;

        /// <summary>
        /// Creates a ray tracing pipeline.
        /// </summary>
        /// <param name="shaderStages">Ray tracing shader stages (raygen, miss, closesthit, etc.).</param>
        /// <returns>Ray tracing pipeline handle.</returns>
        public IntPtr CreateRayTracingPipeline(VulkanShaderStage[] shaderStages) => !IsRayTracingSupported
                ? throw new NotSupportedException("Ray tracing not supported on this device")
                : VulkanDevice.CreateRayTracingPipeline(shaderStages);

        /// <summary>
        /// Creates an acceleration structure for ray tracing.
        /// </summary>
        /// <param name="geometryData">Geometry data for the acceleration structure.</param>
        /// <returns>Acceleration structure handle.</returns>
        public IntPtr CreateAccelerationStructure(VulkanGeometryData geometryData) => !IsRayTracingSupported
                ? throw new NotSupportedException("Ray tracing not supported on this device")
                : VulkanDevice.CreateAccelerationStructure(geometryData);
    }
}