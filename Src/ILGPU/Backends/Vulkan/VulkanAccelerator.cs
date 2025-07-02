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

using ILGPU.Backends;
using ILGPU.Backends.Vulkan;
using ILGPU.Runtime;
using System;
using System.Runtime.InteropServices;
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
        private readonly VulkanDevice _device;
        private readonly VulkanInstance _instance;
        private readonly VulkanCapabilities _capabilities;
        private readonly VulkanQueue _computeQueue;
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
            _instance = vulkanInstance ?? throw new ArgumentNullException(nameof(vulkanInstance));
            _device = vulkanDevice ?? throw new ArgumentNullException(nameof(vulkanDevice));

            // Query device capabilities
            _capabilities = VulkanCapabilities.Query(_device.Handle);

            // Get compute queue
            _computeQueue = _device.GetQueue(VulkanQueueType.Compute, 0);

            // Initialize accelerator properties
            InitializeAcceleratorProperties();
        }

        /// <summary>
        /// Gets the Vulkan device capabilities.
        /// </summary>
        public VulkanCapabilities Capabilities => _capabilities;

        /// <summary>
        /// Gets the Vulkan device.
        /// </summary>
        public VulkanDevice VulkanDevice => _device;

        /// <summary>
        /// Gets the Vulkan instance.
        /// </summary>
        public VulkanInstance VulkanInstance => _instance;

        /// <summary>
        /// Gets the compute queue.
        /// </summary>
        public VulkanQueue ComputeQueue => _computeQueue;

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

            using var commandBuffer = _device.CreateCommandBuffer();
            commandBuffer.Begin();
            commandBuffer.BindComputePipeline(pipeline);
            commandBuffer.BindDescriptorSets(descriptorSet);
            commandBuffer.Dispatch(groupCountX, groupCountY, groupCountZ);
            commandBuffer.End();

            _computeQueue.Submit(commandBuffer);
            _computeQueue.WaitIdle();
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
            CancellationToken cancellationToken = default)
        {
            await Task.Run(() => DispatchCompute(pipeline, descriptorSet, groupCountX, groupCountY, groupCountZ), cancellationToken);
        }

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

            var shaderModule = _device.CreateShaderModule(spirvBytecode);
            var pipelineLayout = _device.CreateComputePipelineLayout(pushConstantSize);
            
            return _device.CreateComputePipeline(shaderModule, pipelineLayout, entryPoint);
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
            return _device.CreateBuffer(size, usage, memoryType);
        }

        /// <summary>
        /// Creates a descriptor set for binding resources to shaders.
        /// </summary>
        /// <param name="layout">Descriptor set layout.</param>
        /// <returns>Vulkan descriptor set.</returns>
        public VulkanDescriptorSet CreateDescriptorSet(VulkanDescriptorSetLayout layout)
        {
            ThrowIfDisposed();
            return _device.CreateDescriptorSet(layout);
        }

        #endregion

        #region Accelerator Implementation

        protected override AcceleratorStream CreateStreamInternal()
        {
            return new VulkanStream(this);
        }

        protected override void SynchronizeInternal()
        {
            _computeQueue.WaitIdle();
        }

        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize)
        {
            return new VulkanBuffer(this, length, elementSize);
        }

        protected override Kernel LoadKernelInternal(CompiledKernel compiledKernel)
        {
            return new VulkanKernel(this, compiledKernel);
        }

        protected override Kernel LoadAutoGroupedKernelInternal(
            CompiledKernel compiledKernel,
            out KernelInfo? kernelInfo)
        {
            kernelInfo = new KernelInfo(
                _capabilities.MaxWorkgroupSize,
                _capabilities.MaxSharedMemorySize);
            return LoadKernelInternal(compiledKernel);
        }

        protected override Kernel LoadImplicitlyGroupedKernelInternal(
            CompiledKernel compiledKernel,
            int customGroupSize,
            out KernelInfo? kernelInfo)
        {
            kernelInfo = new KernelInfo(
                Math.Min(customGroupSize, (int)_capabilities.MaxWorkgroupSize),
                _capabilities.MaxSharedMemorySize);
            return LoadKernelInternal(compiledKernel);
        }

        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(
            Kernel kernel,
            int groupSize,
            int dynamicSharedMemorySizeInBytes)
        {
            // Vulkan uses subgroups - estimate based on device properties
            return (int)(_capabilities.MaxWorkgroupSize / Math.Max(groupSize, 1));
        }

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            Func<int, int> computeSharedMemorySize,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            return Math.Min(maxGroupSize, (int)_capabilities.MaxWorkgroupSize);
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
                ? _capabilities.MaxSharedMemorySize / dynamicSharedMemorySizeInBytes
                : (int)_capabilities.MaxWorkgroupSize;

            return Math.Min(Math.Min(maxGroupSize, maxGroupsForSharedMemory), 
                           (int)_capabilities.MaxWorkgroupSize);
        }

        protected override bool CanAccessPeerInternal(Accelerator otherAccelerator)
        {
            // Vulkan supports device-to-device memory transfer on some implementations
            return otherAccelerator is VulkanAccelerator otherVulkan &&
                   _capabilities.SupportsDeviceCoherentMemory &&
                   otherVulkan._capabilities.SupportsDeviceCoherentMemory;
        }

        protected override void EnablePeerAccessInternal(Accelerator otherAccelerator)
        {
            // Vulkan peer access is enabled through memory allocation flags
        }

        protected override void DisablePeerAccessInternal(Accelerator otherAccelerator)
        {
            // Vulkan peer access is managed through memory allocation
        }

        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(IntPtr pinned, long numElements)
        {
            return new PageLockScope<T>(this, pinned, numElements);
        }

        public override TExtension CreateExtension<TExtension, TExtensionProvider>(TExtensionProvider provider)
        {
            if (typeof(TExtension) == typeof(VulkanRayTracingExtension))
            {
                return (TExtension)(object)new VulkanRayTracingExtension(this);
            }
            
            throw new NotSupportedException($"Extension {typeof(TExtension)} not supported by Vulkan accelerator");
        }

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
                    _computeQueue?.Dispose();
                    _device?.Dispose();
                    _instance?.Dispose();
                }
                _disposed = true;
            }
        }

        #endregion

        #region Private Methods

        private void InitializeAcceleratorProperties()
        {
            Name = $"Vulkan GPU ({_device.Name})";
            MaxGridSize = new Index3D(
                (int)_capabilities.MaxComputeWorkGroupCount[0],
                (int)_capabilities.MaxComputeWorkGroupCount[1],
                (int)_capabilities.MaxComputeWorkGroupCount[2]);
            MaxGroupSize = new Index3D(
                (int)_capabilities.MaxComputeWorkGroupSize[0],
                (int)_capabilities.MaxComputeWorkGroupSize[1],
                (int)_capabilities.MaxComputeWorkGroupSize[2]);
            WarpSize = (int)_capabilities.SubgroupSize;
            NumMultiprocessors = (int)_capabilities.MaxComputeSharedMemorySize / 1024; // Estimate
            MaxSharedMemoryPerMultiprocessor = _capabilities.MaxSharedMemorySize;
            MaxConstantMemory = _capabilities.MaxUniformBufferRange;
            MaxMemoryBandwidth = _capabilities.MemoryBandwidth;
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
        public static bool IsAvailable()
        {
            try
            {
                return VulkanNative.IsVulkanSupported();
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Creates a Vulkan accelerator if available.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="deviceIndex">The Vulkan device index (0 for default).</param>
        /// <returns>Vulkan accelerator or null if not available.</returns>
        public static VulkanAccelerator? CreateIfAvailable(Context context, int deviceIndex = 0)
        {
            if (!IsAvailable()) return null;

            try
            {
                var vulkanInstance = VulkanInstance.Create();
                var physicalDevices = vulkanInstance.EnumeratePhysicalDevices();
                
                if (deviceIndex >= physicalDevices.Length) return null;

                var physicalDevice = physicalDevices[deviceIndex];
                var vulkanDevice = physicalDevice.CreateLogicalDevice();

                var device = new Device(
                    physicalDevice.Name,
                    deviceIndex,
                    AcceleratorType.OpenCL); // Vulkan is closest to OpenCL in ILGPU's type system

                return new VulkanAccelerator(context, device, vulkanInstance, vulkanDevice);
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Enumerates all available Vulkan devices.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>Array of Vulkan accelerators.</returns>
        public static VulkanAccelerator[] EnumerateDevices(Context context)
        {
            if (!IsAvailable()) return Array.Empty<VulkanAccelerator>();

            try
            {
                var vulkanInstance = VulkanInstance.Create();
                var physicalDevices = vulkanInstance.EnumeratePhysicalDevices();
                var accelerators = new VulkanAccelerator[physicalDevices.Length];

                for (int i = 0; i < physicalDevices.Length; i++)
                {
                    var physicalDevice = physicalDevices[i];
                    var vulkanDevice = physicalDevice.CreateLogicalDevice();
                    var device = new Device(physicalDevice.Name, i, AcceleratorType.OpenCL);
                    accelerators[i] = new VulkanAccelerator(context, device, vulkanInstance, vulkanDevice);
                }

                return accelerators;
            }
            catch
            {
                return Array.Empty<VulkanAccelerator>();
            }
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
        public IntPtr CreateRayTracingPipeline(VulkanShaderStage[] shaderStages)
        {
            if (!IsRayTracingSupported)
                throw new NotSupportedException("Ray tracing not supported on this device");

            return _accelerator.VulkanDevice.CreateRayTracingPipeline(shaderStages);
        }

        /// <summary>
        /// Creates an acceleration structure for ray tracing.
        /// </summary>
        /// <param name="geometryData">Geometry data for the acceleration structure.</param>
        /// <returns>Acceleration structure handle.</returns>
        public IntPtr CreateAccelerationStructure(VulkanGeometryData geometryData)
        {
            if (!IsRayTracingSupported)
                throw new NotSupportedException("Ray tracing not supported on this device");

            return _accelerator.VulkanDevice.CreateAccelerationStructure(geometryData);
        }
    }
}