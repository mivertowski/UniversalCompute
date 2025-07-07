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

using ILGPU.Backends.Vulkan.Native;
using ILGPU.Runtime;
using System;
using System.Runtime.InteropServices;

namespace ILGPU.Backends.Vulkan
{
    /// <summary>
    /// Vulkan queue types.
    /// </summary>
    public enum VulkanQueueType
    {
        /// <summary>
        /// Graphics queue.
        /// </summary>
        Graphics = 1,

        /// <summary>
        /// Compute queue.
        /// </summary>
        Compute = 2,

        /// <summary>
        /// Transfer queue.
        /// </summary>
        Transfer = 4
    }

    /// <summary>
    /// Vulkan buffer usage flags.
    /// </summary>
    [Flags]
    public enum VulkanBufferUsage : uint
    {
        /// <summary>
        /// Transfer source buffer.
        /// </summary>
        TransferSrc = 1,

        /// <summary>
        /// Transfer destination buffer.
        /// </summary>
        TransferDst = 2,

        /// <summary>
        /// Uniform buffer.
        /// </summary>
        UniformBuffer = 16,

        /// <summary>
        /// Storage buffer.
        /// </summary>
        StorageBuffer = 32,

        /// <summary>
        /// Index buffer.
        /// </summary>
        IndexBuffer = 64,

        /// <summary>
        /// Vertex buffer.
        /// </summary>
        VertexBuffer = 128
    }

    /// <summary>
    /// Vulkan memory types.
    /// </summary>
    public enum VulkanMemoryType
    {
        /// <summary>
        /// Device local memory (fastest for GPU access).
        /// </summary>
        DeviceLocal,

        /// <summary>
        /// Host visible memory (CPU accessible).
        /// </summary>
        HostVisible,

        /// <summary>
        /// Host coherent memory (no cache management needed).
        /// </summary>
        HostCoherent,

        /// <summary>
        /// Host cached memory (CPU cached).
        /// </summary>
        HostCached
    }

    /// <summary>
    /// Vulkan device capabilities.
    /// </summary>
    public sealed class VulkanCapabilities
    {
        /// <summary>
        /// Gets the device name.
        /// </summary>
        public string DeviceName { get; internal set; } = string.Empty;

        /// <summary>
        /// Gets the vendor ID.
        /// </summary>
        public uint VendorId { get; internal set; }

        /// <summary>
        /// Gets the device ID.
        /// </summary>
        public uint DeviceId { get; internal set; }

        /// <summary>
        /// Gets the device type.
        /// </summary>
        public VulkanDeviceType DeviceType { get; internal set; }

        /// <summary>
        /// Gets the API version.
        /// </summary>
        public uint ApiVersion { get; internal set; }

        /// <summary>
        /// Gets the driver version.
        /// </summary>
        public uint DriverVersion { get; internal set; }

        /// <summary>
        /// Gets the maximum workgroup size.
        /// </summary>
        public uint MaxWorkgroupSize { get; internal set; }

        /// <summary>
        /// Gets the maximum compute workgroup count.
        /// </summary>
#pragma warning disable CA1819 // Properties should not return arrays
        public uint[] MaxComputeWorkGroupCount { get; internal set; } = new uint[3];
#pragma warning restore CA1819 // Properties should not return arrays

        /// <summary>
        /// Gets the maximum compute workgroup size.
        /// </summary>
#pragma warning disable CA1819 // Properties should not return arrays
        public uint[] MaxComputeWorkGroupSize { get; internal set; } = new uint[3];
#pragma warning restore CA1819 // Properties should not return arrays

        /// <summary>
        /// Gets the maximum shared memory size.
        /// </summary>
        public int MaxSharedMemorySize { get; internal set; }

        /// <summary>
        /// Gets the maximum compute shared memory size.
        /// </summary>
        public uint MaxComputeSharedMemorySize { get; internal set; }

        /// <summary>
        /// Gets the maximum uniform buffer range.
        /// </summary>
        public long MaxUniformBufferRange { get; internal set; }

        /// <summary>
        /// Gets the maximum storage buffer range.
        /// </summary>
        public long MaxStorageBufferRange { get; internal set; }

        /// <summary>
        /// Gets the subgroup size.
        /// </summary>
        public uint SubgroupSize { get; internal set; }

        /// <summary>
        /// Gets whether ray tracing is supported.
        /// </summary>
        public bool SupportsRayTracing { get; internal set; }

        /// <summary>
        /// Gets whether device coherent memory is supported.
        /// </summary>
        public bool SupportsDeviceCoherentMemory { get; internal set; }

        /// <summary>
        /// Gets the memory bandwidth estimate in GB/s.
        /// </summary>
        public double MemoryBandwidth { get; internal set; }

        /// <summary>
        /// Queries Vulkan capabilities from the specified device.
        /// </summary>
        /// <param name="physicalDevice">Physical device handle.</param>
        /// <returns>Vulkan capabilities.</returns>
        public static VulkanCapabilities Query(IntPtr physicalDevice)
        {
            if (physicalDevice == IntPtr.Zero)
                return new VulkanCapabilities();

            VulkanNative.vkGetPhysicalDeviceProperties(physicalDevice, out var properties);

            var capabilities = new VulkanCapabilities
            {
                ApiVersion = properties.ApiVersion,
                DriverVersion = properties.DriverVersion,
                VendorId = properties.VendorID,
                DeviceId = properties.DeviceID,
                DeviceType = (VulkanDeviceType)properties.DeviceType,
                MaxWorkgroupSize = properties.Limits.MaxComputeWorkGroupInvocations,
                MaxSharedMemorySize = (int)properties.Limits.MaxComputeSharedMemorySize,
                MaxComputeSharedMemorySize = properties.Limits.MaxComputeSharedMemorySize,
                MaxUniformBufferRange = properties.Limits.MaxUniformBufferRange,
                MaxStorageBufferRange = properties.Limits.MaxStorageBufferRange,
                SubgroupSize = 32, // Default - would query VK_KHR_shader_subgroup_extended_types
                SupportsRayTracing = false, // Would check for VK_KHR_ray_tracing_pipeline
                SupportsDeviceCoherentMemory = false, // Would check device features
                MemoryBandwidth = EstimateMemoryBandwidth(properties.VendorID, properties.DeviceID)
            };

            // Copy workgroup limits
            unsafe
            {
                for (int i = 0; i < 3; i++)
                {
                    capabilities.MaxComputeWorkGroupCount[i] = properties.Limits.MaxComputeWorkGroupCount[i];
                    capabilities.MaxComputeWorkGroupSize[i] = properties.Limits.MaxComputeWorkGroupSize[i];
                }
            }

            // Get device name
            unsafe
            {
                var nameBytes = new byte[256];
                Marshal.Copy((IntPtr)properties.DeviceName, nameBytes, 0, 256);
                capabilities.DeviceName = System.Text.Encoding.UTF8.GetString(nameBytes).TrimEnd('\0');
            }

            return capabilities;
        }

        private static double EstimateMemoryBandwidth(uint vendorId, uint deviceId) =>
            // Rough estimates based on known GPU families
            vendorId switch
            {
                0x10DE => 500.0, // NVIDIA - varies widely (100-1000+ GB/s)
                0x1002 => 400.0, // AMD - varies widely (200-1600+ GB/s)
                0x8086 => 100.0, // Intel - typically lower bandwidth
                _ => 200.0       // Generic estimate
            };
    }

    /// <summary>
    /// Vulkan device types.
    /// </summary>
    public enum VulkanDeviceType : uint
    {
        /// <summary>
        /// Other device type.
        /// </summary>
        Other = 0,

        /// <summary>
        /// Integrated GPU.
        /// </summary>
        IntegratedGpu = 1,

        /// <summary>
        /// Discrete GPU.
        /// </summary>
        DiscreteGpu = 2,

        /// <summary>
        /// Virtual GPU.
        /// </summary>
        VirtualGpu = 3,

        /// <summary>
        /// CPU device.
        /// </summary>
        Cpu = 4
    }

    /// <summary>
    /// Vulkan instance wrapper.
    /// </summary>
    public sealed class VulkanInstance : IDisposable
    {
        private bool _disposed;

        internal VulkanInstance(IntPtr handle)
        {
            Handle = handle;
        }

        /// <summary>
        /// Gets the native Vulkan instance handle.
        /// </summary>
        public IntPtr Handle { get; }

        /// <summary>
        /// Creates a new Vulkan instance.
        /// </summary>
        /// <returns>Vulkan instance.</returns>
        public static VulkanInstance Create()
        {
            // Create minimal instance for compute
            var result = VulkanNative.vkCreateInstance(IntPtr.Zero, IntPtr.Zero, out var instance);
            return result != VkResult.Success || instance == IntPtr.Zero
                ? throw new InvalidOperationException($"Failed to create Vulkan instance: {result}")
                : new VulkanInstance(instance);
        }

        /// <summary>
        /// Enumerates physical devices.
        /// </summary>
        /// <returns>Array of physical devices.</returns>
        public VulkanPhysicalDevice[] EnumeratePhysicalDevices()
        {
            uint deviceCount = 0;
            var result = VulkanNative.vkEnumeratePhysicalDevices(Handle, ref deviceCount, IntPtr.Zero);
            if (result != VkResult.Success || deviceCount == 0)
                return Array.Empty<VulkanPhysicalDevice>();

            var deviceHandles = new IntPtr[deviceCount];
            var devicesPtr = Marshal.AllocHGlobal((int)(deviceCount * IntPtr.Size));
            try
            {
                result = VulkanNative.vkEnumeratePhysicalDevices(Handle, ref deviceCount, devicesPtr);
                if (result != VkResult.Success)
                    return Array.Empty<VulkanPhysicalDevice>();

                Marshal.Copy(devicesPtr, deviceHandles, 0, (int)deviceCount);

                var devices = new VulkanPhysicalDevice[deviceCount];
                for (int i = 0; i < deviceCount; i++)
                {
                    devices[i] = new VulkanPhysicalDevice(deviceHandles[i]);
                }

                return devices;
            }
            finally
            {
                Marshal.FreeHGlobal(devicesPtr);
            }
        }

        /// <summary>
        /// Disposes the Vulkan instance.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (Handle != IntPtr.Zero)
                {
                    VulkanNative.vkDestroyInstance(Handle, IntPtr.Zero);
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Vulkan physical device wrapper.
    /// </summary>
    public sealed class VulkanPhysicalDevice
    {
        internal VulkanPhysicalDevice(IntPtr handle)
        {
            Handle = handle;
        }

        /// <summary>
        /// Gets the native physical device handle.
        /// </summary>
        public IntPtr Handle { get; }

        /// <summary>
        /// Gets the device name.
        /// </summary>
        public string Name
        {
            get
            {
                VulkanNative.vkGetPhysicalDeviceProperties(Handle, out var properties);
                unsafe
                {
                    var nameBytes = new byte[256];
                    Marshal.Copy((IntPtr)properties.DeviceName, nameBytes, 0, 256);
                    return System.Text.Encoding.UTF8.GetString(nameBytes).TrimEnd('\0');
                }
            }
        }

        /// <summary>
        /// Creates a logical device.
        /// </summary>
        /// <returns>Vulkan logical device.</returns>
        public VulkanDevice CreateLogicalDevice()
        {
            // Create minimal device for compute
            var result = VulkanNative.vkCreateDevice(Handle, IntPtr.Zero, IntPtr.Zero, out var device);
            return result != VkResult.Success || device == IntPtr.Zero
                ? throw new InvalidOperationException($"Failed to create Vulkan logical device: {result}")
                : new VulkanDevice(device, Name);
        }
    }

    /// <summary>
    /// Vulkan logical device wrapper.
    /// </summary>
    public sealed class VulkanDevice : IDisposable
    {
        private bool _disposed;

        internal VulkanDevice(IntPtr handle, string name)
        {
            Handle = handle;
            Name = name;
        }

        /// <summary>
        /// Gets the native device handle.
        /// </summary>
        public IntPtr Handle { get; }

        /// <summary>
        /// Gets the device name.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Gets a queue from the device.
        /// </summary>
        /// <param name="queueType">Queue type.</param>
        /// <param name="queueIndex">Queue index.</param>
        /// <returns>Vulkan queue.</returns>
        public VulkanQueue GetQueue(VulkanQueueType queueType, uint queueIndex)
        {
            VulkanNative.vkGetDeviceQueue(Handle, 0, queueIndex, out var queue); // Simplified - would find proper queue family
            return new VulkanQueue(queue);
        }

        /// <summary>
        /// Creates a command buffer.
        /// </summary>
        /// <returns>Vulkan command buffer.</returns>
        public static VulkanCommandBuffer CreateCommandBuffer() =>
            // Simplified implementation - would create command pool first
            new(IntPtr.Zero);

        /// <summary>
        /// Creates a shader module from SPIR-V bytecode.
        /// </summary>
        /// <param name="spirvBytecode">SPIR-V bytecode.</param>
        /// <returns>Shader module handle.</returns>
        public static IntPtr CreateShaderModule(byte[] spirvBytecode) =>
            // Simplified implementation
            IntPtr.Zero;

        /// <summary>
        /// Creates a compute pipeline layout.
        /// </summary>
        /// <param name="pushConstantSize">Push constant size.</param>
        /// <returns>Pipeline layout handle.</returns>
        public static IntPtr CreateComputePipelineLayout(uint pushConstantSize) =>
            // Simplified implementation
            IntPtr.Zero;

        /// <summary>
        /// Creates a compute pipeline.
        /// </summary>
        /// <param name="shaderModule">Shader module handle.</param>
        /// <param name="pipelineLayout">Pipeline layout handle.</param>
        /// <param name="entryPoint">Entry point name.</param>
        /// <returns>Vulkan compute pipeline.</returns>
        public static VulkanComputePipeline CreateComputePipeline(IntPtr shaderModule, IntPtr pipelineLayout, string entryPoint) =>
            // Simplified implementation
            new(IntPtr.Zero);

        /// <summary>
        /// Creates a buffer.
        /// </summary>
        /// <param name="size">Buffer size.</param>
        /// <param name="usage">Buffer usage.</param>
        /// <param name="memoryType">Memory type.</param>
        /// <returns>Vulkan buffer.</returns>
        public static VulkanBuffer CreateBuffer(ulong size, VulkanBufferUsage usage, VulkanMemoryType memoryType) =>
            // Simplified implementation
            new(null!, (long)size, 4); // Placeholder

        /// <summary>
        /// Creates a descriptor set.
        /// </summary>
        /// <param name="layout">Descriptor set layout.</param>
        /// <returns>Vulkan descriptor set.</returns>
        public static VulkanDescriptorSet CreateDescriptorSet(VulkanDescriptorSetLayout layout) =>
            // Simplified implementation
            new(IntPtr.Zero);

        /// <summary>
        /// Creates a ray tracing pipeline.
        /// </summary>
        /// <param name="shaderStages">Shader stages.</param>
        /// <returns>Ray tracing pipeline handle.</returns>
        public static IntPtr CreateRayTracingPipeline(VulkanShaderStage[] shaderStages) =>
            // Simplified implementation for ray tracing
            IntPtr.Zero;

        /// <summary>
        /// Creates an acceleration structure.
        /// </summary>
        /// <param name="geometryData">Geometry data.</param>
        /// <returns>Acceleration structure handle.</returns>
        public static IntPtr CreateAccelerationStructure(VulkanGeometryData geometryData) =>
            // Simplified implementation for acceleration structures
            IntPtr.Zero;

        /// <summary>
        /// Disposes the Vulkan device.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (Handle != IntPtr.Zero)
                {
                    VulkanNative.vkDestroyDevice(Handle, IntPtr.Zero);
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Vulkan queue wrapper.
    /// </summary>
#pragma warning disable CA1711 // Identifiers should not have incorrect suffix
    public sealed class VulkanQueue : IDisposable
#pragma warning restore CA1711 // Identifiers should not have incorrect suffix
    {
        internal VulkanQueue(IntPtr handle)
        {
            Handle = handle;
        }

        /// <summary>
        /// Gets the native queue handle.
        /// </summary>
        public IntPtr Handle { get; }

        /// <summary>
        /// Submits a command buffer to the queue.
        /// </summary>
        /// <param name="commandBuffer">Command buffer to submit.</param>
        public void Submit(VulkanCommandBuffer commandBuffer)
        {
            var result = VulkanNative.vkQueueSubmit(Handle, 1, IntPtr.Zero, IntPtr.Zero); // Simplified
            if (result != VkResult.Success)
                throw new InvalidOperationException($"Failed to submit command buffer: {result}");
        }

        /// <summary>
        /// Waits for the queue to become idle.
        /// </summary>
        public void WaitIdle()
        {
            var result = VulkanNative.vkQueueWaitIdle(Handle);
            if (result != VkResult.Success)
                throw new InvalidOperationException($"Failed to wait for queue idle: {result}");
        }

        /// <summary>
        /// Disposes the queue.
        /// </summary>
        public void Dispose()
        {
            // Queues are implicitly destroyed with the device
        }
    }

    /// <summary>
    /// Vulkan command buffer wrapper.
    /// </summary>
    public sealed class VulkanCommandBuffer : IDisposable
    {
        internal VulkanCommandBuffer(IntPtr handle)
        {
            NativeBuffer = handle;
        }

        /// <summary>
        /// Gets the native command buffer handle.
        /// </summary>
        public IntPtr NativeBuffer { get; }

        /// <summary>
        /// Begins recording commands.
        /// </summary>
        public void Begin()
        {
            var result = VulkanNative.vkBeginCommandBuffer(NativeBuffer, IntPtr.Zero);
            if (result != VkResult.Success)
                throw new InvalidOperationException($"Failed to begin command buffer: {result}");
        }

        /// <summary>
        /// Ends recording commands.
        /// </summary>
        public void End()
        {
            var result = VulkanNative.vkEndCommandBuffer(NativeBuffer);
            if (result != VkResult.Success)
                throw new InvalidOperationException($"Failed to end command buffer: {result}");
        }

        /// <summary>
        /// Binds a compute pipeline.
        /// </summary>
        /// <param name="pipeline">Compute pipeline.</param>
        public void BindComputePipeline(VulkanComputePipeline pipeline) => VulkanNative.vkCmdBindPipeline(NativeBuffer, 1, pipeline.Handle); // VK_PIPELINE_BIND_POINT_COMPUTE = 1

        /// <summary>
        /// Binds descriptor sets.
        /// </summary>
        /// <param name="descriptorSet">Descriptor set.</param>
        public void BindDescriptorSets(VulkanDescriptorSet descriptorSet) => VulkanNative.vkCmdBindDescriptorSets(NativeBuffer, 1, IntPtr.Zero, 0, 1, IntPtr.Zero, 0, IntPtr.Zero); // Simplified

        /// <summary>
        /// Dispatches compute work.
        /// </summary>
        /// <param name="groupCountX">Workgroups in X.</param>
        /// <param name="groupCountY">Workgroups in Y.</param>
        /// <param name="groupCountZ">Workgroups in Z.</param>
        public void Dispatch(uint groupCountX, uint groupCountY, uint groupCountZ) => VulkanNative.vkCmdDispatch(NativeBuffer, groupCountX, groupCountY, groupCountZ);

        /// <summary>
        /// Disposes the command buffer.
        /// </summary>
        public void Dispose()
        {
            // Command buffers are freed with their command pool
        }
    }

    /// <summary>
    /// Vulkan compute pipeline wrapper.
    /// </summary>
    public sealed class VulkanComputePipeline
    {
        internal VulkanComputePipeline(IntPtr handle)
        {
            Handle = handle;
        }

        /// <summary>
        /// Gets the native pipeline handle.
        /// </summary>
        public IntPtr Handle { get; }
    }

    /// <summary>
    /// Vulkan descriptor set wrapper.
    /// </summary>
    public sealed class VulkanDescriptorSet
    {
        internal VulkanDescriptorSet(IntPtr handle)
        {
            Handle = handle;
        }

        /// <summary>
        /// Gets the native descriptor set handle.
        /// </summary>
        public IntPtr Handle { get; }
    }

    /// <summary>
    /// Vulkan descriptor set layout.
    /// </summary>
    public sealed class VulkanDescriptorSetLayout
    {
        internal VulkanDescriptorSetLayout(IntPtr handle)
        {
            Handle = handle;
        }

        /// <summary>
        /// Gets the native descriptor set layout handle.
        /// </summary>
        public IntPtr Handle { get; }
    }

    /// <summary>
    /// Vulkan buffer wrapper.
    /// </summary>
    public sealed class VulkanBuffer : MemoryBuffer
    {
        private readonly IntPtr _buffer;
        private readonly IntPtr _memory;
        private bool _disposed;

        internal VulkanBuffer(Accelerator accelerator, long length, int elementSize)
            : base(accelerator, length, elementSize)
        {
            // Simplified implementation
            _buffer = IntPtr.Zero;
            _memory = IntPtr.Zero;
        }

        /// <summary>
        /// Gets the native buffer pointer.
        /// </summary>
        public unsafe void* GetNativePtr() => (void*)_buffer;

        /// <summary>
        /// Copies data from CPU memory to this buffer.
        /// </summary>
        public static unsafe void CopyFromCPU(IntPtr source, long sourceOffset, long targetOffset, long length)
        {
            // Simplified implementation - would map memory and copy
        }

        /// <summary>
        /// Copies data from this buffer to CPU memory.
        /// </summary>
        public static unsafe void CopyToCPU(IntPtr target, long sourceOffset, long targetOffset, long length)
        {
            // Simplified implementation - would map memory and copy
        }

        /// <summary>
        /// Copies data from another buffer to this buffer.
        /// </summary>
        public static void CopyFrom(MemoryBuffer source, long sourceOffset, long targetOffset, long length)
        {
            // Simplified implementation - would use vkCmdCopyBuffer
        }

        /// <summary>
        /// Copies data from this buffer to another buffer.
        /// </summary>
        public static void CopyTo(MemoryBuffer target, long sourceOffset, long targetOffset, long length)
        {
            // Simplified implementation - would use vkCmdCopyBuffer
        }

        protected internal override void MemSet(AcceleratorStream stream, byte value, in ArrayView<byte> targetView)
        {
            // Simplified implementation - would use vkCmdFillBuffer
        }

        protected internal override void CopyFrom(AcceleratorStream stream, in ArrayView<byte> sourceView, in ArrayView<byte> targetView)
        {
            // Simplified implementation - would use vkCmdCopyBuffer from source
        }

        protected internal override void CopyTo(AcceleratorStream stream, in ArrayView<byte> sourceView, in ArrayView<byte> targetView)
        {
            // Simplified implementation - would use vkCmdCopyBuffer to target
        }

        /// <summary>
        /// Disposes the Vulkan buffer.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Would destroy buffer and free memory
                }
                _disposed = true;
            }
        }
    }

    // Placeholder types for ray tracing support
    public struct VulkanShaderStage { }
    public struct VulkanGeometryData { }
}