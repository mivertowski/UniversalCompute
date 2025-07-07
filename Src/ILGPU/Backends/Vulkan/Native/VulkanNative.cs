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

using System;
using System.Runtime.InteropServices;

namespace ILGPU.Backends.Vulkan.Native
{
    /// <summary>
    /// Vulkan API result codes.
    /// </summary>
    public enum VkResult : int
    {
        Success = 0,
        NotReady = 1,
        Timeout = 2,
        EventSet = 3,
        EventReset = 4,
        Incomplete = 5,
        ErrorOutOfHostMemory = -1,
        ErrorOutOfDeviceMemory = -2,
        ErrorInitializationFailed = -3,
        ErrorDeviceLost = -4,
        ErrorMemoryMapFailed = -5,
        ErrorLayerNotPresent = -6,
        ErrorExtensionNotPresent = -7,
        ErrorFeatureNotPresent = -8,
        ErrorIncompatibleDriver = -9,
        ErrorTooManyObjects = -10,
        ErrorFormatNotSupported = -11,
        ErrorFragmentedPool = -12,
        ErrorUnknown = -13
    }

    /// <summary>
    /// Vulkan queue family properties.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct VkQueueFamilyProperties
    {
        public uint QueueFlags;
        public uint QueueCount;
        public uint TimestampValidBits;
        public VkExtent3D MinImageTransferGranularity;
    }

    /// <summary>
    /// Vulkan 3D extent.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct VkExtent3D
    {
        public uint Width;
        public uint Height;
        public uint Depth;
    }

    /// <summary>
    /// Vulkan physical device properties.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct VkPhysicalDeviceProperties
    {
        public uint ApiVersion;
        public uint DriverVersion;
        public uint VendorID;
        public uint DeviceID;
        public uint DeviceType;
        public fixed byte DeviceName[256];
        public fixed byte PipelineCacheUUID[16];
        public VkPhysicalDeviceLimits Limits;
        public VkPhysicalDeviceSparseProperties SparseProperties;
    }

    /// <summary>
    /// Vulkan physical device limits.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct VkPhysicalDeviceLimits
    {
        public uint MaxImageDimension1D;
        public uint MaxImageDimension2D;
        public uint MaxImageDimension3D;
        public uint MaxImageDimensionCube;
        public uint MaxImageArrayLayers;
        public uint MaxTexelBufferElements;
        public uint MaxUniformBufferRange;
        public uint MaxStorageBufferRange;
        public uint MaxPushConstantsSize;
        public uint MaxMemoryAllocationCount;
        public uint MaxSamplerAllocationCount;
        public ulong BufferImageGranularity;
        public ulong SparseAddressSpaceSize;
        public uint MaxBoundDescriptorSets;
        public uint MaxPerStageDescriptorSamplers;
        public uint MaxPerStageDescriptorUniformBuffers;
        public uint MaxPerStageDescriptorStorageBuffers;
        public uint MaxPerStageDescriptorSampledImages;
        public uint MaxPerStageDescriptorStorageImages;
        public uint MaxPerStageDescriptorInputAttachments;
        public uint MaxPerStageResources;
        public uint MaxDescriptorSetSamplers;
        public uint MaxDescriptorSetUniformBuffers;
        public uint MaxDescriptorSetUniformBuffersDynamic;
        public uint MaxDescriptorSetStorageBuffers;
        public uint MaxDescriptorSetStorageBuffersDynamic;
        public uint MaxDescriptorSetSampledImages;
        public uint MaxDescriptorSetStorageImages;
        public uint MaxDescriptorSetInputAttachments;
        public uint MaxVertexInputAttributes;
        public uint MaxVertexInputBindings;
        public uint MaxVertexInputAttributeOffset;
        public uint MaxVertexInputBindingStride;
        public uint MaxVertexOutputComponents;
        public uint MaxTessellationGenerationLevel;
        public uint MaxTessellationPatchSize;
        public uint MaxTessellationControlPerVertexInputComponents;
        public uint MaxTessellationControlPerVertexOutputComponents;
        public uint MaxTessellationControlPerPatchOutputComponents;
        public uint MaxTessellationControlTotalOutputComponents;
        public uint MaxTessellationEvaluationInputComponents;
        public uint MaxTessellationEvaluationOutputComponents;
        public uint MaxGeometryShaderInvocations;
        public uint MaxGeometryInputComponents;
        public uint MaxGeometryOutputComponents;
        public uint MaxGeometryOutputVertices;
        public uint MaxGeometryTotalOutputComponents;
        public uint MaxFragmentInputComponents;
        public uint MaxFragmentOutputAttachments;
        public uint MaxFragmentDualSrcAttachments;
        public uint MaxFragmentCombinedOutputResources;
        public uint MaxComputeSharedMemorySize;
        public fixed uint MaxComputeWorkGroupCount[3];
        public uint MaxComputeWorkGroupInvocations;
        public fixed uint MaxComputeWorkGroupSize[3];
        public uint SubPixelPrecisionBits;
        public uint SubTexelPrecisionBits;
        public uint MipmapPrecisionBits;
        public uint MaxDrawIndexedIndexValue;
        public uint MaxDrawIndirectCount;
        public float MaxSamplerLodBias;
        public float MaxSamplerAnisotropy;
        public uint MaxViewports;
        public fixed uint MaxViewportDimensions[2];
        public fixed float ViewportBoundsRange[2];
        public uint ViewportSubPixelBits;
        public nuint MinMemoryMapAlignment;
        public ulong MinTexelBufferOffsetAlignment;
        public ulong MinUniformBufferOffsetAlignment;
        public ulong MinStorageBufferOffsetAlignment;
        public int MinTexelOffset;
        public uint MaxTexelOffset;
        public int MinTexelGatherOffset;
        public uint MaxTexelGatherOffset;
        public float MinInterpolationOffset;
        public float MaxInterpolationOffset;
        public uint SubPixelInterpolationOffsetBits;
        public uint MaxFramebufferWidth;
        public uint MaxFramebufferHeight;
        public uint MaxFramebufferLayers;
        public uint FramebufferColorSampleCounts;
        public uint FramebufferDepthSampleCounts;
        public uint FramebufferStencilSampleCounts;
        public uint FramebufferNoAttachmentsSampleCounts;
        public uint MaxColorAttachments;
        public uint SampledImageColorSampleCounts;
        public uint SampledImageIntegerSampleCounts;
        public uint SampledImageDepthSampleCounts;
        public uint SampledImageStencilSampleCounts;
        public uint StorageImageSampleCounts;
        public uint MaxSampleMaskWords;
        public uint TimestampComputeAndGraphics;
        public float TimestampPeriod;
        public uint MaxClipDistances;
        public uint MaxCullDistances;
        public uint MaxCombinedClipAndCullDistances;
        public uint DiscreteQueuePriorities;
        public fixed float PointSizeRange[2];
        public fixed float LineWidthRange[2];
        public float PointSizeGranularity;
        public float LineWidthGranularity;
        public uint StrictLines;
        public uint StandardSampleLocations;
        public ulong OptimalBufferCopyOffsetAlignment;
        public ulong OptimalBufferCopyRowPitchAlignment;
        public ulong NonCoherentAtomSize;
    }

    /// <summary>
    /// Vulkan physical device sparse properties.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceSparseProperties
    {
        public uint ResidencyStandard2DBlockShape;
        public uint ResidencyStandard2DMultisampleBlockShape;
        public uint ResidencyStandard3DBlockShape;
        public uint ResidencyAlignedMipSize;
        public uint ResidencyNonResidentStrict;
    }

    /// <summary>
    /// Native Vulkan API bindings.
    /// </summary>
    internal static partial class VulkanNative
    {
        #region Constants

#if WINDOWS
        private const string VulkanLibrary = "vulkan-1";
#elif MACOS
        private const string VulkanLibrary = "libvulkan.1.dylib";
#else
        private const string VulkanLibrary = "libvulkan.so.1";
#endif

        #endregion

        #region Instance Management

        /// <summary>
        /// Creates a Vulkan instance.
        /// </summary>
        /// <param name="pCreateInfo">Instance creation info.</param>
        /// <param name="pAllocator">Memory allocator.</param>
        /// <param name="pInstance">Created instance handle.</param>
        /// <returns>Vulkan result code.</returns>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial VkResult vkCreateInstance(
            IntPtr pCreateInfo,
            IntPtr pAllocator,
            out IntPtr pInstance);

        /// <summary>
        /// Destroys a Vulkan instance.
        /// </summary>
        /// <param name="instance">Instance handle.</param>
        /// <param name="pAllocator">Memory allocator.</param>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial void vkDestroyInstance(
            IntPtr instance,
            IntPtr pAllocator);

        /// <summary>
        /// Enumerates physical devices.
        /// </summary>
        /// <param name="instance">Instance handle.</param>
        /// <param name="pPhysicalDeviceCount">Device count.</param>
        /// <param name="pPhysicalDevices">Device handles.</param>
        /// <returns>Vulkan result code.</returns>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial VkResult vkEnumeratePhysicalDevices(
            IntPtr instance,
            ref uint pPhysicalDeviceCount,
            IntPtr pPhysicalDevices);

        #endregion

        #region Physical Device Queries

        /// <summary>
        /// Gets physical device properties.
        /// </summary>
        /// <param name="physicalDevice">Physical device handle.</param>
        /// <param name="pProperties">Device properties.</param>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial void vkGetPhysicalDeviceProperties(
            IntPtr physicalDevice,
            out VkPhysicalDeviceProperties pProperties);

        /// <summary>
        /// Gets physical device queue family properties.
        /// </summary>
        /// <param name="physicalDevice">Physical device handle.</param>
        /// <param name="pQueueFamilyPropertyCount">Queue family count.</param>
        /// <param name="pQueueFamilyProperties">Queue family properties.</param>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial void vkGetPhysicalDeviceQueueFamilyProperties(
            IntPtr physicalDevice,
            ref uint pQueueFamilyPropertyCount,
            IntPtr pQueueFamilyProperties);

        /// <summary>
        /// Gets physical device memory properties.
        /// </summary>
        /// <param name="physicalDevice">Physical device handle.</param>
        /// <param name="pMemoryProperties">Memory properties.</param>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial void vkGetPhysicalDeviceMemoryProperties(
            IntPtr physicalDevice,
            IntPtr pMemoryProperties);

        #endregion

        #region Logical Device Management

        /// <summary>
        /// Creates a logical device.
        /// </summary>
        /// <param name="physicalDevice">Physical device handle.</param>
        /// <param name="pCreateInfo">Device creation info.</param>
        /// <param name="pAllocator">Memory allocator.</param>
        /// <param name="pDevice">Created device handle.</param>
        /// <returns>Vulkan result code.</returns>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial VkResult vkCreateDevice(
            IntPtr physicalDevice,
            IntPtr pCreateInfo,
            IntPtr pAllocator,
            out IntPtr pDevice);

        /// <summary>
        /// Destroys a logical device.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="pAllocator">Memory allocator.</param>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial void vkDestroyDevice(
            IntPtr device,
            IntPtr pAllocator);

        /// <summary>
        /// Gets a device queue.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="queueFamilyIndex">Queue family index.</param>
        /// <param name="queueIndex">Queue index.</param>
        /// <param name="pQueue">Queue handle.</param>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial void vkGetDeviceQueue(
            IntPtr device,
            uint queueFamilyIndex,
            uint queueIndex,
            out IntPtr pQueue);

        #endregion

        #region Buffer and Memory Management

        /// <summary>
        /// Creates a buffer.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="pCreateInfo">Buffer creation info.</param>
        /// <param name="pAllocator">Memory allocator.</param>
        /// <param name="pBuffer">Created buffer handle.</param>
        /// <returns>Vulkan result code.</returns>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial VkResult vkCreateBuffer(
            IntPtr device,
            IntPtr pCreateInfo,
            IntPtr pAllocator,
            out IntPtr pBuffer);

        /// <summary>
        /// Destroys a buffer.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="buffer">Buffer handle.</param>
        /// <param name="pAllocator">Memory allocator.</param>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial void vkDestroyBuffer(
            IntPtr device,
            IntPtr buffer,
            IntPtr pAllocator);

        /// <summary>
        /// Allocates device memory.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="pAllocateInfo">Memory allocation info.</param>
        /// <param name="pAllocator">Memory allocator.</param>
        /// <param name="pMemory">Allocated memory handle.</param>
        /// <returns>Vulkan result code.</returns>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial VkResult vkAllocateMemory(
            IntPtr device,
            IntPtr pAllocateInfo,
            IntPtr pAllocator,
            out IntPtr pMemory);

        /// <summary>
        /// Frees device memory.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="memory">Memory handle.</param>
        /// <param name="pAllocator">Memory allocator.</param>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial void vkFreeMemory(
            IntPtr device,
            IntPtr memory,
            IntPtr pAllocator);

        /// <summary>
        /// Binds buffer memory.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="buffer">Buffer handle.</param>
        /// <param name="memory">Memory handle.</param>
        /// <param name="memoryOffset">Memory offset.</param>
        /// <returns>Vulkan result code.</returns>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial VkResult vkBindBufferMemory(
            IntPtr device,
            IntPtr buffer,
            IntPtr memory,
            ulong memoryOffset);

        #endregion

        #region Command Buffer Management

        /// <summary>
        /// Creates a command pool.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="pCreateInfo">Command pool creation info.</param>
        /// <param name="pAllocator">Memory allocator.</param>
        /// <param name="pCommandPool">Created command pool handle.</param>
        /// <returns>Vulkan result code.</returns>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial VkResult vkCreateCommandPool(
            IntPtr device,
            IntPtr pCreateInfo,
            IntPtr pAllocator,
            out IntPtr pCommandPool);

        /// <summary>
        /// Destroys a command pool.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="commandPool">Command pool handle.</param>
        /// <param name="pAllocator">Memory allocator.</param>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial void vkDestroyCommandPool(
            IntPtr device,
            IntPtr commandPool,
            IntPtr pAllocator);

        /// <summary>
        /// Allocates command buffers.
        /// </summary>
        /// <param name="device">Device handle.</param>
        /// <param name="pAllocateInfo">Command buffer allocation info.</param>
        /// <param name="pCommandBuffers">Allocated command buffer handles.</param>
        /// <returns>Vulkan result code.</returns>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial VkResult vkAllocateCommandBuffers(
            IntPtr device,
            IntPtr pAllocateInfo,
            IntPtr pCommandBuffers);

        /// <summary>
        /// Begins command buffer recording.
        /// </summary>
        /// <param name="commandBuffer">Command buffer handle.</param>
        /// <param name="pBeginInfo">Begin info.</param>
        /// <returns>Vulkan result code.</returns>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial VkResult vkBeginCommandBuffer(
            IntPtr commandBuffer,
            IntPtr pBeginInfo);

        /// <summary>
        /// Ends command buffer recording.
        /// </summary>
        /// <param name="commandBuffer">Command buffer handle.</param>
        /// <returns>Vulkan result code.</returns>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial VkResult vkEndCommandBuffer(IntPtr commandBuffer);

        #endregion

        #region Compute Operations

        /// <summary>
        /// Dispatches compute work.
        /// </summary>
        /// <param name="commandBuffer">Command buffer handle.</param>
        /// <param name="groupCountX">Number of workgroups in X dimension.</param>
        /// <param name="groupCountY">Number of workgroups in Y dimension.</param>
        /// <param name="groupCountZ">Number of workgroups in Z dimension.</param>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial void vkCmdDispatch(
            IntPtr commandBuffer,
            uint groupCountX,
            uint groupCountY,
            uint groupCountZ);

        /// <summary>
        /// Binds compute pipeline.
        /// </summary>
        /// <param name="commandBuffer">Command buffer handle.</param>
        /// <param name="pipeline">Pipeline handle.</param>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial void vkCmdBindPipeline(
            IntPtr commandBuffer,
            uint pipelineBindPoint,
            IntPtr pipeline);

        /// <summary>
        /// Binds descriptor sets.
        /// </summary>
        /// <param name="commandBuffer">Command buffer handle.</param>
        /// <param name="pipelineBindPoint">Pipeline bind point.</param>
        /// <param name="layout">Pipeline layout.</param>
        /// <param name="firstSet">First set index.</param>
        /// <param name="descriptorSetCount">Descriptor set count.</param>
        /// <param name="pDescriptorSets">Descriptor sets.</param>
        /// <param name="dynamicOffsetCount">Dynamic offset count.</param>
        /// <param name="pDynamicOffsets">Dynamic offsets.</param>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial void vkCmdBindDescriptorSets(
            IntPtr commandBuffer,
            uint pipelineBindPoint,
            IntPtr layout,
            uint firstSet,
            uint descriptorSetCount,
            IntPtr pDescriptorSets,
            uint dynamicOffsetCount,
            IntPtr pDynamicOffsets);

        #endregion

        #region Queue Operations

        /// <summary>
        /// Submits command buffers to a queue.
        /// </summary>
        /// <param name="queue">Queue handle.</param>
        /// <param name="submitCount">Number of submissions.</param>
        /// <param name="pSubmits">Submit info.</param>
        /// <param name="fence">Fence handle.</param>
        /// <returns>Vulkan result code.</returns>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial VkResult vkQueueSubmit(
            IntPtr queue,
            uint submitCount,
            IntPtr pSubmits,
            IntPtr fence);

        /// <summary>
        /// Waits for queue to become idle.
        /// </summary>
        /// <param name="queue">Queue handle.</param>
        /// <returns>Vulkan result code.</returns>
        [LibraryImport(VulkanLibrary)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [UnmanagedCallConv(CallConvs = [typeof(System.Runtime.CompilerServices.CallConvCdecl)])]
        internal static partial VkResult vkQueueWaitIdle(IntPtr queue);

        #endregion

        #region Helper Methods

        /// <summary>
        /// Checks if Vulkan is supported on this system.
        /// </summary>
        /// <returns>True if Vulkan is available; otherwise, false.</returns>
        internal static bool IsVulkanSupported()
        {
            try
            {
                // Try to create a minimal Vulkan instance
                var result = vkCreateInstance(IntPtr.Zero, IntPtr.Zero, out var instance);
                if (result == VkResult.Success && instance != IntPtr.Zero)
                {
                    vkDestroyInstance(instance, IntPtr.Zero);
                    return true;
                }
                return false;
            }
            catch (DllNotFoundException)
            {
                return false;
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>
        /// Gets Vulkan API version.
        /// </summary>
        /// <returns>Vulkan API version.</returns>
        internal static uint GetApiVersion() =>
            // Vulkan 1.1 minimum required for compute
            MakeVersion(1, 1, 0);

        /// <summary>
        /// Creates a Vulkan version number.
        /// </summary>
        private static uint MakeVersion(uint major, uint minor, uint patch) => (major << 22) | (minor << 12) | patch;

        #endregion
    }
}
