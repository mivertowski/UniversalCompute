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
using ILGPU.IR.Analyses;
using ILGPU.Runtime.Vulkan.Native;
using System;
using System.Collections.Immutable;

namespace ILGPU.Runtime.Vulkan
{
    /// <summary>
    /// Vulkan compute accelerator for cross-platform GPU acceleration.
    /// </summary>
    public sealed class VulkanAccelerator : Accelerator
    {
        #region Instance

        /// <summary>
        /// The associated Vulkan device.
        /// </summary>
        public new VulkanDevice Device { get; }

        /// <summary>
        /// The native Vulkan instance handle.
        /// </summary>
        internal VkInstance Instance { get; private set; }

        /// <summary>
        /// The native Vulkan logical device handle.
        /// </summary>
        internal VkDevice LogicalDevice { get; private set; }

        /// <summary>
        /// The compute queue handle.
        /// </summary>
        internal VkQueue ComputeQueue { get; private set; }

        /// <summary>
        /// The compute queue family index.
        /// </summary>
        internal uint ComputeQueueFamily { get; private set; }

        /// <summary>
        /// Gets whether this accelerator supports unified memory.
        /// </summary>
        public new bool SupportsUnifiedMemory => Device.SupportsUnifiedMemory;

        /// <summary>
        /// Initializes a new Vulkan accelerator.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="device">The Vulkan device.</param>
        internal VulkanAccelerator(Context context, VulkanDevice device)
            : base(context, device)
        {
            Device = device;

            try
            {
                // Initialize Vulkan runtime if not already done
                if (!VulkanNative.InitializeVulkanCompute())
                    throw new NotSupportedException("Failed to initialize Vulkan compute");

                // Create Vulkan instance
                CreateVulkanInstance();

                // Create logical device
                CreateLogicalDevice();

                // Get compute queue
                VkQueue queue;
                VulkanNative.GetDeviceQueue(LogicalDevice, ComputeQueueFamily, 0, out queue);
                ComputeQueue = queue;

                // Initialize device properties
                Init();
            }
            catch (Exception ex)
            {
                throw new VulkanException("Failed to initialize Vulkan accelerator", ex);
            }
        }

        /// <summary>
        /// Creates the Vulkan instance.
        /// </summary>
        private void CreateVulkanInstance()
        {
            var appInfo = new VkApplicationInfo
            {
                sType = VkStructureType.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pApplicationName = "ILGPU Universal Compute",
                applicationVersion = 1,
                pEngineName = "ILGPU",
                engineVersion = 1,
                apiVersion = VulkanNative.VK_API_VERSION_1_1
            };

            var createInfo = new VkInstanceCreateInfo
            {
                sType = VkStructureType.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo = appInfo,
                enabledLayerCount = 0,
                enabledExtensionCount = 0
            };

            try
            {
                VkInstance instance;
                var result = VulkanNative.CreateInstance(ref createInfo, IntPtr.Zero, out instance);
                Instance = instance;
                VulkanException.ThrowIfFailed(result);
            }
            catch (DllNotFoundException)
            {
                // Vulkan not available - use dummy instance
                Instance = new VkInstance { Handle = new IntPtr(-1) };
            }
            catch (EntryPointNotFoundException)
            {
                // Vulkan functions not found - use dummy instance
                Instance = new VkInstance { Handle = new IntPtr(-1) };
            }
        }

        /// <summary>
        /// Creates the logical device.
        /// </summary>
        private void CreateLogicalDevice()
        {
            if (Instance.Handle == new IntPtr(-1))
            {
                // Use dummy device when Vulkan is not available
                LogicalDevice = new VkDevice { Handle = new IntPtr(-1) };
                ComputeQueueFamily = 0;
                return;
            }

            try
            {
                // Find compute queue family
                ComputeQueueFamily = FindComputeQueueFamily();

                var queueCreateInfo = new VkDeviceQueueCreateInfo
                {
                    sType = VkStructureType.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    queueFamilyIndex = ComputeQueueFamily,
                    queueCount = 1
                };

                var deviceCreateInfo = new VkDeviceCreateInfo
                {
                    sType = VkStructureType.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                    queueCreateInfoCount = 1,
                    enabledLayerCount = 0,
                    enabledExtensionCount = 0
                };

                VkDevice device;
                var result = VulkanNative.CreateDevice(Device.PhysicalDevice, ref deviceCreateInfo, IntPtr.Zero, out device);
                LogicalDevice = device;
                VulkanException.ThrowIfFailed(result);
            }
            catch (Exception)
            {
                // Use dummy device on any error
                LogicalDevice = new VkDevice { Handle = new IntPtr(-1) };
                ComputeQueueFamily = 0;
            }
        }

        /// <summary>
        /// Finds a queue family that supports compute operations.
        /// </summary>
        /// <returns>Compute queue family index.</returns>
        private uint FindComputeQueueFamily()
        {
            try
            {
                uint queueFamilyCount = 0;
                VulkanNative.GetPhysicalDeviceQueueFamilyProperties(Device.PhysicalDevice, ref queueFamilyCount, null!);

                if (queueFamilyCount == 0)
                    return 0;

                var queueFamilies = new VkQueueFamilyProperties[queueFamilyCount];
                VulkanNative.GetPhysicalDeviceQueueFamilyProperties(Device.PhysicalDevice, ref queueFamilyCount, queueFamilies);

                // Find the first queue family that supports compute
                for (uint i = 0; i < queueFamilyCount; i++)
                {
                    if ((queueFamilies[i].queueFlags & VkQueueFlagBits.VK_QUEUE_COMPUTE_BIT) != 0)
                        return i;
                }

                return 0; // Fallback to first queue family
            }
            catch
            {
                return 0; // Fallback
            }
        }

        /// <summary>
        /// Initializes the accelerator properties.
        /// </summary>
        private void Init() =>
            // Set device-specific properties
            DefaultStream = CreateStreamInternal();

        #endregion

        #region Properties

        /// <summary>
        /// Gets the accelerator type.
        /// </summary>
        public new static AcceleratorType AcceleratorType => AcceleratorType.Vulkan;

        /// <summary>
        /// Gets the name of this accelerator.
        /// </summary>
        public new string Name => Device.Name;

        /// <summary>
        /// Gets the memory information of this accelerator.
        /// </summary>
        public MemoryInfo MemoryInfo => new(Device.MemorySize, Device.MemorySize, 0, 0, 0, false, false, false, 0, 0);

        /// <summary>
        /// Gets the maximum grid size supported by this accelerator.
        /// </summary>
        public new Index3D MaxGridSize => Device.MaxGridSize;

        /// <summary>
        /// Gets the maximum group size supported by this accelerator.
        /// </summary>
        public new Index3D MaxGroupSize => Device.MaxGroupSize;

        /// <summary>
        /// Gets the maximum number of threads per group.
        /// </summary>
        public new int MaxNumThreadsPerGroup => Device.MaxNumThreadsPerGroup;

        /// <summary>
        /// Gets the maximum shared memory per group in bytes.
        /// </summary>
        public new static long MaxSharedMemoryPerGroup => VulkanDevice.MaxSharedMemoryPerGroup;

        /// <summary>
        /// Gets the maximum constant memory in bytes.
        /// </summary>
        public new static long MaxConstantMemory => VulkanDevice.MaxConstantMemory;

        /// <summary>
        /// Gets the warp size (subgroup size on Vulkan).
        /// </summary>
        public new int WarpSize => Device.WarpSize;

        /// <summary>
        /// Gets the number of multiprocessors (shader engines).
        /// </summary>
        public new int NumMultiprocessors => Device.NumMultiprocessors;

        #endregion

        #region Memory Management

        /// <summary>
        /// Allocates a chunk of memory.
        /// </summary>
        /// <param name="length">The length in elements.</param>
        /// <param name="elementSize">The element size in bytes.</param>
        /// <returns>The allocated memory buffer.</returns>
        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize) => new VulkanMemoryBuffer(this, length, elementSize);

        /// <summary>
        /// Creates a page-lock scope for the given array.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="pinned">The pinned array.</param>
        /// <param name="numElements">The number of elements.</param>
        /// <returns>The page-lock scope.</returns>
        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(
            IntPtr pinned,
            long numElements) =>
            new VulkanPageLockScope<T>(this, pinned, numElements);

        #endregion

        #region Kernel Management

        /// <summary>
        /// Loads the given kernel.
        /// </summary>
        /// <param name="kernel">The kernel to load.</param>
        /// <returns>The loaded kernel.</returns>
        protected override Kernel LoadKernelInternal(CompiledKernel kernel) =>
            new VulkanKernel(this, kernel as VulkanCompiledKernel ?? throw new ArgumentException("Invalid kernel type"));

        /// <summary>
        /// Loads an auto-grouped kernel.
        /// </summary>
        protected override Kernel LoadAutoGroupedKernelInternal(
            CompiledKernel kernel,
            out KernelInfo? kernelInfo)
        {
            kernelInfo = new KernelInfo(0, 0, new AllocaKindInformation(), []);
            return LoadKernelInternal(kernel);
        }

        /// <summary>
        /// Loads an implicitly grouped kernel.
        /// </summary>
        protected override Kernel LoadImplicitlyGroupedKernelInternal(
            CompiledKernel kernel,
            int customGroupSize,
            out KernelInfo? kernelInfo)
        {
            kernelInfo = new KernelInfo(0, 0, new AllocaKindInformation(), []);
            return LoadKernelInternal(kernel);
        }

        #endregion

        #region Stream Management

        /// <summary>
        /// Creates a new Vulkan stream.
        /// </summary>
        /// <returns>The created stream.</returns>
        protected override AcceleratorStream CreateStreamInternal() =>
            new VulkanStream(this);

        /// <summary>
        /// Synchronizes all pending operations.
        /// </summary>
        protected override void SynchronizeInternal()
        {
            if (DefaultStream is VulkanStream stream)
                stream.Synchronize();
        }

        #endregion

        #region Peer Access

        /// <summary>
        /// Checks whether this accelerator can access the given accelerator.
        /// </summary>
        /// <param name="otherAccelerator">The other accelerator.</param>
        /// <returns>True if peer access is possible.</returns>
        protected override bool CanAccessPeerInternal(Accelerator otherAccelerator) =>
            // Vulkan doesn't have direct peer access like CUDA
            false;

        /// <summary>
        /// Enables peer access to the given accelerator.
        /// </summary>
        /// <param name="otherAccelerator">The other accelerator.</param>
        protected override void EnablePeerAccessInternal(Accelerator otherAccelerator) => throw new NotSupportedException("Vulkan does not support peer access");

        /// <summary>
        /// Disables peer access to the given accelerator.
        /// </summary>
        /// <param name="otherAccelerator">The other accelerator.</param>
        protected override void DisablePeerAccessInternal(Accelerator otherAccelerator) => throw new NotSupportedException("Vulkan does not support peer access");

        #endregion

        #region Kernel Estimation

        /// <summary>
        /// Estimates the maximum number of active groups per multiprocessor.
        /// </summary>
        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(
            Kernel kernel,
            int groupSize,
            int dynamicSharedMemorySizeInBytes) =>
            // Conservative estimation for Vulkan compute
            Math.Max(1, Device.MaxNumThreadsPerGroup / groupSize);

        /// <summary>
        /// Estimates the group size for the given kernel.
        /// </summary>
        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            Func<int, int> computeSharedMemorySize,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = NumMultiprocessors;
            
            // Vulkan subgroup-aligned sizes
            for (int groupSize = WarpSize; groupSize <= maxGroupSize; groupSize += WarpSize)
            {
                int sharedMemSize = computeSharedMemorySize(groupSize);
                if (sharedMemSize <= MaxSharedMemoryPerGroup)
                    return groupSize;
            }

            return WarpSize;
        }

        /// <summary>
        /// Estimates the group size for the given kernel.
        /// </summary>
        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            int dynamicSharedMemorySizeInBytes,
            int maxGroupSize,
            out int minGridSize) => EstimateGroupSizeInternal(
                kernel,
                _ => dynamicSharedMemorySizeInBytes,
                maxGroupSize,
                out minGridSize);

        #endregion

        #region Extensions

        /// <summary>
        /// Creates an accelerator extension.
        /// </summary>
        /// <typeparam name="TExtension">The extension type.</typeparam>
        /// <typeparam name="TExtensionProvider">The extension provider type.</typeparam>
        /// <param name="provider">The provider instance.</param>
        /// <returns>The created extension.</returns>
        public override TExtension CreateExtension<TExtension, TExtensionProvider>(
            TExtensionProvider provider) =>
            throw new NotSupportedException($"Extension {typeof(TExtension)} is not supported by Vulkan accelerator");

        #endregion

        #region Vulkan Compute Operations

        /// <summary>
        /// Executes matrix multiplication using Vulkan compute shaders.
        /// </summary>
        /// <param name="a">Matrix A.</param>
        /// <param name="b">Matrix B.</param>
        /// <param name="c">Result matrix C.</param>
        /// <param name="m">Rows in A and C.</param>
        /// <param name="k">Columns in A, rows in B.</param>
        /// <param name="n">Columns in B and C.</param>
        /// <param name="stream">Vulkan stream.</param>
        public static unsafe void ExecuteMatMul(
            IntPtr a, IntPtr b, IntPtr c,
            int m, int k, int n,
            VulkanStream? stream = null)
        {
            try
            {
                // Try to use Vulkan compute for hardware acceleration
                VulkanNative.ExecuteVulkanMatMul(
                    a.ToPointer(), b.ToPointer(), c.ToPointer(),
                    m, k, n);
            }
            catch (DllNotFoundException)
            {
                throw new NotSupportedException("Vulkan library not found. Install Vulkan runtime for optimal performance.");
            }
            catch (EntryPointNotFoundException)
            {
                throw new NotSupportedException("Vulkan functions not found. Check Vulkan installation.");
            }
        }

        #endregion

        #region Disposal

        /// <summary>
        /// Disposes this Vulkan accelerator.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (disposing)
            {
                try
                {
                    // Cleanup Vulkan resources
                    if (LogicalDevice.Handle != IntPtr.Zero && LogicalDevice.Handle != new IntPtr(-1))
                    {
                        VulkanNative.DestroyDevice(LogicalDevice, IntPtr.Zero);
                        LogicalDevice = new VkDevice { Handle = IntPtr.Zero };
                    }

                    if (Instance.Handle != IntPtr.Zero && Instance.Handle != new IntPtr(-1))
                    {
                        VulkanNative.DestroyInstance(Instance, IntPtr.Zero);
                        Instance = new VkInstance { Handle = IntPtr.Zero };
                    }
                }
                catch
                {
                    // Ignore errors during disposal
                }
            }
        }

        #endregion

        #region GPU Information

        /// <summary>
        /// Prints detailed GPU information.
        /// </summary>
        public void PrintGPUInformation()
        {
            Console.WriteLine($"Vulkan Device Information:");
            Console.WriteLine($"  Name: {Name}");
            Console.WriteLine($"  Device Type: {Device.DeviceType}");
            Console.WriteLine($"  Vendor: {Device.Vendor}");
            Console.WriteLine($"  Driver Version: {Device.DriverVersion}");
            Console.WriteLine($"  API Version: {Device.APIVersion}");
            Console.WriteLine($"  Total Memory: {MemorySize / (1024 * 1024)} MB");
            Console.WriteLine($"  Compute Units: {NumMultiprocessors}");
            Console.WriteLine($"  Max Work Group Size: {MaxNumThreadsPerGroup}");
            Console.WriteLine($"  Subgroup Size: {WarpSize}");
            Console.WriteLine($"  Max Shared Memory/Group: {MaxSharedMemoryPerGroup} bytes");
            Console.WriteLine($"  Max Constant Memory: {MaxConstantMemory} bytes");
            Console.WriteLine($"  Unified Memory: {SupportsUnifiedMemory}");
            Console.WriteLine($"  Max Grid Size: {MaxGridSize}");
            Console.WriteLine($"  Max Group Size: {MaxGroupSize}");
        }

        #endregion

        #region Abstract Method Implementations

        /// <summary>
        /// Called when the accelerator is bound to the current thread.
        /// </summary>
        protected override void OnBind()
        {
            // Vulkan-specific binding logic if needed
        }

        /// <summary>
        /// Called when the accelerator is unbound from the current thread.
        /// </summary>
        protected override void OnUnbind()
        {
            // Vulkan-specific unbinding logic if needed
        }

        #endregion
    }

    /// <summary>
    /// Vulkan-specific exception.
    /// </summary>
    public class VulkanException : AcceleratorException
    {
        public VulkanException(string message) : base(message) { }
        public VulkanException(string message, Exception innerException) : base(message, innerException) { }

        /// <summary>
        /// Gets the accelerator type.
        /// </summary>
        public override AcceleratorType AcceleratorType => AcceleratorType.Vulkan;

        internal static void ThrowIfFailed(VkResult result)
        {
            if (result != VkResult.VK_SUCCESS)
                throw new VulkanException($"Vulkan operation failed with result: {result}");
        }
    }
}