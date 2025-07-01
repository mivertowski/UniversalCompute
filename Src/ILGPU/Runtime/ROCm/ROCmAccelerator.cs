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

#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member

using ILGPU.Backends;
using ILGPU.Backends.ROCm;
using ILGPU.Resources;
using ILGPU.Runtime.ROCm.Native;
using ILGPU.Util;
using System;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Reflection;
using System.Threading.Tasks;

namespace ILGPU.Runtime.ROCm
{
    /// <summary>
    /// AMD ROCm GPU accelerator for high-performance computing.
    /// </summary>
    public sealed partial class ROCmAccelerator : KernelAccelerator<ROCmCompiledKernel, ROCmKernel>
    {
        #region Instance

        /// <summary>
        /// Stores the associated ROCm device.
        /// </summary>
        public ROCmDevice Device { get; }

        /// <summary>
        /// Gets the backend of this accelerator.
        /// </summary>
        public new ROCmBackend Backend => base.Backend.AsNotNullCast<ROCmBackend>();

        /// <summary>
        /// Gets the native ROCm context handle.
        /// </summary>
        internal IntPtr NativePtr { get; private set; }

        /// <summary>
        /// Gets whether this accelerator supports unified memory.
        /// </summary>
        public bool SupportsUnifiedMemory => Device.SupportsUnifiedAddressing;

        /// <summary>
        /// Gets whether this accelerator supports page-locked memory.
        /// </summary>
        public bool SupportsPageLockedMemory => true;

        /// <summary>
        /// Gets whether this accelerator supports managed memory.
        /// </summary>
        public bool SupportsManagedMemory => Device.SupportsManagedMemory;

        /// <summary>
        /// Initializes a new ROCm accelerator.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="device">The ROCm device.</param>
        internal ROCmAccelerator(Context context, ROCmDevice device)
            : base(context, device)
        {
            Device = device;
            
            try
            {
                // Initialize ROCm runtime if not already done
                if (!ROCmNative.InitializeROCm())
                    throw new NotSupportedException("Failed to initialize ROCm runtime");

                // Set the current device
                var result = ROCmNative.SetDevice(device.DeviceId);
                ROCmException.ThrowIfFailed(result);

                // Store device context (simplified - real implementation would create HIP context)
                NativePtr = new IntPtr(device.DeviceId + 1); // Dummy context handle

                // Initialize device properties
                Init(device);
            }
            catch (Exception ex)
            {
                throw new ROCmException("Failed to initialize ROCm accelerator", ex);
            }
        }

        /// <summary>
        /// Initializes the accelerator properties.
        /// </summary>
        /// <param name="device">The ROCm device.</param>
        private void Init(ROCmDevice device)
        {
            // Set device-specific properties
            DefaultStream = CreateStreamInternal();
        }

        #endregion

        #region Properties


        #endregion

        #region Memory Management

        /// <summary>
        /// Allocates a chunk of memory.
        /// </summary>
        /// <param name="length">The length in elements.</param>
        /// <param name="elementSize">The element size in bytes.</param>
        /// <returns>The allocated memory buffer.</returns>
        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize)
        {
            return new ROCmMemoryBuffer(this, length, elementSize);
        }

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
            new ROCmPageLockScope<T>(this, pinned, numElements);

        #endregion

        #region Kernel Management

        /// <summary>
        /// Creates a kernel from the given compiled kernel.
        /// </summary>
        /// <param name="compiledKernel">The compiled kernel.</param>
        /// <returns>The created kernel.</returns>
        protected override ROCmKernel CreateKernel(ROCmCompiledKernel compiledKernel) =>
            new ROCmKernel(this, compiledKernel, MaxNumThreadsPerGroup);

        /// <summary>
        /// Creates a kernel from the given compiled kernel with launcher.
        /// </summary>
        /// <param name="compiledKernel">The compiled kernel.</param>
        /// <param name="launcher">The kernel launcher method.</param>
        /// <returns>The created kernel.</returns>
        protected override ROCmKernel CreateKernel(ROCmCompiledKernel compiledKernel, MethodInfo launcher) =>
            new ROCmKernel(this, compiledKernel, MaxNumThreadsPerGroup);

        /// <summary>
        /// Generates a kernel launcher method.
        /// </summary>
        /// <param name="kernel">The compiled kernel.</param>
        /// <param name="customGroupSize">The custom group size.</param>
        /// <returns>The launcher method.</returns>
        protected override MethodInfo GenerateKernelLauncherMethod(
            ROCmCompiledKernel kernel,
            int customGroupSize)
        {
            // For ROCm accelerator, use default launcher generation
            // In a real implementation, this would generate optimized ROCm-specific launchers
            return typeof(ROCmAccelerator).GetMethod(nameof(DefaultLauncher), 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)
                ?? throw new InvalidOperationException("Default launcher method not found");
        }

        /// <summary>
        /// Default kernel launcher for ROCm kernels.
        /// </summary>
        [System.Diagnostics.CodeAnalysis.DoesNotReturn]
        private static void DefaultLauncher()
        {
            // Default launcher implementation
            throw new NotImplementedException("ROCm kernel launcher not implemented");
        }

        #endregion

        #region Stream Management

        /// <summary>
        /// Creates a new ROCm stream.
        /// </summary>
        /// <returns>The created stream.</returns>
        protected override AcceleratorStream CreateStreamInternal() =>
            new ROCmStream(this);

        /// <summary>
        /// Synchronizes all pending operations.
        /// </summary>
        protected override void SynchronizeInternal()
        {
            // Synchronize the default stream
            if (DefaultStream is ROCmStream stream)
                stream.Synchronize();
        }

        #endregion

        #region Peer Access

        /// <summary>
        /// Checks whether this accelerator can access the given accelerator.
        /// </summary>
        /// <param name="otherAccelerator">The other accelerator.</param>
        /// <returns>True if peer access is possible.</returns>
        protected override bool CanAccessPeerInternal(Accelerator otherAccelerator)
        {
            if (otherAccelerator is not ROCmAccelerator otherROCm)
                return false;

            // Check if peer access is supported between devices
            // For simplicity, assume peer access is supported between all ROCm devices
            return true;
        }

        /// <summary>
        /// Enables peer access to the given accelerator.
        /// </summary>
        /// <param name="otherAccelerator">The other accelerator.</param>
        protected override void EnablePeerAccessInternal(Accelerator otherAccelerator)
        {
            if (otherAccelerator is not ROCmAccelerator)
                throw new InvalidOperationException("Cannot enable peer access to non-ROCm accelerator");

            // Enable peer access (implementation would use hipDeviceEnablePeerAccess)
        }

        /// <summary>
        /// Disables peer access to the given accelerator.
        /// </summary>
        /// <param name="otherAccelerator">The other accelerator.</param>
        protected override void DisablePeerAccessInternal(Accelerator otherAccelerator)
        {
            if (otherAccelerator is not ROCmAccelerator)
                throw new InvalidOperationException("Cannot disable peer access to non-ROCm accelerator");

            // Disable peer access (implementation would use hipDeviceDisablePeerAccess)
        }

        #endregion

        #region Kernel Estimation

        /// <summary>
        /// Estimates the maximum number of active groups per multiprocessor.
        /// </summary>
        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(
            Kernel kernel,
            int groupSize,
            int dynamicSharedMemorySizeInBytes)
        {
            // Conservative estimation based on resource constraints
            int maxThreadsPerMP = Device.MaxThreadsPerMultiprocessor;
            int maxGroupsFromThreads = maxThreadsPerMP / groupSize;

            // Estimate based on shared memory constraints
            long sharedMemPerMP = Device.MaxSharedMemoryPerGroup;
            long sharedMemPerGroup = Math.Max(dynamicSharedMemorySizeInBytes, WarpSize * sizeof(int));
            int maxGroupsFromSharedMem = (int)(sharedMemPerMP / sharedMemPerGroup);

            return Math.Min(maxGroupsFromThreads, maxGroupsFromSharedMem);
        }

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

            // Start with warp-aligned sizes and work up
            for (int groupSize = WarpSize; groupSize <= maxGroupSize; groupSize += WarpSize)
            {
                int sharedMemSize = computeSharedMemorySize(groupSize);
                if (sharedMemSize <= MaxSharedMemoryPerGroup)
                {
                    int activeGroups = EstimateMaxActiveGroupsPerMultiprocessorInternal(
                        kernel, groupSize, sharedMemSize);
                    if (activeGroups > 0)
                        return groupSize;
                }
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
            out int minGridSize)
        {
            return EstimateGroupSizeInternal(
                kernel,
                _ => dynamicSharedMemorySizeInBytes,
                maxGroupSize,
                out minGridSize);
        }

        #endregion

        #region Binding

        /// <summary>
        /// Called when the accelerator is bound to the current thread.
        /// </summary>
        protected override void OnBind()
        {
            // ROCm-specific binding logic
            // In a real implementation, this would set the current HIP device context
        }

        /// <summary>
        /// Called when the accelerator is unbound from the current thread.
        /// </summary>
        protected override void OnUnbind()
        {
            // ROCm-specific unbinding logic
            // In a real implementation, this would clear the current HIP device context
        }

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
            throw new NotSupportedException($"Extension {typeof(TExtension)} is not supported by ROCm accelerator");

        #endregion

        #region GPU Kernels

        /// <summary>
        /// Executes matrix multiplication using ROCBlas if available.
        /// </summary>
        /// <param name="a">Matrix A.</param>
        /// <param name="b">Matrix B.</param>
        /// <param name="c">Result matrix C.</param>
        /// <param name="m">Rows in A and C.</param>
        /// <param name="k">Columns in A, rows in B.</param>
        /// <param name="n">Columns in B and C.</param>
        /// <param name="stream">ROCm stream.</param>
        public unsafe void ExecuteMatMul(
            IntPtr a, IntPtr b, IntPtr c,
            int m, int k, int n,
            ROCmStream? stream = null)
        {
            var streamHandle = stream?.NativePtr ?? IntPtr.Zero;
            
            try
            {
                // Try to use ROCBlas for hardware acceleration
                ROCmNative.ExecuteROCBlasMatMul(
                    a.ToPointer(), b.ToPointer(), c.ToPointer(),
                    m, k, n, streamHandle);
            }
            catch (DllNotFoundException)
            {
                throw new NotSupportedException("ROCBlas library not found. Install ROCm with ROCBlas for optimal performance.");
            }
            catch (EntryPointNotFoundException)
            {
                throw new NotSupportedException("ROCBlas functions not found. Check ROCm installation.");
            }
        }

        #endregion

        #region Disposal

        /// <summary>
        /// Disposes this ROCm accelerator.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (disposing)
            {
                // Cleanup ROCm resources
                NativePtr = IntPtr.Zero;
            }
        }

        #endregion

        #region GPU Information

        /// <summary>
        /// Prints detailed GPU information.
        /// </summary>
        public void PrintGPUInformation()
        {
            Console.WriteLine($"ROCm Device Information:");
            Console.WriteLine($"  Name: {Name}");
            Console.WriteLine($"  Device ID: {Device.DeviceId}");
            Console.WriteLine($"  Architecture: {Device.Architecture}");
            Console.WriteLine($"  Compute Capability: {Device.ComputeCapability}");
            Console.WriteLine($"  Total Memory: {MemorySize / (1024 * 1024)} MB");
            Console.WriteLine($"  Compute Units: {NumMultiprocessors}");
            Console.WriteLine($"  Max Threads/CU: {Device.MaxThreadsPerMultiprocessor}");
            Console.WriteLine($"  Warp Size: {WarpSize}");
            Console.WriteLine($"  Max Shared Memory/Block: {MaxSharedMemoryPerGroup} bytes");
            Console.WriteLine($"  Clock Rate: {Device.ClockRate / 1000} MHz");
            Console.WriteLine($"  Memory Clock: {Device.MemoryClockRate / 1000} MHz");
            Console.WriteLine($"  Memory Bus Width: {Device.MemoryBusWidth} bits");
            Console.WriteLine($"  L2 Cache Size: {Device.L2CacheSize / 1024} KB");
            Console.WriteLine($"  PCI Bus: {Device.PciBusId:X2}:{Device.PciDeviceId:X2}.{Device.PciDomainId}");
            Console.WriteLine($"  Unified Addressing: {SupportsUnifiedMemory}");
            Console.WriteLine($"  Managed Memory: {SupportsManagedMemory}");
            Console.WriteLine($"  Cooperative Launch: {Device.SupportsCooperativeLaunch}");
            Console.WriteLine($"  Concurrent Kernels: {Device.SupportsConcurrentKernels}");
        }

        #endregion
    }
}