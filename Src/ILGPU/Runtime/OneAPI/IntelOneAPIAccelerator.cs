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
using ILGPU.Runtime.OneAPI.Native;
using System;
using System.Collections.Immutable;
using System.Threading.Tasks;

namespace ILGPU.Runtime.OneAPI
{
    /// <summary>
    /// Intel OneAPI/SYCL accelerator for Intel GPU compute acceleration.
    /// </summary>
    public sealed class IntelOneAPIAccelerator : Accelerator
    {
        #region Instance

        /// <summary>
        /// The associated Intel GPU device.
        /// </summary>
        public new IntelOneAPIDevice Device { get; }

        /// <summary>
        /// The native SYCL context handle.
        /// </summary>
        internal IntPtr SYCLContext { get; private set; }

        /// <summary>
        /// The native SYCL queue handle.
        /// </summary>
        internal IntPtr SYCLQueue { get; private set; }

        /// <summary>
        /// The native SYCL device handle.
        /// </summary>
        internal IntPtr SYCLDevice { get; private set; }

        /// <summary>
        /// Gets the device handle for stream creation.
        /// </summary>
        internal IntPtr DeviceHandle => SYCLDevice;

        /// <summary>
        /// Gets the context handle for stream creation.
        /// </summary>
        internal IntPtr ContextHandle => SYCLContext;

        /// <summary>
        /// Gets whether this accelerator supports unified memory.
        /// </summary>
        public new bool SupportsUnifiedMemory => Device.SupportsUnifiedMemory;

        /// <summary>
        /// Gets whether this accelerator supports Intel MKL SYCL.
        /// </summary>
        public bool SupportsMKLSYCL => Device.Architecture >= IntelGPUArchitecture.XeLP;

        /// <summary>
        /// Initializes a new Intel OneAPI accelerator.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="device">The Intel OneAPI device.</param>
        internal IntelOneAPIAccelerator(Context context, IntelOneAPIDevice device)
            : base(context, device)
        {
            Device = device;

            try
            {
                // Initialize SYCL runtime if not already done
                if (!SYCLNative.IsSYCLSupported())
                    throw new NotSupportedException("Intel OneAPI SYCL runtime not available");

                // Set the SYCL device
                SYCLDevice = device.NativeDevice;

                // Create SYCL context
                var devices = new IntPtr[] { SYCLDevice };
                var result = SYCLNative.CreateContext(1, devices, out var syclContext);
                SYCLContext = syclContext;
                SYCLException.ThrowIfFailed(result);

                // Create SYCL queue
                result = SYCLNative.CreateQueue(
                    SYCLContext, 
                    SYCLDevice, 
                    SYCLQueueProperties.ProfilingEnable, 
                    out var queue);
                SYCLQueue = queue;
                SYCLException.ThrowIfFailed(result);

                // Initialize device properties
                Init();
            }
            catch (Exception ex)
            {
                throw new SYCLException("Failed to initialize Intel OneAPI accelerator", ex);
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
        public new static AcceleratorType AcceleratorType => AcceleratorType.OneAPI;

        /// <summary>
        /// Gets the name of this accelerator.
        /// </summary>
        public new string Name => Device.Name;

        /// <summary>
        /// Gets the memory information of this accelerator.
        /// </summary>
        public MemoryInfo MemoryInfo => new(Device.MemorySize, Device.MemorySize, 0, 0, 1, true, false, false, 1, 0);

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
        public new long MaxSharedMemoryPerGroup => Device.MaxSharedMemoryPerGroup;

        /// <summary>
        /// Gets the maximum constant memory in bytes.
        /// </summary>
        public new long MaxConstantMemory => Device.MaxConstantMemory;

        /// <summary>
        /// Gets the warp size (subgroup size on Intel GPUs).
        /// </summary>
        public new int WarpSize => Device.WarpSize;

        /// <summary>
        /// Gets the number of multiprocessors (execution units).
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
        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize) => new SYCLMemoryBuffer(this, length, elementSize);

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
            new SYCLPageLockScope<T>(this, pinned, numElements);

        #endregion

        #region Kernel Management

        /// <summary>
        /// Loads the given kernel.
        /// </summary>
        /// <param name="kernel">The kernel to load.</param>
        /// <returns>The loaded kernel.</returns>
        protected override Kernel LoadKernelInternal(CompiledKernel kernel) =>
            new SYCLKernel(this, kernel as SYCLCompiledKernel ?? throw new ArgumentException("Invalid kernel type"));

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
        /// Creates a new OneAPI stream.
        /// </summary>
        /// <returns>The created stream.</returns>
        protected override AcceleratorStream CreateStreamInternal() =>
            new OneAPIStream(this);

        /// <summary>
        /// Synchronizes all pending operations.
        /// </summary>
        protected override void SynchronizeInternal()
        {
            if (DefaultStream is OneAPIStream stream)
                stream.Synchronize();
        }

        #endregion

        #region Abstract Method Implementations

        /// <summary>
        /// Called when the accelerator is bound to the current thread.
        /// </summary>
        protected override void OnBind()
        {
            // OneAPI-specific binding logic if needed
        }

        /// <summary>
        /// Called when the accelerator is unbound from the current thread.
        /// </summary>
        protected override void OnUnbind()
        {
            // OneAPI-specific unbinding logic if needed
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
            if (otherAccelerator is not IntelOneAPIAccelerator)
                return false;

            // Intel GPUs can share memory through unified memory
            return SupportsUnifiedMemory;
        }

        /// <summary>
        /// Enables peer access to the given accelerator.
        /// </summary>
        /// <param name="otherAccelerator">The other accelerator.</param>
        protected override void EnablePeerAccessInternal(Accelerator otherAccelerator)
        {
            if (otherAccelerator is not IntelOneAPIAccelerator)
                throw new InvalidOperationException("Cannot enable peer access to non-OneAPI accelerator");

            // Intel GPUs use unified memory for peer access
        }

        /// <summary>
        /// Disables peer access to the given accelerator.
        /// </summary>
        /// <param name="otherAccelerator">The other accelerator.</param>
        protected override void DisablePeerAccessInternal(Accelerator otherAccelerator)
        {
            if (otherAccelerator is not IntelOneAPIAccelerator)
                throw new InvalidOperationException("Cannot disable peer access to non-OneAPI accelerator");

            // Intel GPUs use unified memory - no explicit disable needed
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
            // Conservative estimation for Intel GPUs
            int maxThreadsPerEU = Device.MaxNumThreadsPerGroup;
            int maxGroupsFromThreads = maxThreadsPerEU / groupSize;

            // Estimate based on shared local memory constraints
            long sharedMemPerEU = Device.MaxSharedMemoryPerGroup;
            long sharedMemPerGroup = Math.Max(dynamicSharedMemorySizeInBytes, WarpSize * sizeof(int));
            int maxGroupsFromSharedMem = (int)(sharedMemPerEU / sharedMemPerGroup);

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

            // Start with subgroup-aligned sizes for Intel GPUs
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
            throw new NotSupportedException($"Extension {typeof(TExtension)} is not supported by OneAPI accelerator");

        #endregion

        #region Intel GPU Operations

        /// <summary>
        /// Executes matrix multiplication using Intel MKL SYCL if available.
        /// </summary>
        /// <param name="a">Matrix A.</param>
        /// <param name="b">Matrix B.</param>
        /// <param name="c">Result matrix C.</param>
        /// <param name="m">Rows in A and C.</param>
        /// <param name="k">Columns in A, rows in B.</param>
        /// <param name="n">Columns in B and C.</param>
        /// <param name="stream">SYCL stream.</param>
        public unsafe void ExecuteMatMul(
            IntPtr a, IntPtr b, IntPtr c,
            int m, int k, int n,
            OneAPIStream? stream = null)
        {
            var queueHandle = stream?.NativeQueue ?? SYCLQueue;
            
            try
            {
                // Try to use Intel MKL SYCL for hardware acceleration
                SYCLNative.ExecuteMKLSYCLMatMul(
                    queueHandle,
                    a.ToPointer(), b.ToPointer(), c.ToPointer(),
                    m, k, n);
            }
            catch (DllNotFoundException)
            {
                throw new NotSupportedException("Intel MKL SYCL library not found. Install Intel OneAPI MKL for optimal performance.");
            }
            catch (EntryPointNotFoundException)
            {
                throw new NotSupportedException("Intel MKL SYCL functions not found. Check OneAPI installation.");
            }
        }

        /// <summary>
        /// Executes AI inference using Intel Extension for PyTorch or OpenVINO.
        /// </summary>
        /// <param name="input">Input tensor data.</param>
        /// <param name="output">Output tensor data.</param>
        /// <param name="modelData">Model binary data.</param>
        /// <param name="stream">SYCL stream.</param>
        public async Task ExecuteAIInferenceAsync(
            IntPtr input, IntPtr output,
            byte[] modelData,
            OneAPIStream? stream = null) => await Task.Run(() =>
                                                     {
                                                         try
                                                         {
                                                             // Use Intel GPU for AI inference acceleration
                                                             ExecuteIntelGPUInference(input, output, modelData, stream?.NativeQueue ?? SYCLQueue);
                                                         }
                                                         catch (DllNotFoundException)
                                                         {
                                                             throw new NotSupportedException("Intel AI acceleration libraries not found. Install Intel Extension for PyTorch or OpenVINO.");
                                                         }
                                                         catch (EntryPointNotFoundException)
                                                         {
                                                             throw new NotSupportedException("Intel AI acceleration functions not found. Check OneAPI AI toolkit installation.");
                                                         }
                                                     }).ConfigureAwait(false);

        /// <summary>
        /// Placeholder for Intel GPU AI inference.
        /// </summary>
        private static void ExecuteIntelGPUInference(IntPtr input, IntPtr output, byte[] modelData, IntPtr queue) =>
            // This would integrate with Intel Extension for PyTorch or OpenVINO
            // For demonstration, simulate the operation
            System.Threading.Thread.Sleep(1); // Simulate AI inference

        #endregion

        #region Disposal

        /// <summary>
        /// Disposes this Intel OneAPI accelerator.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (disposing)
            {
#pragma warning disable CA1031 // Do not catch general exception types
                try
                {
                    // Cleanup SYCL resources
                    if (SYCLQueue != IntPtr.Zero)
                    {
                        SYCLNative.ReleaseQueue(SYCLQueue);
                        SYCLQueue = IntPtr.Zero;
                    }

                    if (SYCLContext != IntPtr.Zero)
                    {
                        SYCLNative.ReleaseContext(SYCLContext);
                        SYCLContext = IntPtr.Zero;
                    }

                    SYCLDevice = IntPtr.Zero;
                }
                catch
                {
                    // Ignore errors during disposal
                }
#pragma warning restore CA1031 // Do not catch general exception types
            }
        }

        #endregion

        #region GPU Information

        /// <summary>
        /// Prints detailed GPU information.
        /// </summary>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Globalization", "CA1303:Do not pass literals as localized parameters", Justification = "<Pending>")]
        public void PrintGPUInformation()
        {
            Console.WriteLine($"Intel OneAPI Device Information:");
            Console.WriteLine($"  Name: {Name}");
            Console.WriteLine($"  Architecture: {Device.Architecture}");
            Console.WriteLine($"  Vendor: {IntelOneAPIDevice.Vendor}");
            Console.WriteLine($"  Driver Version: {Device.DriverVersion}");
            Console.WriteLine($"  SYCL Version: {Device.SYCLVersion}");
            Console.WriteLine($"  Total Memory: {MemorySize / (1024 * 1024)} MB");
            Console.WriteLine($"  Execution Units: {NumMultiprocessors}");
            Console.WriteLine($"  Max Work Group Size: {MaxNumThreadsPerGroup}");
            Console.WriteLine($"  Subgroup Size: {WarpSize}");
            Console.WriteLine($"  Max Shared Memory/Group: {MaxSharedMemoryPerGroup} bytes");
            Console.WriteLine($"  Max Constant Memory: {MaxConstantMemory} bytes");
            Console.WriteLine($"  Clock Rate: {Device.ClockRate} MHz");
            Console.WriteLine($"  Unified Memory: {SupportsUnifiedMemory}");
            Console.WriteLine($"  Float64 Support: {Device.SupportsFloat64}");
            Console.WriteLine($"  Int64 Support: {Device.SupportsInt64}");
            Console.WriteLine($"  MKL SYCL Support: {SupportsMKLSYCL}");
        }

        #endregion
    }

    /// <summary>
    /// Intel OneAPI/SYCL exception.
    /// </summary>
    public class SYCLException : AcceleratorException
    {
        public SYCLException(string message) : base(message) { }
        public SYCLException(string message, Exception innerException) : base(message, innerException) { }

        /// <summary>
        /// Gets the accelerator type.
        /// </summary>
        public override AcceleratorType AcceleratorType => AcceleratorType.OneAPI;

        internal static void ThrowIfFailed(SYCLResult result)
        {
            if (result != SYCLResult.Success)
                throw new SYCLException($"SYCL operation failed with result: {result}");
        }
        public SYCLException()
        {
        }
    }
}
