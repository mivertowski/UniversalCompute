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
// Change License: Apache License, Version 2.0#if ENABLE_ONEAPI_ACCELERATOR
namespace ILGPU.Backends.OneAPI
{
    /// <summary>
    /// Intel OneAPI accelerator supporting various Intel hardware through SYCL/DPC++.
    /// </summary>
    public sealed class OneAPIAccelerator : Accelerator
    {
        private readonly IntPtr _device;
        private readonly IntPtr _context;
        private readonly IntPtr _queue;
        private readonly OneAPIDevice _deviceInfo;
        private readonly OneAPICapabilities _capabilities;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the OneAPIAccelerator class.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="device">The OneAPI device.</param>
        public OneAPIAccelerator(Context context, OneAPIDevice device)
            : base(context, device.Device)
        {
            _deviceInfo = device ?? throw new ArgumentNullException(nameof(device));
            
            // Initialize OneAPI/SYCL context
            _device = OneAPINative.CreateDevice(device.DeviceId, device.DeviceType);
            if (_device == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OneAPI device: {device.Name}");

            _context = OneAPINative.CreateContext(_device);
            if (_context == IntPtr.Zero)
            {
                OneAPINative.ReleaseDevice(_device);
                throw new InvalidOperationException("Failed to create OneAPI context");
            }

            _queue = OneAPINative.CreateQueue(_context, _device);
            if (_queue == IntPtr.Zero)
            {
                OneAPINative.ReleaseContext(_context);
                OneAPINative.ReleaseDevice(_device);
                throw new InvalidOperationException("Failed to create OneAPI queue");
            }

            // Query device capabilities
            _capabilities = QueryCapabilities();

            // Set accelerator properties based on device type
            Name = $"Intel OneAPI {device.Name}";
            InitializeAcceleratorProperties();
        }

        /// <summary>
        /// Gets the OneAPI device information.
        /// </summary>
        public OneAPIDevice DeviceInfo => _deviceInfo;

        /// <summary>
        /// Gets the OneAPI capabilities.
        /// </summary>
        public OneAPICapabilities Capabilities => _capabilities;

        /// <summary>
        /// Gets whether this accelerator supports unified memory.
        /// </summary>
        public override bool SupportsUnifiedMemory => _capabilities.SupportsUSM;

        /// <summary>
        /// Gets the memory information for this accelerator.
        /// </summary>
        public override MemoryInfo MemoryInfo => new MemoryInfo(
            _capabilities.GlobalMemorySize - _capabilities.GlobalMemoryUsed,
            _capabilities.GlobalMemorySize
        );

        #region Kernel Compilation and Execution

        /// <summary>
        /// Compiles a kernel for OneAPI execution.
        /// </summary>
        /// <param name="source">The kernel source code.</param>
        /// <param name="kernelName">The kernel name.</param>
        /// <param name="options">Compilation options.</param>
        /// <returns>The compiled kernel.</returns>
        public OneAPIKernel CompileKernel(string source, string kernelName, OneAPICompilationOptions options = null)
        {
            ThrowIfDisposed();

            options ??= OneAPICompilationOptions.Default;
            
            // Compile SYCL/DPC++ kernel
            var program = OneAPINative.CompileKernel(_context, _device, source, options.BuildOptions);
            if (program == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to compile kernel: {kernelName}");

            return new OneAPIKernel(this, program, kernelName);
        }

        /// <summary>
        /// Launches a kernel on the OneAPI device.
        /// </summary>
        /// <param name="kernel">The kernel to launch.</param>
        /// <param name="gridSize">The grid size.</param>
        /// <param name="blockSize">The block size.</param>
        /// <param name="args">The kernel arguments.</param>
        public void LaunchKernel(OneAPIKernel kernel, Index3D gridSize, Index3D blockSize, params object[] args)
        {
            ThrowIfDisposed();

            if (kernel == null)
                throw new ArgumentNullException(nameof(kernel));

            // Convert grid/block sizes to OneAPI nd_range
            var globalSize = new Index3D(
                gridSize.X * blockSize.X,
                gridSize.Y * blockSize.Y,
                gridSize.Z * blockSize.Z
            );

            // Set kernel arguments
            for (int i = 0; i < args.Length; i++)
            {
                kernel.SetArgument(i, args[i]);
            }

            // Enqueue kernel
            kernel.Launch(_queue, globalSize, blockSize);
        }

        #endregion

        #region Memory Management

        /// <summary>
        /// Allocates unified shared memory (USM) accessible by both host and device.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="length">The number of elements.</param>
        /// <returns>The allocated USM buffer.</returns>
        public OneAPIUSMBuffer<T> AllocateUSM<T>(long length) where T : unmanaged
        {
            ThrowIfDisposed();

            if (!_capabilities.SupportsUSM)
                throw new NotSupportedException("Unified Shared Memory not supported on this device");

            return new OneAPIUSMBuffer<T>(this, _context, _device, length);
        }

        #endregion

        #region Accelerator Implementation

        protected override AcceleratorStream CreateStreamInternal() => new OneAPIStream(this, _queue);

        protected override void SynchronizeInternal()
        {
            OneAPINative.QueueWait(_queue);
        }

        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize)
        {
            return new OneAPIBuffer(this, _context, _device, _queue, length, elementSize);
        }

        protected override Kernel LoadKernelInternal(CompiledKernel compiledKernel)
        {
            // Convert ILGPU kernel to OneAPI kernel
            var source = GenerateSYCLKernel(compiledKernel);
            var oneapiKernel = CompileKernel(source, compiledKernel.EntryName);
            return new OneAPIKernelAdapter(this, compiledKernel, oneapiKernel);
        }

        protected override Kernel LoadAutoGroupedKernelInternal(
            CompiledKernel compiledKernel,
            out KernelInfo? kernelInfo)
        {
            kernelInfo = new KernelInfo(0, 0, _capabilities.MaxWorkGroupSize);
            return LoadKernelInternal(compiledKernel);
        }

        protected override Kernel LoadImplicitlyGroupedKernelInternal(
            CompiledKernel compiledKernel,
            int customGroupSize,
            out KernelInfo? kernelInfo)
        {
            kernelInfo = new KernelInfo(0, 0, customGroupSize);
            return LoadKernelInternal(compiledKernel);
        }

        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(
            Kernel kernel,
            int groupSize,
            int dynamicSharedMemorySizeInBytes)
        {
            // Estimate based on device resources
            var maxThreadsPerEU = _capabilities.MaxThreadsPerEU;
            var numEUs = _capabilities.NumExecutionUnits;
            var maxActiveGroups = (maxThreadsPerEU * numEUs) / groupSize;
            
            return Math.Max(1, maxActiveGroups);
        }

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            Func<int, int> computeSharedMemorySize,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            // Use preferred work group size multiple for optimal performance
            return Math.Min(maxGroupSize, _capabilities.PreferredWorkGroupSizeMultiple);
        }

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            int dynamicSharedMemorySizeInBytes,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            return Math.Min(maxGroupSize, _capabilities.PreferredWorkGroupSizeMultiple);
        }

        protected override bool CanAccessPeerInternal(Accelerator otherAccelerator)
        {
            if (otherAccelerator is OneAPIAccelerator otherOneAPI)
            {
                // Check if devices can share memory
                return OneAPINative.CanShareMemory(_device, otherOneAPI._device);
            }
            return false;
        }

        protected override void EnablePeerAccessInternal(Accelerator otherAccelerator)
        {
            if (otherAccelerator is OneAPIAccelerator otherOneAPI)
            {
                OneAPINative.EnablePeerAccess(_context, otherOneAPI._context);
            }
            else
            {
                throw new NotSupportedException("Peer access only supported between OneAPI devices");
            }
        }

        protected override void DisablePeerAccessInternal(Accelerator otherAccelerator)
        {
            if (otherAccelerator is OneAPIAccelerator otherOneAPI)
            {
                OneAPINative.DisablePeerAccess(_context, otherOneAPI._context);
            }
        }

        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(IntPtr pinned, long numElements)
        {
            // OneAPI supports pinned memory through USM
            return new PageLockScope<T>(this, pinned, numElements);
        }

        protected override TExtension CreateExtension<TExtension, TExtensionProvider>(TExtensionProvider provider)
        {
            throw new NotSupportedException($"Extension {typeof(TExtension)} not supported by OneAPI accelerator");
        }

        protected override void OnBind()
        {
            // Set OneAPI device as current
            OneAPINative.SetCurrentDevice(_device);
        }

        protected override void OnUnbind()
        {
            // No specific unbind action needed for OneAPI
        }

        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Release OneAPI resources in reverse order
                    if (_queue != IntPtr.Zero)
                        OneAPINative.ReleaseQueue(_queue);
                    if (_context != IntPtr.Zero)
                        OneAPINative.ReleaseContext(_context);
                    if (_device != IntPtr.Zero)
                        OneAPINative.ReleaseDevice(_device);
                }
                _disposed = true;
            }
        }

        #endregion

        #region Private Helpers

        private void InitializeAcceleratorProperties()
        {
            MaxGridSize = new Index3D(
                _capabilities.MaxWorkItemSizes[0],
                _capabilities.MaxWorkItemSizes[1],
                _capabilities.MaxWorkItemSizes[2]
            );
            
            MaxGroupSize = new Index3D(
                _capabilities.MaxWorkGroupSize,
                _capabilities.MaxWorkGroupSize,
                _capabilities.MaxWorkGroupSize
            );
            
            WarpSize = _capabilities.SubGroupSize;
            NumMultiprocessors = _capabilities.NumExecutionUnits;
            MaxSharedMemoryPerMultiprocessor = _capabilities.LocalMemorySize;
            MaxConstantMemory = _capabilities.MaxConstantBufferSize;
            MaxMemoryBandwidth = _capabilities.GlobalMemoryBandwidth;
        }

        private OneAPICapabilities QueryCapabilities()
        {
            var caps = new OneAPICapabilities();
            
            // Query device capabilities
            caps.DeviceType = _deviceInfo.DeviceType;
            caps.MaxComputeUnits = OneAPINative.GetDeviceInfo<int>(_device, OneAPIDeviceInfo.MaxComputeUnits);
            caps.MaxWorkGroupSize = OneAPINative.GetDeviceInfo<int>(_device, OneAPIDeviceInfo.MaxWorkGroupSize);
            caps.MaxWorkItemSizes = OneAPINative.GetDeviceInfo<long[]>(_device, OneAPIDeviceInfo.MaxWorkItemSizes);
            caps.GlobalMemorySize = OneAPINative.GetDeviceInfo<long>(_device, OneAPIDeviceInfo.GlobalMemSize);
            caps.LocalMemorySize = OneAPINative.GetDeviceInfo<long>(_device, OneAPIDeviceInfo.LocalMemSize);
            caps.MaxConstantBufferSize = OneAPINative.GetDeviceInfo<long>(_device, OneAPIDeviceInfo.MaxConstantBufferSize);
            caps.GlobalMemoryBandwidth = OneAPINative.GetDeviceInfo<long>(_device, OneAPIDeviceInfo.GlobalMemCacheBandwidth);
            
            // Query Intel-specific extensions
            caps.SupportsUSM = OneAPINative.SupportsUSM(_device);
            caps.SupportsFP16 = OneAPINative.SupportsFP16(_device);
            caps.SupportsFP64 = OneAPINative.SupportsFP64(_device);
            caps.SupportsSubgroups = OneAPINative.SupportsSubgroups(_device);
            caps.SubGroupSize = OneAPINative.GetSubgroupSize(_device);
            caps.NumExecutionUnits = OneAPINative.GetNumExecutionUnits(_device);
            caps.MaxThreadsPerEU = OneAPINative.GetMaxThreadsPerEU(_device);
            caps.PreferredWorkGroupSizeMultiple = OneAPINative.GetPreferredWorkGroupSizeMultiple(_device);
            
            return caps;
        }

        private string GenerateSYCLKernel(CompiledKernel compiledKernel)
        {
            // This would translate ILGPU IR to SYCL/DPC++ code
            // For now, return a placeholder
            throw new NotImplementedException("ILGPU to SYCL translation not implemented");
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(OneAPIAccelerator));
        }

        #endregion

        #region Static Device Enumeration

        /// <summary>
        /// Gets all available OneAPI devices.
        /// </summary>
        /// <returns>A collection of available OneAPI devices.</returns>
        public static IEnumerable<OneAPIDevice> GetDevices()
        {
            var platforms = OneAPINative.GetPlatforms();
            
            foreach (var platform in platforms)
            {
                var devices = OneAPINative.GetDevices(platform);
                foreach (var device in devices)
                {
                    yield return new OneAPIDevice(device, platform);
                }
            }
        }

        /// <summary>
        /// Gets OneAPI devices of a specific type.
        /// </summary>
        /// <param name="deviceType">The device type to filter by.</param>
        /// <returns>A collection of OneAPI devices of the specified type.</returns>
        public static IEnumerable<OneAPIDevice> GetDevices(OneAPIDeviceType deviceType)
        {
            return GetDevices().Where(d => d.DeviceType == deviceType);
        }

        #endregion
    }

    /// <summary>
    /// OneAPI stream implementation.
    /// </summary>
    public sealed class OneAPIStream : AcceleratorStream
    {
        private readonly IntPtr _queue;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the OneAPIStream class.
        /// </summary>
        /// <param name="accelerator">The OneAPI accelerator.</param>
        /// <param name="queue">The native queue handle.</param>
        public OneAPIStream(OneAPIAccelerator accelerator, IntPtr queue)
            : base(accelerator)
        {
            _queue = queue;
        }

        /// <summary>
        /// Synchronizes the stream.
        /// </summary>
        public override void Synchronize()
        {
            OneAPINative.QueueWait(_queue);
        }

        /// <summary>
        /// Synchronizes the stream asynchronously.
        /// </summary>
        /// <param name="cancellationToken">A cancellation token.</param>
        /// <returns>A task representing the synchronization.</returns>
        public override async Task SynchronizeAsync(CancellationToken cancellationToken = default)
        {
            await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                Synchronize();
            }, cancellationToken);
        }

        /// <summary>
        /// Disposes the OneAPI stream.
        /// </summary>
        /// <param name="disposing">Whether disposing is in progress.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (!_disposed)
            {
                // Stream uses shared queue, don't release it here
                _disposed = true;
            }
        }
    }
}
#endif