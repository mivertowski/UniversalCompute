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
using ILGPU.Intel.AMX.Native;
using ILGPU.Runtime;
using ILGPU.IR.Analyses;
using System;

namespace ILGPU.Intel.AMX
{
    /// <summary>
    /// Intel Advanced Matrix Extensions (AMX) accelerator for tile-based matrix operations.
    /// </summary>
    public sealed class AMXAccelerator : Accelerator
    {
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the AMXAccelerator class.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="device">The CPU device.</param>
        public AMXAccelerator(Context context, Device device)
            : base(context, device)
        {
            if (!AMXCapabilities.IsAMXSupported())
                throw new NotSupportedException("Intel AMX not supported on this processor");

            AMXCapabilities = AMXCapabilities.Query();
            TileConfiguration = AMXTileConfiguration.CreateDefault(AMXCapabilities);

            // Initialize AMX state
            AMXNative.InitializeAMX();
            ConfigureTiles();

            // Properties are inherited from the device parameter
        }

        /// <summary>
        /// Gets the AMX capabilities.
        /// </summary>
        public AMXCapabilities AMXCapabilities { get; }

        /// <summary>
        /// Gets the current tile configuration.
        /// </summary>
        public AMXTileConfiguration TileConfiguration { get; }



        #region Matrix Operations

        /// <summary>
        /// Performs matrix multiplication using AMX tiles.
        /// </summary>
        /// <param name="a">Matrix A (left operand).</param>
        /// <param name="b">Matrix B (right operand).</param>
        /// <param name="c">Matrix C (result).</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        public unsafe void MatrixMultiply(float* a, float* b, float* c, int m, int n, int k)
        {
            ThrowIfDisposed();
            
            // Ensure AMX is configured for FP32 operations
            if (TileConfiguration.DataType != AMXDataType.Float32)
            {
                ConfigureTilesForDataType(AMXDataType.Float32);
            }

            AMXOperations.MatrixMultiplyFP32(a, b, c, m, n, k, TileConfiguration);
        }

        /// <summary>
        /// Performs matrix multiplication using BF16 precision.
        /// </summary>
        /// <param name="a">Matrix A (BF16 data).</param>
        /// <param name="b">Matrix B (BF16 data).</param>
        /// <param name="c">Matrix C (FP32 result).</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        public unsafe void MatrixMultiplyBF16(ushort* a, ushort* b, float* c, int m, int n, int k)
        {
            ThrowIfDisposed();
            
            // Configure AMX for BF16 operations
            if (TileConfiguration.DataType != AMXDataType.BFloat16)
            {
                ConfigureTilesForDataType(AMXDataType.BFloat16);
            }

            AMXOperations.MatrixMultiplyBF16(a, b, c, m, n, k, TileConfiguration);
        }

        /// <summary>
        /// Performs matrix multiplication using INT8 precision.
        /// </summary>
        /// <param name="a">Matrix A (INT8 data).</param>
        /// <param name="b">Matrix B (INT8 data).</param>
        /// <param name="c">Matrix C (INT32 result).</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        public unsafe void MatrixMultiplyINT8(sbyte* a, sbyte* b, int* c, int m, int n, int k)
        {
            ThrowIfDisposed();
            
            // Configure AMX for INT8 operations
            if (TileConfiguration.DataType != AMXDataType.Int8)
            {
                ConfigureTilesForDataType(AMXDataType.Int8);
            }

            AMXOperations.MatrixMultiplyINT8(a, b, c, m, n, k, TileConfiguration);
        }

        #endregion

        #region Accelerator Implementation

        protected override AcceleratorStream CreateStreamInternal() => new AMXStream(this);

        protected override void SynchronizeInternal() =>
            // AMX operations are synchronous by nature
            System.Threading.Thread.MemoryBarrier();

        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize) => new AMXBuffer(this, length, elementSize);

        protected override Kernel LoadKernelInternal(CompiledKernel kernel) => new AMXKernel(this, kernel);

        protected override Kernel LoadAutoGroupedKernelInternal(
            CompiledKernel kernel,
            out KernelInfo? kernelInfo)
        {
            var allocaInfo = default(AllocaKindInformation);
            kernelInfo = new KernelInfo(0, 0, in allocaInfo, []);
            return LoadKernelInternal(kernel);
        }

        protected override Kernel LoadImplicitlyGroupedKernelInternal(
            CompiledKernel kernel,
            int customGroupSize,
            out KernelInfo? kernelInfo)
        {
            var allocaInfo = default(AllocaKindInformation);
            kernelInfo = new KernelInfo(0, 0, in allocaInfo, []);
            return LoadKernelInternal(kernel);
        }

        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(
            Kernel kernel,
            int groupSize,
            int dynamicSharedMemorySizeInBytes) =>
            // AMX doesn't have the concept of active groups like GPU accelerators
            Math.Max(1, NumMultiprocessors / groupSize);

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            Func<int, int> computeSharedMemorySize,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            return Math.Min(maxGroupSize, WarpSize);
        }

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            int dynamicSharedMemorySizeInBytes,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            return Math.Min(maxGroupSize, WarpSize);
        }

        protected override bool CanAccessPeerInternal(Accelerator otherAccelerator) => false;

        protected override void EnablePeerAccessInternal(Accelerator otherAccelerator) => throw new NotSupportedException("AMX does not support peer access");

        protected override void DisablePeerAccessInternal(Accelerator otherAccelerator) => throw new NotSupportedException("AMX does not support peer access");

        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(IntPtr pinned, long numElements) =>
            // AMX operates on CPU memory, so no special page locking needed
            new NullPageLockScope<T>(this, pinned, numElements);

        public override TExtension CreateExtension<TExtension, TExtensionProvider>(TExtensionProvider provider) => throw new NotSupportedException($"Extension {typeof(TExtension)} not supported by AMX accelerator");

        protected override void OnBind() =>
            // Configure AMX when bound
            ConfigureTiles();

        protected override void OnUnbind() =>
            // Release AMX configuration when unbound
            AMXNative.ReleaseAMX();

        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Release AMX resources
                    AMXNative.ReleaseAMX();
                }
                _disposed = true;
            }
        }

        #endregion

        #region Tile Management

        /// <summary>
        /// Loads data into an AMX tile.
        /// </summary>
        /// <param name="tileId">The tile ID (0-7).</param>
        /// <param name="data">Pointer to the data.</param>
        /// <param name="stride">The row stride in bytes.</param>
        public unsafe void LoadTile(int tileId, void* data, int stride)
        {
            ThrowIfDisposed();
            
            if (tileId < 0 || tileId >= AMXCapabilities.MaxTiles)
                throw new ArgumentOutOfRangeException(nameof(tileId), "Invalid tile ID");

            AMXNative.LoadTile(tileId, data, stride);
        }

        /// <summary>
        /// Stores data from an AMX tile.
        /// </summary>
        /// <param name="tileId">The tile ID (0-7).</param>
        /// <param name="data">Pointer to the destination.</param>
        /// <param name="stride">The row stride in bytes.</param>
        public unsafe void StoreTile(int tileId, void* data, int stride)
        {
            ThrowIfDisposed();
            
            if (tileId < 0 || tileId >= AMXCapabilities.MaxTiles)
                throw new ArgumentOutOfRangeException(nameof(tileId), "Invalid tile ID");

            AMXNative.StoreTile(tileId, data, stride);
        }

        /// <summary>
        /// Zeros an AMX tile.
        /// </summary>
        /// <param name="tileId">The tile ID (0-7).</param>
        public void ZeroTile(int tileId)
        {
            ThrowIfDisposed();
            
            if (tileId < 0 || tileId >= AMXCapabilities.MaxTiles)
                throw new ArgumentOutOfRangeException(nameof(tileId), "Invalid tile ID");

            AMXNative.ZeroTile(tileId);
        }

        #endregion

        #region Private Implementation

        private void ConfigureTiles() => AMXNative.ConfigureTiles(TileConfiguration);

        private void ConfigureTilesForDataType(AMXDataType dataType)
        {
            var newConfig = TileConfiguration.WithDataType(dataType);
            AMXNative.ConfigureTiles(newConfig);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(AMXAccelerator));
        }

        #endregion
    }

    /// <summary>
    /// AMX stream implementation.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the AMXStream class.
    /// </remarks>
    /// <param name="accelerator">The AMX accelerator.</param>
#pragma warning disable CA1711 // Identifiers should not have incorrect suffix
    public sealed class AMXStream(AMXAccelerator accelerator) : AcceleratorStream(accelerator)
#pragma warning restore CA1711 // Identifiers should not have incorrect suffix
    {
        private readonly AMXAccelerator _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));

        /// <summary>
        /// Synchronizes the stream.
        /// </summary>
        public override void Synchronize() =>
            // AMX operations are inherently synchronous
            System.Threading.Thread.MemoryBarrier();


        /// <summary>
        /// Adds a profiling marker for AMX operations.
        /// </summary>
        /// <returns>A profiling marker for timing measurements.</returns>
        protected override ProfilingMarker AddProfilingMarkerInternal() => new AMXProfilingMarker(_accelerator);

        /// <summary>
        /// Disposes the AMX stream.
        /// </summary>
        /// <param name="disposing">Whether disposing is in progress.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            // No resources to dispose for AMX streams
        }
    }

    /// <summary>
    /// AMX profiling marker implementation.
    /// </summary>
    internal sealed class AMXProfilingMarker : ProfilingMarker
    {
        private readonly DateTime _timestamp;

        /// <summary>
        /// Initializes a new instance of the AMXProfilingMarker class.
        /// </summary>
        /// <param name="accelerator">The AMX accelerator.</param>
        internal AMXProfilingMarker(Accelerator accelerator) : base(accelerator)
        {
            _timestamp = DateTime.UtcNow;
        }

        /// <summary>
        /// Synchronizes the profiling marker.
        /// </summary>
        public override void Synchronize()
        {
            // AMX operations are typically synchronous, no action needed
        }

        /// <summary>
        /// Measures elapsed time from another marker.
        /// </summary>
        /// <param name="marker">The starting marker.</param>
        /// <returns>The elapsed time.</returns>
        public override TimeSpan MeasureFrom(ProfilingMarker marker) => marker is AMXProfilingMarker amxMarker
                ? _timestamp - amxMarker._timestamp
                : throw new ArgumentException("Marker must be an AMX profiling marker", nameof(marker));

        /// <summary>
        /// Disposes the profiling marker.
        /// </summary>
        /// <param name="disposing">Whether disposing is in progress.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            // No resources to dispose
        }
    }
}