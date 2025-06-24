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

using ILGPU.Runtime;
using System;
using System.Threading.Tasks;

namespace ILGPU.Memory.Unified
{
    /// <summary>
    /// Represents a universal buffer that can work across all ILGPU accelerator types
    /// with optimized memory placement and transfer strategies.
    /// </summary>
    /// <typeparam name="T">The element type of the buffer.</typeparam>
    public interface IUniversalBuffer<T> : IDisposable
        where T : unmanaged
    {
        /// <summary>
        /// Gets the length of the buffer in elements.
        /// </summary>
        long Length { get; }

        /// <summary>
        /// Gets the size of the buffer in bytes.
        /// </summary>
        long SizeInBytes { get; }

        /// <summary>
        /// Gets the memory placement strategy used by this buffer.
        /// </summary>
        MemoryPlacement Placement { get; }

        /// <summary>
        /// Gets whether this buffer supports zero-copy access.
        /// </summary>
        bool SupportsZeroCopy { get; }

        /// <summary>
        /// Gets whether this buffer supports automatic memory migration.
        /// </summary>
        bool SupportsAutoMigration { get; }

        /// <summary>
        /// Gets the current location of the data (host, device, or both).
        /// </summary>
        DataLocation CurrentLocation { get; }

        /// <summary>
        /// Gets a view of this buffer that can be used in kernels.
        /// </summary>
        ArrayView<T> View { get; }

        /// <summary>
        /// Gets a 2D view of this buffer with the specified dimensions.
        /// </summary>
        /// <param name="width">The width of the 2D view.</param>
        /// <param name="height">The height of the 2D view.</param>
        /// <returns>A 2D array view.</returns>
        ArrayView2D<T, Stride2D.DenseX> As2DView(int width, int height);

        /// <summary>
        /// Gets a 3D view of this buffer with the specified dimensions.
        /// </summary>
        /// <param name="width">The width of the 3D view.</param>
        /// <param name="height">The height of the 3D view.</param>
        /// <param name="depth">The depth of the 3D view.</param>
        /// <returns>A 3D array view.</returns>
        ArrayView3D<T, Stride3D.DenseXY> As3DView(int width, int height, int depth);

        /// <summary>
        /// Copies data from a CPU array to this buffer.
        /// </summary>
        /// <param name="source">The source CPU array.</param>
        void CopyFromCPU(ReadOnlySpan<T> source);

        /// <summary>
        /// Copies data from this buffer to a CPU array.
        /// </summary>
        /// <param name="destination">The destination CPU array.</param>
        void CopyToCPU(Span<T> destination);

        /// <summary>
        /// Asynchronously copies data from a CPU array to this buffer.
        /// </summary>
        /// <param name="source">The source CPU array.</param>
        /// <returns>A task representing the asynchronous copy operation.</returns>
        Task CopyFromCPUAsync(ReadOnlyMemory<T> source);

        /// <summary>
        /// Asynchronously copies data from this buffer to a CPU array.
        /// </summary>
        /// <param name="destination">The destination CPU array.</param>
        /// <returns>A task representing the asynchronous copy operation.</returns>
        Task CopyToCPUAsync(Memory<T> destination);

        /// <summary>
        /// Copies data from another universal buffer to this buffer.
        /// </summary>
        /// <param name="source">The source buffer.</param>
        void CopyFrom(IUniversalBuffer<T> source);

        /// <summary>
        /// Asynchronously copies data from another universal buffer to this buffer.
        /// </summary>
        /// <param name="source">The source buffer.</param>
        /// <returns>A task representing the asynchronous copy operation.</returns>
        Task CopyFromAsync(IUniversalBuffer<T> source);

        /// <summary>
        /// Ensures the data is available on the specified accelerator.
        /// This may trigger data migration if necessary.
        /// </summary>
        /// <param name="accelerator">The target accelerator.</param>
        void EnsureAvailable(Accelerator accelerator);

        /// <summary>
        /// Asynchronously ensures the data is available on the specified accelerator.
        /// </summary>
        /// <param name="accelerator">The target accelerator.</param>
        /// <returns>A task representing the migration operation.</returns>
        Task EnsureAvailableAsync(Accelerator accelerator);

        /// <summary>
        /// Prefetches the data to the specified accelerator for better performance.
        /// This is a hint to the memory manager and may be ignored if not beneficial.
        /// </summary>
        /// <param name="accelerator">The target accelerator.</param>
        void Prefetch(Accelerator accelerator);

        /// <summary>
        /// Gets the native buffer for the specified accelerator.
        /// This allows direct access to underlying accelerator-specific buffers.
        /// </summary>
        /// <param name="accelerator">The target accelerator.</param>
        /// <returns>The native buffer, or null if not available.</returns>
        MemoryBuffer1D<T, Stride1D.Dense>? GetNativeBuffer(Accelerator accelerator);

        /// <summary>
        /// Invalidates cached copies of the data, forcing a refresh on next access.
        /// This is useful when the data has been modified externally.
        /// </summary>
        void InvalidateCache();

        /// <summary>
        /// Gets memory usage statistics for this buffer.
        /// </summary>
        /// <returns>Memory usage information.</returns>
        UniversalBufferStats GetStats();
    }

    /// <summary>
    /// Indicates the current location of buffer data.
    /// </summary>
    [Flags]
    public enum DataLocation
    {
        /// <summary>
        /// Data location is unknown or not tracked.
        /// </summary>
        Unknown = 0,

        /// <summary>
        /// Data is present in host (CPU) memory.
        /// </summary>
        Host = 1,

        /// <summary>
        /// Data is present in device (GPU/accelerator) memory.
        /// </summary>
        Device = 2,

        /// <summary>
        /// Data is present in both host and device memory (coherent).
        /// </summary>
        Both = Host | Device,

        /// <summary>
        /// Data is in unified memory accessible by both host and device.
        /// </summary>
        Unified = 4
    }

    /// <summary>
    /// Provides statistics about universal buffer memory usage.
    /// </summary>
    public readonly struct UniversalBufferStats
    {
        /// <summary>
        /// Gets the total memory allocated for this buffer across all locations.
        /// </summary>
        public long TotalAllocatedBytes { get; }

        /// <summary>
        /// Gets the number of times data has been migrated between locations.
        /// </summary>
        public int MigrationCount { get; }

        /// <summary>
        /// Gets the total time spent on data migrations in milliseconds.
        /// </summary>
        public double TotalMigrationTimeMs { get; }

        /// <summary>
        /// Gets the average bandwidth achieved during migrations in GB/s.
        /// </summary>
        public double AverageMigrationBandwidthGBps { get; }

        /// <summary>
        /// Gets the number of accelerators that have copies of this buffer.
        /// </summary>
        public int ActiveCopies { get; }

        /// <summary>
        /// Gets the current data location.
        /// </summary>
        public DataLocation CurrentLocation { get; }

        /// <summary>
        /// Initializes a new instance of the UniversalBufferStats struct.
        /// </summary>
        public UniversalBufferStats(
            long totalAllocatedBytes,
            int migrationCount,
            double totalMigrationTimeMs,
            double averageMigrationBandwidthGBps,
            int activeCopies,
            DataLocation currentLocation)
        {
            TotalAllocatedBytes = totalAllocatedBytes;
            MigrationCount = migrationCount;
            TotalMigrationTimeMs = totalMigrationTimeMs;
            AverageMigrationBandwidthGBps = averageMigrationBandwidthGBps;
            ActiveCopies = activeCopies;
            CurrentLocation = currentLocation;
        }
    }
}