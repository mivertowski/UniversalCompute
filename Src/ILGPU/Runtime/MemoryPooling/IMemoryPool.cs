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
using System.Threading;
using System.Threading.Tasks;

namespace ILGPU.Runtime.MemoryPooling
{
    /// <summary>
    /// Represents a memory pool for efficient buffer allocation and reuse.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public interface IMemoryPool<T> : IDisposable where T : unmanaged
    {
        /// <summary>
        /// Gets the maximum pool size in bytes.
        /// </summary>
        long MaxPoolSizeBytes { get; }

        /// <summary>
        /// Gets the current pool utilization percentage.
        /// </summary>
        double UtilizationPercentage { get; }

        /// <summary>
        /// Gets a value indicating whether the pool is full.
        /// </summary>
        bool IsFull { get; }

        /// <summary>
        /// Rents a buffer from the pool with at least the specified minimum length.
        /// </summary>
        /// <param name="minLength">The minimum required length.</param>
        /// <returns>A pooled memory buffer.</returns>
        IPooledMemoryBuffer<T> Rent(long minLength);

        /// <summary>
        /// Asynchronously rents a buffer from the pool with at least the specified minimum length.
        /// </summary>
        /// <param name="minLength">The minimum required length.</param>
        /// <param name="ct">The cancellation token.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task<IPooledMemoryBuffer<T>> RentAsync(long minLength, CancellationToken ct = default);

        /// <summary>
        /// Returns a buffer to the pool.
        /// </summary>
        /// <param name="buffer">The buffer to return.</param>
        /// <param name="clearBuffer">Whether to clear the buffer contents.</param>
        void ReturnBuffer(IPooledMemoryBuffer<T> buffer, bool clearBuffer = false);

        /// <summary>
        /// Trims unused buffers from the pool.
        /// </summary>
        void Trim();

        /// <summary>
        /// Gets detailed statistics about the pool.
        /// </summary>
        /// <returns>Pool statistics.</returns>
        MemoryPoolStatistics GetStatistics();

        /// <summary>
        /// Sets the retention policy for the pool.
        /// </summary>
        /// <param name="policy">The retention policy.</param>
        void SetRetentionPolicy(PoolRetentionPolicy policy);
    }

    /// <summary>
    /// Represents a pooled memory buffer that automatically returns to the pool when disposed.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public interface IPooledMemoryBuffer<T> : ITypedMemoryBuffer<T> where T : unmanaged
    {
        /// <summary>
        /// Gets the pool that owns this buffer.
        /// </summary>
        IMemoryPool<T> Pool { get; }

        /// <summary>
        /// Gets the actual buffer length (may be larger than requested).
        /// </summary>
        long ActualLength { get; }

        /// <summary>
        /// Gets the underlying 1D memory buffer.
        /// </summary>
        MemoryBuffer1D<T, Stride1D.Dense> Buffer { get; }

        /// <summary>
        /// Gets a value indicating whether this buffer has been returned to the pool.
        /// </summary>
        bool IsReturned { get; }

        /// <summary>
        /// Manually returns the buffer to the pool without disposing.
        /// </summary>
        /// <param name="clearBuffer">Whether to clear the buffer contents.</param>
        void ReturnToPool(bool clearBuffer = false);
    }

    /// <summary>
    /// Defines retention policies for memory pools.
    /// </summary>
    public enum PoolRetentionPolicy
    {
        /// <summary>
        /// Keep buffers indefinitely until manually trimmed.
        /// </summary>
        KeepAll,

        /// <summary>
        /// Automatically trim unused buffers based on usage patterns.
        /// </summary>
        Adaptive,

        /// <summary>
        /// Aggressively trim buffers to minimize memory usage.
        /// </summary>
        Aggressive,

        /// <summary>
        /// Custom retention policy with user-defined parameters.
        /// </summary>
        Custom
    }

    /// <summary>
    /// Contains detailed statistics about a memory pool.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the <see cref="MemoryPoolStatistics"/> class.
    /// </remarks>
    public sealed class MemoryPoolStatistics(
        long totalPoolSizeBytes,
        long usedSizeBytes,
        long availableSizeBytes,
        int totalBuffers,
        int rentedBuffers,
        int availableBuffers,
        long totalAllocations,
        long totalDeallocations,
        double hitRatio,
        TimeSpan averageRentTime)
    {

        /// <summary>
        /// Gets the total pool size in bytes.
        /// </summary>
        public long TotalPoolSizeBytes { get; } = totalPoolSizeBytes;

        /// <summary>
        /// Gets the currently used size in bytes.
        /// </summary>
        public long UsedSizeBytes { get; } = usedSizeBytes;

        /// <summary>
        /// Gets the available size in bytes.
        /// </summary>
        public long AvailableSizeBytes { get; } = availableSizeBytes;

        /// <summary>
        /// Gets the total number of buffers in the pool.
        /// </summary>
        public int TotalBuffers { get; } = totalBuffers;

        /// <summary>
        /// Gets the number of currently rented buffers.
        /// </summary>
        public int RentedBuffers { get; } = rentedBuffers;

        /// <summary>
        /// Gets the number of available buffers.
        /// </summary>
        public int AvailableBuffers { get; } = availableBuffers;

        /// <summary>
        /// Gets the total number of allocations performed.
        /// </summary>
        public long TotalAllocations { get; } = totalAllocations;

        /// <summary>
        /// Gets the total number of deallocations performed.
        /// </summary>
        public long TotalDeallocations { get; } = totalDeallocations;

        /// <summary>
        /// Gets the cache hit ratio (0.0 to 1.0).
        /// </summary>
        public double HitRatio { get; } = hitRatio;

        /// <summary>
        /// Gets the average time buffers are rented.
        /// </summary>
        public TimeSpan AverageRentTime { get; } = averageRentTime;

        /// <summary>
        /// Gets the pool utilization percentage.
        /// </summary>
        public double UtilizationPercentage => TotalPoolSizeBytes > 0 ? 
            (double)UsedSizeBytes / TotalPoolSizeBytes * 100.0 : 0.0;
    }
}
